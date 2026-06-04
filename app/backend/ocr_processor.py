"""
OCR Pipeline Module - Image Processing and Text Extraction

Uses PaddleOCR as the primary OCR engine and provides an improved
preprocessing pipeline (resize → denoise → contrast enhancement → deskew → OCR).

Also includes a simple OpenCV-based table detection helper and hooks for
layout detection if pp-structure/advanced Paddle modules are available.
"""

import cv2
from pathlib import Path
import logging
import json
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger("OCRProcessor")

# PaddleOCR optional import; raise informative error when used if missing
try:
    from paddleocr import PaddleOCR
except Exception:  # pragma: no cover - optional dependency
    PaddleOCR = None


class ImagePreprocessor:
    """Improved image preprocessing utilities."""

    @staticmethod
    def grayscale(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    @staticmethod
    def resize_image(image: np.ndarray, width: int = 1600) -> np.ndarray:
        h, w = image.shape[:2]
        if w <= width:
            return image
        aspect = w / h
        height = int(width / aspect)
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def denoise_fastNlMeans(image: np.ndarray, h: float = 10) -> np.ndarray:
        if len(image.shape) == 2:
            return cv2.fastNlMeansDenoising(image, None, h, 7, 21)
        else:
            return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)

    @staticmethod
    def denoise_bilateral(image: np.ndarray, diameter: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)

    @staticmethod
    def contrast_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        gray = ImagePreprocessor.grayscale(image)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(gray)

    @staticmethod
    def deskew(image: np.ndarray) -> np.ndarray:
        gray = ImagePreprocessor.grayscale(image)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        if coords.size == 0:
            return image
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        logger.debug(f"Deskew applied: angle={angle:.2f}")
        return rotated

    @staticmethod
    def threshold_otsu(image: np.ndarray) -> np.ndarray:
        gray = ImagePreprocessor.grayscale(image)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    @staticmethod
    def sharpen_unsharp_mask(image: np.ndarray, strength: float = 1.5) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
        sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    @staticmethod
    def invert_image(image: np.ndarray) -> np.ndarray:
        return cv2.bitwise_not(image)


class PaddleOCREngine:
    """Primary OCR engine using PaddleOCR."""

    def __init__(self, language: str = 'en', use_gpu: bool = False):
        if PaddleOCR is None:
            raise RuntimeError("PaddleOCR not available. Install with: pip install paddleocr paddlepaddle")
        self.language = language
        self.use_gpu = use_gpu
        self._ocr = None

    def _get_ocr(self):
        if self._ocr is not None:
            return self._ocr
        kwargs = {'use_angle_cls': True, 'lang': self._map_language(self.language)}
        if self.use_gpu:
            kwargs['use_gpu'] = True
        self._ocr = PaddleOCR(**kwargs)
        logger.info(f"PaddleOCR initialized (lang={self.language}, gpu={self.use_gpu})")
        return self._ocr

    def _map_language(self, code: str) -> str:
        """Map various UI/language codes to PaddleOCR language identifiers."""
        if not code:
            return 'en'
        code = code.lower()
        # Common mappings
        mapping = {
            'eng': 'en',
            'en': 'en',
            'eng+spa': 'en',
            'spa': 'es',
            'es': 'es',
            'fra': 'fr',
            'fr': 'fr',
            'deu': 'german',
            'de': 'german',
            'chi_sim': 'ch',
            'chi_tra': 'ch',
        }
        # support composite codes: take first
        if '+' in code:
            code = code.split('+')[0]
        return mapping.get(code, 'en')

    def extract_with_confidence(self, image: np.ndarray) -> Dict:
        ocr = self._get_ocr()
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        results = ocr.ocr(image, cls=True)
        lines = results[0] if results else []
        texts = []
        confidences = []
        for line in lines:
            if len(line) < 2 or not line[1]:
                continue
            text, score = line[1][0], line[1][1]
            if text:
                texts.append(text)
            if isinstance(score, (int, float)):
                confidences.append(score)
        text_output = " ".join(texts).strip()
        avg_confidence = (sum(confidences) / len(confidences)) * 100 if confidences else 0
        return {
            'text': text_output,
            'confidence': avg_confidence,
            'word_count': len(text_output.split())
        }

    def detect_layout(self, image: np.ndarray) -> List[Dict]:
        """Return OCR boxes as a simple layout representation."""
        try:
            ocr = self._get_ocr()
            results = ocr.ocr(image, cls=True)
            boxes = []
            for line in (results[0] if results else []):
                if len(line) >= 2 and isinstance(line[0], list):
                    box = line[0]
                    text = line[1][0] if line[1] else ''
                    boxes.append({'box': box, 'text': text})
            return boxes
        except Exception as e:
            logger.debug(f"Layout detection fallback: {e}")
            return []


class OCRPipeline:
    """Orchestrates preprocessing and PaddleOCR extraction."""

    def __init__(
        self,
        language: str = 'en',
        paddle_use_gpu: bool = False,
        enable_table_detection: bool = True,
        supported_formats: Optional[set] = None,
    ):
        self.preprocessor = ImagePreprocessor()
        self.ocr_engine = PaddleOCREngine(language=language, use_gpu=paddle_use_gpu)
        self.supported_formats = supported_formats or {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.pdf'}
        self.enable_table_detection = enable_table_detection

    def load_image(self, image_path: str) -> np.ndarray:
        path = Path(image_path)
        if not path.exists():
            raise ValueError(f"Image file not found: {image_path}")
        if path.suffix.lower() not in self.supported_formats and path.suffix.lower() != '.pdf':
            raise ValueError(f"Unsupported image format: {path.suffix}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        logger.info(f"Loaded image: {path.name} ({image.shape[0]}x{image.shape[1]})")
        return image

    def _detect_tables_cv(self, image: np.ndarray) -> List[Dict]:
        """Detect table regions using morphological operations.
        Returns list of {'bbox': (x,y,w,h), 'image': crop}
        """
        gray = ImagePreprocessor.grayscale(image)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)
        horizontal = thresh.copy()
        vertical = thresh.copy()
        cols = horizontal.shape[1]
        horizontal_size = max(1, cols // 30)
        horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
        horizontal = cv2.erode(horizontal, horizontal_structure)
        horizontal = cv2.dilate(horizontal, horizontal_structure)

        rows = vertical.shape[0]
        vertical_size = max(1, rows // 30)
        vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
        vertical = cv2.erode(vertical, vertical_structure)
        vertical = cv2.dilate(vertical, vertical_structure)

        mask = cv2.add(horizontal, vertical)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tables = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 50 and h > 50:
                crop = image[y:y+h, x:x+w]
                tables.append({'bbox': (x, y, w, h), 'image': crop})
        logger.debug(f"Detected {len(tables)} table(s) via CV heuristic")
        return tables

    def process(self, image_path_or_array, preprocessing_steps: Optional[List[str]] = None) -> Dict:
        try:
            if isinstance(image_path_or_array, str):
                image = self.load_image(image_path_or_array)
            else:
                image = image_path_or_array

            if preprocessing_steps is None:
                preprocessing_steps = ['resize', 'denoise_fastNlMeans', 'contrast_clahe', 'deskew']

            preprocessed = self.preprocess(image, preprocessing_steps)

            paddle_result = self.ocr_engine.extract_with_confidence(preprocessed)
            result = {
                **paddle_result,
                'engine': 'paddleocr',
                'preprocessing_steps_used': preprocessing_steps,
            }

            if self.enable_table_detection:
                try:
                    tables = self._detect_tables_cv(preprocessed)
                    result['tables'] = []
                    for tbl in tables:
                        tbl_text = self.ocr_engine.extract_with_confidence(tbl['image'])
                        result['tables'].append({
                            'bbox': tbl['bbox'],
                            'text': tbl_text.get('text', ''),
                            'confidence': tbl_text.get('confidence', 0)
                        })
                except Exception as e:
                    logger.warning(f"Table detection failed: {e}")

            logger.info(f"OCR pipeline complete. Text length: {len(result.get('text',''))} chars")
            return result
        except Exception as e:
            logger.error(f"OCR pipeline failed: {e}")
            raise

    def preprocess(self, image: np.ndarray, steps: List[str]) -> np.ndarray:
        result = image.copy()
        for step in steps:
            if step == 'grayscale':
                result = self.preprocessor.grayscale(result)
                logger.debug("Applied: grayscale")
            elif step == 'resize':
                result = self.preprocessor.resize_image(result)
                logger.debug("Applied: resize")
            elif step == 'denoise_bilateral':
                result = self.preprocessor.denoise_bilateral(result)
                logger.debug("Applied: denoise_bilateral")
            elif step == 'denoise_fastNlMeans':
                result = self.preprocessor.denoise_fastNlMeans(result)
                logger.debug("Applied: denoise_fastNlMeans")
            elif step == 'contrast_clahe':
                result = self.preprocessor.contrast_clahe(result)
                logger.debug("Applied: contrast_clahe")
            elif step == 'threshold_otsu':
                result = self.preprocessor.threshold_otsu(result)
                logger.debug("Applied: threshold_otsu")
            elif step == 'deskew':
                result = self.preprocessor.deskew(result)
                logger.debug("Applied: deskew")
            elif step == 'sharpen':
                result = self.preprocessor.sharpen_unsharp_mask(result)
                logger.debug("Applied: sharpen")
            elif step == 'invert':
                result = self.preprocessor.invert_image(result)
                logger.debug("Applied: invert")
            else:
                logger.warning(f"Unknown preprocessing step: {step}")
        return result

    def get_preprocessing_preview(self, image: np.ndarray, step: str) -> np.ndarray:
        if step == 'grayscale':
            return self.preprocessor.grayscale(image)
        if step == 'resize':
            return self.preprocessor.resize_image(image)
        if step == 'denoise_fastNlMeans':
            return self.preprocessor.denoise_fastNlMeans(image)
        if step == 'denoise_bilateral':
            return self.preprocessor.denoise_bilateral(image)
        if step == 'contrast_clahe':
            return self.preprocessor.contrast_clahe(image)
        if step == 'threshold_otsu':
            return self.preprocessor.threshold_otsu(image)
        if step == 'deskew':
            return self.preprocessor.deskew(image)
        if step == 'sharpen':
            return self.preprocessor.sharpen_unsharp_mask(image)
        return image


def simple_ocr(image_path: str) -> str:
    pipeline = OCRPipeline()
    result = pipeline.process(image_path)
    return result.get('text', '')


def ocr_with_preprocessing(image_path: str, preprocessing_steps: List[str]) -> Dict:
    pipeline = OCRPipeline()
    return pipeline.process(image_path, preprocessing_steps)
