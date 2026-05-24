"""
OCR Pipeline Module - Image Processing and Text Extraction

Provides customizable image preprocessing and Tesseract-based OCR for extracting
text from images (PNG, JPG, JPEG, BMP, TIFF).

Pipeline: Image → Preprocessing → Tesseract OCR → Text Extraction
"""

import cv2
import pytesseract
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import json
import numpy as np
from PIL import Image
import shutil

logger = logging.getLogger("OCRProcessor")


def configure_tesseract_path():
    """Point pytesseract to a local Windows install when Tesseract is not on PATH."""
    if shutil.which("tesseract"):
        return

    common_paths = [
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files\Tesseract-OCR\bin\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\bin\tesseract.exe"),
    ]
    for path in common_paths:
        if path.exists():
            pytesseract.pytesseract.tesseract_cmd = str(path)
            logger.info(f"Using Tesseract executable: {path}")
            return


class ImagePreprocessor:
    """Handles image preprocessing with multiple techniques."""
    
    @staticmethod
    def grayscale(image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Grayscale image
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    @staticmethod
    def threshold_otsu(image: np.ndarray) -> np.ndarray:
        """
        Apply Otsu's binary thresholding.
        
        Automatically finds optimal threshold value. Good for high-contrast images.
        
        Args:
            image: Grayscale image
            
        Returns:
            Binary image (black & white only)
        """
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    @staticmethod
    def threshold_adaptive(image: np.ndarray, block_size: int = 11) -> np.ndarray:
        """
        Apply adaptive thresholding.
        
        Threshold varies across image. Good for varying lighting conditions.
        
        Args:
            image: Grayscale image
            block_size: Size of neighborhood area (must be odd)
            
        Returns:
            Binary image
        """
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, 2
        )
    
    @staticmethod
    def threshold_binary(image: np.ndarray, threshold: int = 127) -> np.ndarray:
        """
        Apply fixed binary thresholding.
        
        Args:
            image: Grayscale image
            threshold: Threshold value (0-255)
            
        Returns:
            Binary image
        """
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        return binary
    
    @staticmethod
    def denoise_bilateral(image: np.ndarray, diameter: int = 9, 
                         sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        """
        Apply bilateral filtering for denoising.
        
        Preserves edges while removing noise. Good for text documents.
        
        Args:
            image: Input image
            diameter: Diameter of pixel neighborhood
            sigma_color: Filter sigma in color space
            sigma_space: Filter sigma in coordinate space
            
        Returns:
            Denoised image
        """
        return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
    
    @staticmethod
    def denoise_morphological(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apply morphological operations for denoising.
        
        Uses closing operation (dilation then erosion) to remove small noise.
        
        Args:
            image: Binary image
            kernel_size: Size of morphological kernel
            
        Returns:
            Denoised binary image
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        return opened
    
    @staticmethod
    def denoise_fastNlMeans(image: np.ndarray, h: float = 10) -> np.ndarray:
        """
        Apply Fast Non-Local Means Denoising.
        
        More advanced denoising. Works on color or grayscale.
        
        Args:
            image: Input image
            h: Filter strength (higher = more denoising)
            
        Returns:
            Denoised image
        """
        if len(image.shape) == 2:  # Grayscale
            return cv2.fastNlMeansDenoising(image, None, 10, 10, 21)
        else:  # Color
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    @staticmethod
    def sharpen_kernel(image: np.ndarray) -> np.ndarray:
        """
        Apply kernel-based sharpening.
        
        Uses unsharp masking kernel.
        
        Args:
            image: Input image
            
        Returns:
            Sharpened image
        """
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) / 1.0
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened
    
    @staticmethod
    def sharpen_unsharp_mask(image: np.ndarray, strength: float = 1.5) -> np.ndarray:
        """
        Apply unsharp mask sharpening.
        
        Blurs image, subtracts from original, adds back to enhance edges.
        
        Args:
            image: Input image
            strength: Sharpening strength (1.0-3.0)
            
        Returns:
            Sharpened image
        """
        blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
        sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    @staticmethod
    def resize_image(image: np.ndarray, width: int = 1000) -> np.ndarray:
        """
        Resize image proportionally.
        
        Args:
            image: Input image
            width: Target width
            
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        aspect = w / h
        height = int(width / aspect)
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def invert_image(image: np.ndarray) -> np.ndarray:
        """
        Invert image colors.
        
        Useful for white text on dark background.
        
        Args:
            image: Input image
            
        Returns:
            Inverted image
        """
        return cv2.bitwise_not(image)


class OCREngine:
    """Handles Tesseract OCR extraction."""
    
    def __init__(self, language: str = 'eng'):
        """
        Initialize OCR Engine.
        
        Args:
            language: Tesseract language code ('eng', 'fra', 'deu', etc.)
        """
        self.language = language
        self._verify_tesseract()
    
    def _verify_tesseract(self):
        """Verify Tesseract is installed and accessible."""
        try:
            configure_tesseract_path()
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR engine verified")
        except Exception as e:
            logger.error(f"Tesseract not found: {e}")
            raise RuntimeError(
                "Tesseract OCR not installed. "
                "Please install: https://github.com/UB-Mannheim/tesseract/wiki"
            )
    
    def extract_text(self, image: np.ndarray) -> str:
        """
        Extract text from image using Tesseract.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Extracted text
        """
        try:
            text = pytesseract.image_to_string(image, lang=self.language)
            logger.info(f"Extracted {len(text)} characters from image")
            return text.strip()
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            raise
    
    def extract_with_confidence(self, image: np.ndarray) -> Dict:
        """
        Extract text and confidence scores.
        
        Args:
            image: Input image
            
        Returns:
            Dict with 'text' and 'confidence' (0-100)
        """
        try:
            data = pytesseract.image_to_data(image, lang=self.language, output_type=pytesseract.Output.DICT)
            
            # Tesseract commonly returns confidence values under "conf".
            # Some wrappers/examples use "confidence", so support both.
            raw_confidences = data.get('conf', data.get('confidence', []))
            confidences = []
            for conf in raw_confidences:
                try:
                    value = float(conf)
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    confidences.append(value)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Extract full text
            text = pytesseract.image_to_string(image, lang=self.language).strip()
            
            logger.info(f"OCR confidence: {avg_confidence:.1f}%")
            return {
                'text': text,
                'confidence': avg_confidence,
                'word_count': len(text.split())
            }
        except Exception as e:
            logger.warning(f"Confidence extraction failed, falling back to text-only OCR: {e}")
            text = self.extract_text(image)
            return {
                'text': text,
                'confidence': 0,
                'word_count': len(text.split())
            }
    
    def set_language(self, language: str):
        """
        Set OCR language.
        
        Args:
            language: Language code (e.g., 'eng', 'fra', 'deu')
        """
        self.language = language
        logger.info(f"OCR language set to: {language}")


class PaddleOCREngine:
    """Handles PaddleOCR extraction for fallback scenarios."""

    def __init__(self, language: str = 'eng', use_gpu: bool = False):
        """
        Initialize PaddleOCR Engine.

        Args:
            language: Tesseract-style language code ('eng', 'fra', 'deu', etc.)
            use_gpu: Enable GPU acceleration if available
        """
        self.language = language
        self.use_gpu = use_gpu
        self._ocr = None

    def _map_language(self) -> str:
        """Map Tesseract language codes to PaddleOCR language identifiers."""
        language_map = {
            'eng': 'en',
            'fra': 'fr',
            'deu': 'german',
            'spa': 'es',
            'ita': 'it',
            'por': 'pt',
            'rus': 'ru',
            'chi_sim': 'ch',
            'chi_tra': 'ch',
        }
        return language_map.get(self.language, 'en')

    def _get_ocr(self):
        if self._ocr is not None:
            return self._ocr

        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR not installed. Please install paddleocr and paddlepaddle "
                "to enable the fallback OCR engine."
            ) from exc

        paddle_language = self._map_language()

        kwargs = {'use_angle_cls': True, 'lang': paddle_language}
        if self.use_gpu:
            kwargs['use_gpu'] = True

        try:
            self._ocr = PaddleOCR(**kwargs)
        except (TypeError, ValueError) as e:
            if 'use_gpu' in kwargs:
                kwargs.pop('use_gpu', None)
                logger.warning(
                    f"PaddleOCR init does not support use_gpu; falling back to CPU. Details: {e}"
                )
                self._ocr = PaddleOCR(**kwargs)
            else:
                raise

        logger.info(f"PaddleOCR engine initialized (lang={paddle_language}, gpu={self.use_gpu})")
        return self._ocr

    def extract_with_confidence(self, image: np.ndarray) -> Dict:
        """
        Extract text and confidence scores using PaddleOCR.

        Args:
            image: Input image

        Returns:
            Dict with 'text' and 'confidence' (0-100)
        """
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

        logger.info(f"PaddleOCR confidence: {avg_confidence:.1f}%")
        return {
            'text': text_output,
            'confidence': avg_confidence,
            'word_count': len(text_output.split())
        }


class OCRPipeline:
    """Orchestrates preprocessing and OCR extraction."""
    
    def __init__(
        self,
        language: str = 'eng',
        enable_paddle_fallback: bool = True,
        paddle_confidence_threshold: float = 70.0,
        paddle_use_gpu: bool = False
    ):
        """
        Initialize OCR Pipeline.
        
        Args:
            language: Tesseract language code
            enable_paddle_fallback: Use PaddleOCR when Tesseract confidence is low
            paddle_confidence_threshold: Minimum Tesseract confidence before fallback
            paddle_use_gpu: Enable GPU acceleration for PaddleOCR when available
        """
        self.preprocessor = ImagePreprocessor()
        self.ocr_engine = OCREngine(language)
        self.supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        self.enable_paddle_fallback = enable_paddle_fallback
        self.paddle_confidence_threshold = paddle_confidence_threshold
        self.paddle_use_gpu = paddle_use_gpu
        self._paddle_engine = None

    def _get_paddle_engine(self) -> PaddleOCREngine:
        if self._paddle_engine is None:
            self._paddle_engine = PaddleOCREngine(
                language=self.ocr_engine.language,
                use_gpu=self.paddle_use_gpu
            )
        return self._paddle_engine

    def _should_use_paddle(self, tesseract_result: Dict) -> bool:
        confidence = float(tesseract_result.get('confidence', 0) or 0)
        word_count = int(tesseract_result.get('word_count', 0) or 0)
        text = tesseract_result.get('text', '') or ''
        if not text.strip():
            return True
        if word_count == 0:
            return True
        return confidence < self.paddle_confidence_threshold
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load image from file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array
            
        Raises:
            ValueError: If file not found or not supported format
        """
        path = Path(image_path)
        
        if not path.exists():
            raise ValueError(f"Image file not found: {image_path}")
        
        if path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported image format: {path.suffix}")
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        logger.info(f"Loaded image: {path.name} ({image.shape[0]}x{image.shape[1]})")
        return image
    
    def process(self, image_path_or_array, preprocessing_steps: Optional[List[str]] = None) -> Dict:
        """
        Run full OCR pipeline with preprocessing.
        
        Args:
            image_path_or_array: Path to image file or numpy array
            preprocessing_steps: List of preprocessing steps to apply
                Options: 'grayscale', 'threshold_otsu', 'threshold_adaptive',
                        'denoise_bilateral', 'denoise_morphological', 'sharpen'
                
        Returns:
            Dict with 'text', 'confidence', 'preprocessing_steps_used'
        """
        try:
            # Load image
            if isinstance(image_path_or_array, str):
                image = self.load_image(image_path_or_array)
            else:
                image = image_path_or_array
            
            # Default preprocessing
            if preprocessing_steps is None:
                preprocessing_steps = ['grayscale']
            
            # Apply preprocessing
            preprocessed = self.preprocess(image, preprocessing_steps)
            
            # Extract text with confidence
            tesseract_result = self.ocr_engine.extract_with_confidence(preprocessed)
            result = {
                **tesseract_result,
                'engine': 'tesseract',
                'fallback_used': False,
                'fallback_attempted': False,
                'fallback_error': None,
                'preprocessing_steps_used': preprocessing_steps
            }

            if self.enable_paddle_fallback and self._should_use_paddle(tesseract_result):
                try:
                    paddle_result = self._get_paddle_engine().extract_with_confidence(preprocessed)
                except Exception as e:
                    logger.warning(f"PaddleOCR fallback failed, returning Tesseract result: {e}")
                    result['fallback_attempted'] = True
                    result['fallback_error'] = str(e)
                else:
                    result = {
                        **paddle_result,
                        'engine': 'paddleocr',
                        'fallback_used': True,
                        'fallback_attempted': True,
                        'fallback_error': None,
                        'tesseract_confidence': tesseract_result.get('confidence', 0),
                        'tesseract_word_count': tesseract_result.get('word_count', 0),
                        'tesseract_text': tesseract_result.get('text', ''),
                        'preprocessing_steps_used': preprocessing_steps
                    }
            
            logger.info(f"OCR pipeline complete. Text length: {len(result['text'])} chars")
            return result
            
        except Exception as e:
            logger.error(f"OCR pipeline failed: {e}")
            raise
    
    def preprocess(self, image: np.ndarray, steps: List[str]) -> np.ndarray:
        """
        Apply preprocessing steps to image.
        
        Args:
            image: Input image
            steps: List of preprocessing steps
            
        Returns:
            Preprocessed image
        """
        result = image.copy()
        
        for step in steps:
            if step == 'grayscale':
                result = self.preprocessor.grayscale(result)
                logger.debug("Applied: grayscale")
                
            elif step == 'threshold_otsu':
                result = self.preprocessor.threshold_otsu(result)
                logger.debug("Applied: threshold_otsu")
                
            elif step == 'threshold_adaptive':
                result = self.preprocessor.threshold_adaptive(result)
                logger.debug("Applied: threshold_adaptive")
                
            elif step == 'denoise_bilateral':
                result = self.preprocessor.denoise_bilateral(result)
                logger.debug("Applied: denoise_bilateral")
                
            elif step == 'denoise_morphological':
                result = self.preprocessor.denoise_morphological(result)
                logger.debug("Applied: denoise_morphological")
                
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
        """
        Get preview of single preprocessing step.
        
        Args:
            image: Input image
            step: Preprocessing step name
            
        Returns:
            Image after applying step
        """
        if step == 'grayscale':
            return self.preprocessor.grayscale(image)
        elif step == 'threshold_otsu':
            gray = self.preprocessor.grayscale(image)
            return self.preprocessor.threshold_otsu(gray)
        elif step == 'threshold_adaptive':
            gray = self.preprocessor.grayscale(image)
            return self.preprocessor.threshold_adaptive(gray)
        elif step == 'denoise_bilateral':
            gray = self.preprocessor.grayscale(image)
            return self.preprocessor.denoise_bilateral(gray)
        elif step == 'denoise_morphological':
            gray = self.preprocessor.grayscale(image)
            threshold = self.preprocessor.threshold_otsu(gray)
            return self.preprocessor.denoise_morphological(threshold)
        elif step == 'sharpen':
            gray = self.preprocessor.grayscale(image)
            return self.preprocessor.sharpen_unsharp_mask(gray)
        else:
            return image
    
    def save_image(self, image: np.ndarray, output_path: str):
        """
        Save image to file.
        
        Args:
            image: Image to save
            output_path: Output file path
        """
        cv2.imwrite(output_path, image)
        logger.info(f"Saved image: {output_path}")


# Convenience functions

def simple_ocr(image_path: str) -> str:
    """
    Quick OCR on image with default preprocessing.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Extracted text
    """
    pipeline = OCRPipeline()
    result = pipeline.process(image_path)
    return result['text']


def ocr_with_preprocessing(image_path: str, preprocessing_steps: List[str]) -> Dict:
    """
    OCR with custom preprocessing.
    
    Args:
        image_path: Path to image file
        preprocessing_steps: List of preprocessing steps
        
    Returns:
        Dict with text, confidence, and preprocessing steps
    """
    pipeline = OCRPipeline()
    return pipeline.process(image_path, preprocessing_steps)
