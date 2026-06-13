#!/usr/bin/env python
"""
IntelliSafe - Sensitive Data Detection Application
Main entry point with complete implementation
"""

import sys
import os
import logging
from pathlib import Path
from typing import Optional

from app.backend.privacy_risk_manager import PrivacyRiskManager
from app.backend.redaction_engine import RedactionEngine
from app.backend.database import get_db
from app.backend.sensitive_data_tracker import SensitiveDataTracker
from app.backend.app_settings import AppSettings
from app.backend.session_store import SessionStore
from app.backend.compliance_engine import ComplianceEngine
from app.backend.redaction_service import filter_findings_by_mode, run_redaction_for_file, selected_findings

# Load Torch before Qt to avoid a Windows DLL initialization conflict where
# importing Qt first can prevent torch\lib\c10.dll from initializing.
try:
    import torch  # noqa: F401
except ImportError:
    torch = None

# ============================================================================
# PROJECT SETUP
# ============================================================================

def setup_project_structure():
    """Create necessary project directories"""
    base_path = Path(__file__).parent
    
    dirs = [
        'app/ui', 'app/backend', 'app/ocr', 'app/nip',
        'app/backend/detection', 'app/encryption', 'app/compliance', 'app/database', 'app/reports', 'app/utils',
        'assets/icons', 'assets/styles', 'test_files', 'database', 'logs'
    ]
    
    for dir_path in dirs:
        (base_path / dir_path).mkdir(parents=True, exist_ok=True)
    
    # Create __init__.py files
    packages = ['app', 'app/ui', 'app/backend', 'app/backend/detection', 'app/ocr', 'app/nip',
                'app/encryption', 'app/compliance', 'app/database', 'app/reports', 'app/utils']
    
    for pkg in packages:
        init_file = base_path / pkg / '__init__.py'
        if not init_file.exists():
            init_file.write_text(f'"""Package: {pkg}"""\n')

def setup_logging():
    """Configure application logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "app.log"),
            logging.StreamHandler()
        ]
    )

    # Suppress noisy third-party loggers that produce expected/benign warnings.
    # huggingface_hub emits unauthenticated-request warnings even when models
    # are already cached locally; these are irrelevant for end users.
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)

# ============================================================================
# STYLES
# ============================================================================

STYLESHEET = """
/* Main Application */
QMainWindow, QWidget {
    background-color: #0f1419;
    color: #e0e6ed;
}

/* Sidebar */
#sidebar {
    background-color: #1a1f2e;
    border-right: 1px solid #2a3f5f;
    min-width: 220px;
}

/* Buttons */
QPushButton {
    background-color: #1e40af;
    color: #ffffff;
    border: none;
    padding: 10px 20px;
    border-radius: 4px;
    font-weight: 500;
    font-size: 13px;
}

QPushButton:hover {
    background-color: #1e3a8a;
}

QPushButton:pressed {
    background-color: #1a3165;
}

#sidebar QPushButton {
    background-color: transparent;
    color: #a0aec0;
    padding: 12px 16px;
    text-align: left;
    border: none;
    border-left: 3px solid transparent;
}

#sidebar QPushButton:hover {
    background-color: #2a3f5f;
    color: #e0e6ed;
}

#sidebar QPushButton[active="true"] {
    background-color: #1e3a8a;
    color: #60a5fa;
    border-left: 3px solid #60a5fa;
}

/* Labels */
QLabel {
    color: #e0e6ed;
}

#title {
    font-size: 24px;
    font-weight: bold;
    color: #f1f5f9;
}

#subtitle {
    font-size: 14px;
    color: #a0aec0;
}

#stat-label {
    font-size: 12px;
    color: #64748b;
}

#stat-value {
    font-size: 32px;
    font-weight: bold;
    color: #60a5fa;
}

/* Frames */
QFrame[class="card"] {
    background-color: #1a1f2e;
    border: 1px solid #2a3f5f;
    border-radius: 8px;
    padding: 20px;
}

QFrame[class="dropzone"] {
    background-color: #1a1f2e;
    border: 2px dashed #334155;
    border-radius: 8px;
}

QFrame[class="dropzone"][hover="true"] {
    border: 2px dashed #3b82f6;
    background-color: #1e3a8a;
}

/* Input */
QLineEdit, QTextEdit {
    background-color: #1e293b;
    color: #e0e6ed;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 8px 12px;
}

QLineEdit:focus, QTextEdit:focus {
    border: 1px solid #3b82f6;
}

/* Scroll Bar */
QScrollBar:vertical {
    background-color: #0f1419;
    width: 12px;
}

QScrollBar::handle:vertical {
    background-color: #475569;
    border-radius: 6px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #64748b;
}
"""

# ============================================================================
# FILE HANDLER
# ============================================================================

class FileHandler:
    """Handle file uploads and validation"""
    
    SUPPORTED_FORMATS = {'.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.docx', '.sql', '.txt'}
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
    LARGE_FILE_WARNING = 10 * 1024 * 1024  # 10 MB warning threshold

    @staticmethod
    def validate_file(file_path: Path) -> tuple:
        if not file_path.exists():
            return False, "File does not exist"

        if file_path.suffix.lower() not in FileHandler.SUPPORTED_FORMATS:
            return False, f"Unsupported format: {file_path.suffix}"

        if file_path.stat().st_size > FileHandler.MAX_FILE_SIZE:
            size_mb = FileHandler.MAX_FILE_SIZE / 1024 / 1024
            return False, f"File exceeds maximum size of {size_mb:.0f} MB"

        return True, "Valid"

    @staticmethod
    def get_file_info(file_path: Path) -> dict:
        stat = file_path.stat()
        size_bytes = stat.st_size
        warning = size_bytes > FileHandler.LARGE_FILE_WARNING
        return {
            'name': file_path.name,
            'path': str(file_path),
            'size': size_bytes,
            'format': file_path.suffix.lower(),
            'created': stat.st_ctime,
            'modified': stat.st_mtime,
            'size_warning': warning,
        }

# ============================================================================
# PYSIDE6 IMPORTS AND COMPONENTS
# ============================================================================

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QLabel, QFrame, QFileDialog, QListWidget, QListWidgetItem, QScrollArea,
    QTextEdit, QCheckBox, QComboBox, QSizePolicy, QDialog, QLineEdit,
)
from PySide6.QtCore import Qt, QSize, QMimeData, QTimer, Signal, QObject, QThread
from PySide6.QtGui import QIcon, QFont, QDrag, QColor, QDropEvent, QPixmap

logger = logging.getLogger(__name__)

# ============================================================================
# COMPONENTS
# ============================================================================

class UploadDropZone(QFrame):
    """Drag and drop zone for file uploads"""
    
    files_dropped = Signal(list)  # Signal when files are dropped
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("dropzone")
        self.setAcceptDrops(True)
        self.is_hover = False
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        
        # Icon
        icon = QLabel("📁")
        icon.setFont(QFont("Arial", 48))
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon)
        
        # Main text
        text = QLabel("Drag files here")
        text.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text.setObjectName("title")
        layout.addWidget(text)
        
        # Subtext
        subtext = QLabel("or")
        subtext.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtext)
        
        # Browse button
        browse_btn = QPushButton("Browse your device")
        browse_btn.clicked.connect(self.browse_files)
        layout.addWidget(browse_btn)
        
        layout.addSpacing(20)
        
        # File type hints
        hints_layout = QHBoxLayout()
        hints_layout.addStretch()
        for ftype in ["PDF", "IMG", "DOCX", "SQL"]:
            label = QLabel(ftype)
            label.setStyleSheet("color: #60a5fa; font-size: 11px; padding: 4px 8px; border: 1px solid #334155; border-radius: 4px;")
            hints_layout.addWidget(label)
        hints_layout.addStretch()
        layout.addLayout(hints_layout)
        
        layout.addStretch()
        
        self.setMinimumHeight(400)
    
    def browse_files(self):
        """Open file browser dialog"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select files to upload",
            "",
            "All Supported (*.pdf *.png *.jpg *.jpeg *.docx *.sql *.txt);;PDF Files (*.pdf);;Images (*.png *.jpg *.jpeg);;Word (*.docx);;SQL (*.sql);;Text (*.txt)"
        )
        if files:
            self.files_dropped.emit([Path(f) for f in files])
    
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.is_hover = True
            self.setProperty("hover", "true")
            self.style().unpolish(self)
            self.style().polish(self)
    
    def dragLeaveEvent(self, event):
        self.is_hover = False
        self.setProperty("hover", "false")
        self.style().unpolish(self)
        self.style().polish(self)
    
    def dropEvent(self, event: QDropEvent):
        files = []
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            files.append(Path(file_path))
        
        self.files_dropped.emit(files)
        self.is_hover = False
        self.setProperty("hover", "false")
        self.style().unpolish(self)
        self.style().polish(self)

class FileListWidget(QWidget):
    """Widget to display uploaded files"""

    def __init__(self, files=None, parent=None):
        super().__init__(parent)
        self.files = files if files is not None else []
        self._validation_pipeline = None
        self.init_ui()
        self.refresh_list()

    def init_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("Uploaded Files")
        title.setObjectName("title")
        layout.addWidget(title)

        self.list_widget = QListWidget()
        self.list_widget.setMaximumHeight(300)
        layout.addWidget(self.list_widget)

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.clear_files)
        layout.addWidget(clear_btn)

        layout.addStretch()

    def _get_validation_pipeline(self):
        if self._validation_pipeline is None:
            try:
                from app.backend.detection import ValidationPipeline
                self._validation_pipeline = ValidationPipeline()
            except ImportError:
                pass
        return self._validation_pipeline
    
    def add_files(self, files: list):
        """Add files to the list and parse them"""
        # Import parser
        try:
            from app.backend.file_parser import FileParser
        except ImportError:
            logger.error("Could not import FileParser")
            return

        host = self.window()

        for file_path in files:
            valid, msg = FileHandler.validate_file(file_path)
            if not valid:
                logger.warning(f"Invalid file {file_path}: {msg}")
                continue

            suffix = file_path.suffix.lower()
            is_image = suffix in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

            if is_image and host is not None and hasattr(host, "perform_image_ocr"):
                logger.info("Routing image upload through OCR flow: %s", file_path.name)
                ocr_pending = host.perform_image_ocr(file_path)
                if ocr_pending is None:
                    logger.info("OCR canceled for %s", file_path.name)
                    continue
                self.refresh_list()
                continue

            info = FileHandler.get_file_info(file_path)
            self.files.append(info)

            logger.info(f"Parsing file: {file_path.name}")
            parse_result = FileParser.parse(file_path)

            if parse_result['status'] == 'success':
                info['parsed_content'] = parse_result['content']
                info['parsed_metadata'] = parse_result['metadata']
                validator = self._get_validation_pipeline()
                scan_mode = "standard"
                if host is not None and hasattr(host, "_app_settings"):
                    scan_mode = host._app_settings.get("default_scan_mode", "standard")
                if validator:
                    validation = validator.run(parse_result['content'], mode=scan_mode)
                    info['findings'] = validation['findings']
                    info['findings_summary'] = validation['summary']
                    info['validation_tier'] = validation['validation_tier']
                    info['confidence_breakdown'] = validation['confidence_breakdown']
                    info['regex_findings'] = validation['findings']
                    info['regex_summary'] = validation['summary']
                else:
                    info['findings'] = []
                    info['findings_summary'] = {}
                    info['validation_tier'] = "standard"
                    info['confidence_breakdown'] = {}
                    info['regex_findings'] = []
                    info['regex_summary'] = {}
                assessment = None
                if host is not None and hasattr(host, "finalize_file_scan"):
                    assessment = host.finalize_file_scan(info)
                elif host is not None and hasattr(host, "_risk_manager"):
                    assessment = host._risk_manager.assess(info.get('findings', []))
                    info['risk_assessment'] = assessment
                logger.info(f"Successfully parsed: {file_path.name}")
            else:
                logger.error(f"Failed to parse {file_path.name}: {parse_result.get('error', 'Unknown error')}")
                info['parse_error'] = parse_result.get('error', 'Unknown error')
                if host is not None and hasattr(host, "finalize_file_scan"):
                    host.finalize_file_scan(info)

            self.add_file_item(info)

            logger.info(f"File added to list: {info['name']}")

    def refresh_list(self):
        """Render the current uploaded files."""
        self.list_widget.clear()
        for info in self.files:
            self.add_file_item(info)

    def add_file_item(self, info: dict):
        size_mb = info['size'] / 1024 / 1024
        warning = ""
        if info.get('size_warning'):
            warning = " ⚠"
        if 'parse_error' in info:
            item_text = f"✗ {info['name']} ({size_mb:.2f} MB){warning} - Parse Error"
        elif 'parsed_content' in info:
            finding_count = len(info.get('findings', info.get('regex_findings', [])))
            dup = f" [dup:{info['duplicate_of']}]" if info.get('duplicate_of') else ""
            item_text = f"✓ {info['name']} ({size_mb:.2f} MB){warning} - {finding_count} findings{dup}"
        elif info.get('processing_status'):
            item_text = f"{info['name']} ({size_mb:.2f} MB){warning} - {info['processing_status']}"
        else:
            item_text = f"{info['name']} ({size_mb:.2f} MB){warning} - {info['format']}"

        item = QListWidgetItem(item_text)
        self.list_widget.addItem(item)
    
    def clear_files(self):
        """Clear all files"""
        self.files.clear()
        self.list_widget.clear()
        logger.info("Files cleared")


class ImageOCRPromptDialog(QDialog):
    """Prompt for OCR preprocessing options when an image is uploaded."""

    def __init__(self, file_path: Path, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.setWindowTitle(f"OCR Options - {file_path.name}")
        self.setModal(True)
        self.selected_steps: list[str] = []
        self.selected_language = "eng"
        self.selected_scan_mode = "standard"
        self.skip_preprocessing = False
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        title = QLabel(f"OCR preprocessing for {self.file_path.name}")
        title.setObjectName("title")
        layout.addWidget(title)

        file_size_mb = self.file_path.stat().st_size / 1024 / 1024
        if file_size_mb > 10:
            warning = QLabel(f"⚠ Large file ({file_size_mb:.1f} MB). Consider skipping preprocessing for faster processing.")
            warning.setStyleSheet("color: #fbbf24; font-size: 11px;")
            warning.setWordWrap(True)
            layout.addWidget(warning)

        subtitle = QLabel("Choose image preprocessing before IntelliSafe extracts text and runs detection.")
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        form_row = QHBoxLayout()

        language_label = QLabel("Language")
        form_row.addWidget(language_label)
        self.language_combo = QComboBox()
        for language in ["eng", "eng+spa", "spa", "fra", "deu"]:
            self.language_combo.addItem(language)
        form_row.addWidget(self.language_combo)

        scan_label = QLabel("Scan Mode")
        form_row.addWidget(scan_label)
        self.scan_mode_combo = QComboBox()
        self.scan_mode_combo.addItem("Quick Scan", "quick")
        self.scan_mode_combo.addItem("Standard", "standard")
        self.scan_mode_combo.addItem("Deep Analysis", "deep")
        self.scan_mode_combo.setCurrentIndex(1)
        form_row.addWidget(self.scan_mode_combo)

        form_row.addStretch()
        layout.addLayout(form_row)

        steps_title = QLabel("Preprocessing")
        steps_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(steps_title)

        self.step_checkboxes: dict[str, QCheckBox] = {}
        steps_layout = QHBoxLayout()
        default_steps = {"resize", "denoise_fastNlMeans", "contrast_clahe", "deskew"}
        for step, label in [
            ("resize", "Resize"),
            ("grayscale", "Grayscale"),
            ("threshold_otsu", "Otsu Threshold"),
            ("threshold_adaptive", "Adaptive Threshold"),
            ("denoise_fastNlMeans", "Denoise (FastNLMeans)"),
            ("denoise_bilateral", "Denoise (Bilateral)"),
            ("denoise_morphological", "Morphology"),
            ("contrast_clahe", "CLAHE Contrast"),
            ("deskew", "Deskew"),
            ("sharpen", "Sharpen"),
            ("invert", "Invert"),
        ]:
            checkbox = QCheckBox(label)
            checkbox.setChecked(step in default_steps)
            self.step_checkboxes[step] = checkbox
            steps_layout.addWidget(checkbox)

        steps_layout.addStretch()
        layout.addLayout(steps_layout)

        self.skip_checkbox = QCheckBox("Skip preprocessing (fastest)")
        self.skip_checkbox.setStyleSheet("color: #fbbf24;")
        self.skip_checkbox.toggled.connect(self._on_skip_toggled)
        layout.addWidget(self.skip_checkbox)

        actions = QHBoxLayout()
        actions.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        actions.addWidget(cancel_btn)
        process_btn = QPushButton("Process Image")
        process_btn.clicked.connect(self.accept)
        actions.addWidget(process_btn)
        layout.addLayout(actions)

    def _on_skip_toggled(self, checked: bool):
        for checkbox in self.step_checkboxes.values():
            checkbox.setEnabled(not checked)

    def get_settings(self) -> dict:
        if self.skip_checkbox.isChecked():
            steps = []
        else:
            steps = [step for step, checkbox in self.step_checkboxes.items() if checkbox.isChecked()]
            if not steps:
                steps = ["grayscale"]
        return {
            "selected_steps": steps,
            "language": self.language_combo.currentText(),
            "scan_mode": self.scan_mode_combo.currentData(),
        }

# ============================================================================
# MAIN WINDOW
# ============================================================================

class _OcrWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, file_path, settings, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.settings = settings

    def run(self):
        try:
            from app.backend.ocr_processor import OCRPipeline
            from app.backend.detection import ValidationPipeline
            pipeline = OCRPipeline(language=self.settings['language'])
            result = pipeline.process(str(self.file_path), self.settings['selected_steps'])
            validation = ValidationPipeline().run(result.get('text', ''), mode=self.settings['scan_mode'])
            self.finished.emit({'ocr': result, 'validation': validation})
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("IntelliSafe - Sensitive Data Detection")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet(STYLESHEET)
        self._active_ocr_threads: list[QThread] = []
        self._active_ocr_workers: list[_OcrWorker] = []
        self.current_page = "upload"
        self.uploaded_files = []
        self.findings_file_list = None
        self.findings_findings_output = None
        self.findings_text_output = None
        self._validation_pipeline = None
        self._pipeline_ready = False
        self._risk_manager = PrivacyRiskManager()
        self._redaction_output_dir = str(Path(__file__).parent / "redacted_output")
        self._redaction_engine = RedactionEngine(self._redaction_output_dir)
        self._sensitive_tracker = SensitiveDataTracker(get_db())
        self._app_settings = AppSettings(get_db())
        self._session_store = SessionStore(get_db())
        self._compliance_engine = ComplianceEngine()
        self._load_app_settings()
        self._reload_session_from_db()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.sidebar = self.create_sidebar()
        main_layout.addWidget(self.sidebar)
        self.content_area = QWidget()
        self.content_layout = QVBoxLayout(self.content_area)
        main_layout.addWidget(self.content_area, 1)

        self._start_background_init()
        self._pending_redaction_mode = None
        self.switch_page("upload")
    
    def create_sidebar(self) -> QFrame:
        """Create navigation sidebar"""
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 20, 0, 20)
        layout.setSpacing(0)
        
        # Logo
        logo = QLabel("IntelliSafe")
        logo.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        logo.setStyleSheet("color: #60a5fa; padding: 0px 16px;")
        layout.addWidget(logo)
        
        layout.addSpacing(30)
        
        # Navigation items
        self.nav_buttons = {}
        nav_items = [
            ("📤 Upload Files", "upload"),
            ("📊 Dashboard", "dashboard"),
            ("✓ Compliance", "compliance"),
            ("🛡️ Protection", "protection"),
            ("⚙️ Detection", "detection"),
            ("🔎 Findings", "findings"),
            ("🛡️ Risk Assessment", "risk"),
            ("🔎 Redaction", "redact"),
            ("📋 Reports", "reports"),
        ]
        
        for label, page_id in nav_items:
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked, p=page_id: self.switch_page(p))
            self.nav_buttons[page_id] = btn
            layout.addWidget(btn)
        
        layout.addStretch()
        
        # Settings
        settings_btn = QPushButton("⚙️ Settings")
        settings_btn.clicked.connect(lambda: self.switch_page("settings"))
        layout.addWidget(settings_btn)
        
        return sidebar

    def _load_app_settings(self):
        settings = self._app_settings.load()
        self._redaction_output_dir = settings.get("redaction_output_dir", self._redaction_output_dir)
        self._redaction_engine.set_output_dir(self._redaction_output_dir)

    def _reload_session_from_db(self):
        if not self._app_settings.get("reload_session_on_startup", True):
            return
        try:
            loaded = self._session_store.load_uploaded_files()
            if loaded:
                self.uploaded_files = loaded
                logger.info("Restored %s file(s) from database", len(loaded))
        except Exception as exc:
            logger.warning("Could not reload session from database: %s", exc)

    def _get_selected_findings(self, file_info: dict) -> list:
        if "selected_finding_indices" not in file_info:
            file_info["selected_finding_indices"] = list(range(len(file_info.get("findings", []))))
        return selected_findings(file_info)
    
    def switch_page(self, page_id: str):
        """Switch to a different page"""
        self.current_page = page_id
        logger.info(f"Switching to page: {page_id}")
        
        # Update active button
        for btn_id, btn in self.nav_buttons.items():
            btn.setProperty("active", "true" if btn_id == page_id else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        
        # Clear all widgets from the content area
        self.clear_layout(self.content_layout)
        
        # Now display the appropriate page
        if page_id == "upload":
            self.show_upload_page()
        elif page_id == "dashboard":
            self.show_dashboard_page()
        elif page_id == "compliance":
            self.show_compliance_page()
        elif page_id == "protection":
            self.show_protection_page()
        elif page_id == "detection":
            self.show_detection_page()
        elif page_id == "findings":
            self.show_findings_page()
        elif page_id == "risk":
            self.show_risk_dashboard()
        elif page_id == "redact":
            self.show_redact_page()
        elif page_id == "reports":
            self.show_reports_page()
        elif page_id == "settings":
            self.show_settings_page()

    def clear_layout(self, layout):
        """Remove all widgets and nested layouts from a layout."""
        while layout.count() > 0:
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            elif child.layout():
                self.clear_layout(child.layout())
                child.layout().deleteLater()

    def _start_background_init(self):
        """Initialize heavy components in background thread."""
        from threading import Thread
        def init_models():
            try:
                from app.backend.detection import ValidationPipeline
                self._validation_pipeline = ValidationPipeline()
                self._pipeline_ready = True
                logger.info("Background model initialization complete")
            except Exception as e:
                logger.warning(f"Background initialization failed: {e}")

        thread = Thread(target=init_models, daemon=True)
        thread.start()

    def _get_validation_pipeline(self):
        return self._validation_pipeline

    def show_upload_page(self):
        """Display upload page"""
        layout = self.content_layout
        layout.setContentsMargins(40, 40, 40, 40)
        
        # Header
        title = QLabel("Upload Files")
        title.setObjectName("title")
        layout.addWidget(title)
        
        info = QLabel("PDF • IMG • DOCX • SQL")
        info.setObjectName("subtitle")
        layout.addWidget(info)
        
        layout.addSpacing(20)
        
        # Drop zone
        drop_zone = UploadDropZone()
        drop_zone.files_dropped.connect(self.on_files_uploaded)
        layout.addWidget(drop_zone)
        
        layout.addSpacing(30)
        
        # File list
        self.file_list = FileListWidget(self.uploaded_files, self)
        layout.addWidget(self.file_list)
        
        layout.addStretch()
    
    def show_dashboard_page(self):
        """Display dashboard page"""
        layout = self.content_layout
        layout.setContentsMargins(40, 40, 40, 40)
        
        title = QLabel("Detection Dashboard")
        title.setObjectName("title")
        layout.addWidget(title)
        
        subtitle = QLabel("Sensitive data • Confidence scores • File risk levels")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)
        
        layout.addSpacing(30)
        
        # Stats cards
        stats_layout = QHBoxLayout()
        files_scanned = sum(1 for file_info in self.uploaded_files if 'parsed_content' in file_info)
        sensitive_items = sum(len(file_info.get('regex_findings', [])) for file_info in self.uploaded_files)
        high_risk_files = sum(
            1
            for file_info in self.uploaded_files
            if any(finding.get('severity') == 'high' for finding in file_info.get('regex_findings', []))
        )
        
        for stat_name, stat_value, icon in [
            ("Files Scanned", "0", "📁"),
            ("Sensitive Items", "0", "⚠️"),
            ("High-Risk Files", "0", "🔴"),
        ]:
            if stat_name == "Files Scanned":
                stat_value = str(files_scanned)
            elif stat_name == "Sensitive Items":
                stat_value = str(sensitive_items)
            elif stat_name == "High-Risk Files":
                stat_value = str(high_risk_files)

            card = QFrame()
            card.setObjectName("card")
            card_layout = QVBoxLayout(card)
            
            icon_label = QLabel(icon)
            icon_label.setFont(QFont("Arial", 24))
            card_layout.addWidget(icon_label)
            
            name_label = QLabel(stat_name)
            name_label.setObjectName("stat-label")
            card_layout.addWidget(name_label)
            
            value_label = QLabel(stat_value)
            value_label.setObjectName("stat-value")
            card_layout.addWidget(value_label)
            
            stats_layout.addWidget(card)
        
        layout.addLayout(stats_layout)

        layout.addSpacing(30)

        review_title = QLabel("Parsed File Review")
        review_title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(review_title)

        review_layout = QHBoxLayout()

        file_list = QListWidget()
        file_list.setMinimumWidth(320)
        file_list.setMaximumWidth(420)

        preview = QTextEdit()
        preview.setReadOnly(True)
        preview.setPlaceholderText("Select a parsed file to preview extracted text.")

        if self.uploaded_files:
            for index, file_info in enumerate(self.uploaded_files):
                if 'parse_error' in file_info:
                    item_text = f"{file_info['name']} - Parse Error"
                elif 'parsed_content' in file_info:
                    finding_count = len(file_info.get('regex_findings', []))
                    item_text = f"{file_info['name']} - {finding_count} findings"
                else:
                    item_text = f"{file_info['name']} - Pending"

                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, index)
                file_list.addItem(item)
        else:
            file_list.addItem("No uploaded files yet")
            file_list.setEnabled(False)

        def show_selected_file():
            selected_items = file_list.selectedItems()
            if not selected_items:
                return

            file_index = selected_items[0].data(Qt.ItemDataRole.UserRole)
            if file_index is None:
                return

            file_info = self.uploaded_files[file_index]
            if 'parse_error' in file_info:
                preview.setPlainText(
                    f"Could not parse {file_info['name']}.\n\n"
                    f"Error: {file_info['parse_error']}"
                )
                return

            content = file_info.get('parsed_content')
            if content is not None:
                metadata = file_info.get('parsed_metadata', {})
                findings = file_info.get('regex_findings', [])
                metadata_lines = [
                    f"{key}: {value}"
                    for key, value in metadata.items()
                    if value not in (None, "")
                ]
                header = f"File: {file_info['name']}\n"
                if metadata_lines:
                    header += "\nMetadata:\n" + "\n".join(metadata_lines) + "\n"

                if findings:
                    finding_lines = []
                    for index, finding in enumerate(findings, start=1):
                        confidence = finding.get('consensus_score', finding.get('highest_confidence', finding.get('confidence')))
                        confidence_parts = []
                        if isinstance(confidence, (int, float)):
                            confidence_parts.append(f"confidence {confidence * 100:.1f}%" if confidence <= 1.0 else f"confidence {confidence:.1f}%")
                        transformer_conf = finding.get('transformer_confidence')
                        if isinstance(transformer_conf, (int, float)):
                            confidence_parts.append(f"transformer {transformer_conf * 100:.1f}%" if transformer_conf <= 1.0 else f"transformer {transformer_conf:.1f}%")
                        confidence_str = f" [{', '.join(confidence_parts)}]" if confidence_parts else ""
                        finding_lines.append(
                            f"{index}. {finding.get('type', finding['type'])} ({finding.get('severity', finding['severity'])}) "
                            f"line {finding.get('line', finding['line'])}: {finding.get('masked_value', finding.get('value', ''))}{confidence_str}\n"
                            f"   Context: {finding.get('context', finding['context'])}"
                        )
                    findings_text = "\n\nRegex Findings:\n" + "\n".join(finding_lines) + "\n"
                else:
                    findings_text = "\n\nRegex Findings:\nNo regex findings detected.\n"

                preview.setPlainText(f"{header}{findings_text}\nExtracted Text:\n\n{content}")
            else:
                preview.setPlainText(f"{file_info['name']} has not been parsed yet.")

        file_list.itemSelectionChanged.connect(show_selected_file)
        if self.uploaded_files:
            file_list.setCurrentRow(0)

        review_layout.addWidget(file_list)
        review_layout.addWidget(preview, 1)
        layout.addLayout(review_layout)

        layout.addStretch()
    
    def show_compliance_page(self):
        """Display compliance page"""
        layout = self.content_layout
        layout.setContentsMargins(40, 40, 40, 40)
        
        title = QLabel("Compliance Panel")
        title.setObjectName("title")
        layout.addWidget(title)
        
        subtitle = QLabel("GDPR • HIPAA • DPA checks based on detected entities")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)
        
        layout.addSpacing(30)

        if not self.uploaded_files:
            layout.addWidget(QLabel("Upload and scan files to run compliance checks."))
            layout.addStretch()
            return

        idx = 0
        fi = self.uploaded_files[idx]
        findings = fi.get("findings", [])
        assessment = fi.get("compliance_assessment") or self._compliance_engine.assess(findings)

        for framework in ("GDPR", "HIPAA", "DPA"):
            result = assessment.get(framework, {})
            status = result.get("status", "Not Checked")
            card = QFrame()
            card.setObjectName("card")
            card_layout = QVBoxLayout(card)

            header = QHBoxLayout()
            label = QLabel(framework)
            label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
            header.addWidget(label)
            header.addStretch()
            color = "#22c55e" if status == "Pass" else "#fbbf24" if status == "Review" else "#ef4444"
            status_label = QLabel(status)
            status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
            header.addWidget(status_label)
            card_layout.addLayout(header)
            card_layout.addWidget(QLabel(result.get("detail", "")))
            card_layout.addWidget(QLabel(f"Action: {result.get('action', '')}"))
            layout.addWidget(card)

        layout.addSpacing(16)
        layout.addWidget(QLabel(f"Based on: {fi['name']} ({len(findings)} findings)"))
        layout.addStretch()
    
    def show_protection_page(self):
        """Display protection options page"""
        layout = self.content_layout
        layout.setContentsMargins(40, 40, 40, 40)
        
        title = QLabel("Protection Options")
        title.setObjectName("title")
        layout.addWidget(title)
        
        subtitle = QLabel("Mask • Redact • Export safe copies of scanned files")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)
        
        layout.addSpacing(30)
        
        for option, desc, action in [
            ("Mask (Partial)", "Hide part of sensitive values in text exports", lambda: self._open_redaction_with_mode("partial")),
            ("Full Redact", "Black out or replace all high-risk fields", lambda: self._open_redaction_with_mode("full")),
            ("Export Safe Copy", "Redact all uploaded files with findings", self._batch_redact_all),
        ]:
            card = QFrame()
            card.setObjectName("card")
            card_layout = QHBoxLayout(card)
            
            text_col = QVBoxLayout()
            label = QLabel(option)
            label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
            text_col.addWidget(label)
            text_col.addWidget(QLabel(desc))
            card_layout.addLayout(text_col)
            
            card_layout.addStretch()
            
            btn = QPushButton("Run")
            btn.setFixedWidth(120)
            btn.clicked.connect(action)
            card_layout.addWidget(btn)
            
            layout.addWidget(card)
        
        layout.addStretch()

    def _open_redaction_with_mode(self, mode: str):
        self._pending_redaction_mode = mode
        self.switch_page("redact")

    def _batch_redact_all(self):
        self.switch_page("redact")
        if hasattr(self, "_run_batch_redaction"):
            self._run_batch_redaction()
    
    def show_detection_page(self):
        """Display detection engines page"""
        layout = self.content_layout
        layout.setContentsMargins(40, 40, 40, 40)
        
        title = QLabel("Detection Design")
        title.setObjectName("title")
        layout.addWidget(title)
        
        subtitle = QLabel("Engines • Open-source models")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)
        
        layout.addSpacing(30)
        
        # Detection engines
        engines_title = QLabel("Detection Engines")
        engines_title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(engines_title)
        
        for engine, desc in [
            ("Regex Engine", "Pattern matching - SSN, CC, email, phone formats"),
            ("GLiNER NER", "Graph-enhanced NER - GLiNER"),
            ("Transformer AI", "DistilBERT - Contextual semantic detection"),
        ]:
            card = QFrame()
            card.setObjectName("card")
            card_layout = QHBoxLayout(card)
            
            label = QLabel(engine)
            label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
            card_layout.addWidget(label)
            
            desc_label = QLabel(desc)
            desc_label.setStyleSheet("color: #a0aec0; font-size: 11px;")
            card_layout.addWidget(desc_label)
            
            layout.addWidget(card)
        
        layout.addSpacing(20)
        
        # Models
        models_title = QLabel("Open-source Models")
        models_title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(models_title)
        
        for model, desc in [
            ("DistilBERT", "Transformer - 256 MB, ~42ms avg"),
            ("GLiNER", "Graph-augmented NER - model dependent"),
            ("Presidio Analyzer", "Recognizer-based contextual detection"),
        ]:
            card = QFrame()
            card.setObjectName("card")
            card_layout = QHBoxLayout(card)
            
            status = QLabel("✓ Loaded")
            status.setStyleSheet("color: #22c55e;")
            card_layout.addWidget(status)
            
            label = QLabel(model)
            label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
            card_layout.addWidget(label)
            
            desc_label = QLabel(desc)
            desc_label.setStyleSheet("color: #a0aec0; font-size: 11px;")
            card_layout.addWidget(desc_label)
            
            layout.addWidget(card)
        
        layout.addStretch()

    def prompt_image_ocr_settings(self, file_path: Path) -> dict | None:
        """Prompt the user for OCR preprocessing options before running OCR."""
        dialog = ImageOCRPromptDialog(file_path, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.get_settings()

    def perform_image_ocr(self, file_path: Path) -> dict | None:
        """Run OCR for an uploaded image and store the result in the shared list."""
        settings = self.prompt_image_ocr_settings(file_path)
        if settings is None:
            return None

        worker = _OcrWorker(file_path, settings)
        thread = QThread(self)
        worker.moveToThread(thread)
        result_holder = {}
        _OCR_PENDING = object()
        pending_info = FileHandler.get_file_info(file_path)
        pending_info['processing_status'] = 'OCR in progress'
        self._upsert_uploaded_file(pending_info)
        self.refresh_upload_file_list()
        self.refresh_findings_page()

        def _on_finished(payload):
            result_holder['success'] = payload
            thread.quit()

        def _on_failed(err: str):
            result_holder['error'] = err
            thread.quit()

        self._active_ocr_threads.append(thread)
        self._active_ocr_workers.append(worker)
        worker.finished.connect(_on_finished)
        worker.failed.connect(_on_failed)
        thread.started.connect(worker.run)
        thread.finished.connect(lambda: self._finalize_ocr(file_path, result_holder, settings))
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(lambda: self._forget_ocr_job(thread, worker))
        thread.finished.connect(thread.deleteLater)
        thread.start()
        return _OCR_PENDING

    def _forget_ocr_job(self, thread: QThread, worker: _OcrWorker):
        if thread in self._active_ocr_threads:
            self._active_ocr_threads.remove(thread)
        if worker in self._active_ocr_workers:
            self._active_ocr_workers.remove(worker)

    def _finalize_ocr(self, file_path: Path, result_holder: dict, settings: dict):
        if not self.isVisible():
            return
        if 'error' in result_holder:
            logger.error("OCR failed for %s: %s", file_path.name, result_holder['error'])
            file_info = FileHandler.get_file_info(file_path)
            file_info['parse_error'] = result_holder['error']
            self._upsert_uploaded_file(file_info)
            self.refresh_upload_file_list()
            self.refresh_findings_page()
            return
        if 'success' not in result_holder:
            logger.error("OCR failed for %s: worker finished without a result", file_path.name)
            file_info = FileHandler.get_file_info(file_path)
            file_info['parse_error'] = "OCR worker finished without a result"
            self._upsert_uploaded_file(file_info)
            self.refresh_upload_file_list()
            self.refresh_findings_page()
            return
        payload = result_holder['success']
        result = payload['ocr']
        validation = payload['validation']
        findings = validation['findings']

        ocr_record = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'text': result.get('text', ''),
            'confidence': result.get('confidence', 0),
            'word_count': result.get('word_count', 0),
            'word_boxes': result.get('word_boxes', []),
            'engine': result.get('engine', 'paddleocr'),
            'fallback_used': result.get('fallback_used', False),
            'preprocessing_steps': result.get('preprocessing_steps_used', settings['selected_steps']),
            'language': settings['language'],
            'scan_mode': settings['scan_mode'],
            'validation_tier': validation['validation_tier'],
            'confidence_breakdown': validation['confidence_breakdown'],
            'findings': findings,
            'findings_summary': validation['summary'],
            'regex_findings': findings,
            'regex_summary': validation['summary'],
        }

        file_info = self.store_ocr_result_as_uploaded_file(ocr_record)
        self.refresh_upload_file_list()
        self.refresh_findings_page()
        logger.info(
            "OCR completed for %s: %s words, %s findings",
            file_path.name,
            ocr_record['word_count'],
            len(findings),
        )
        return file_info

    def _upsert_uploaded_file(self, file_info: dict):
        existing_index = next(
            (
                index
                for index, uploaded in enumerate(self.uploaded_files)
                if uploaded.get('path') == file_info.get('path')
            ),
            None
        )
        if existing_index is None:
            self.uploaded_files.append(file_info)
        else:
            self.uploaded_files[existing_index] = file_info

    def refresh_upload_file_list(self):
        file_list = getattr(self, "file_list", None)
        if file_list is None:
            return
        try:
            file_list.refresh_list()
        except RuntimeError:
            self.file_list = None

    def refresh_findings_page(self):
        """Refresh the Findings page list and displayed details."""
        if self.findings_file_list is None:
            return

        try:
            self.findings_file_list.clear()
        except RuntimeError:
            self.findings_file_list = None
            self.findings_findings_output = None
            self.findings_text_output = None
            return
        for index, file_info in enumerate(self.uploaded_files):
            if 'parse_error' in file_info:
                item_text = f"{file_info['name']} - Parse Error"
            elif 'parsed_content' in file_info:
                finding_count = len(file_info.get('regex_findings', []))
                item_text = f"{file_info['name']} - {finding_count} findings"
            elif file_info.get('processing_status'):
                item_text = f"{file_info['name']} - {file_info['processing_status']}"
            else:
                item_text = f"{file_info['name']} - Pending"

            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, index)
            self.findings_file_list.addItem(item)

        if self.uploaded_files:
            self.findings_file_list.setCurrentRow(0)
        else:
            if self.findings_findings_output is not None:
                self.findings_findings_output.setPlainText("No uploaded files yet.")
            if self.findings_text_output is not None:
                self.findings_text_output.clear()

    def show_findings_page(self):
        """Display the unified findings page."""
        layout = self.content_layout
        layout.setContentsMargins(40, 40, 40, 40)

        title = QLabel("Findings")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel("Uploaded files, OCR output, and detection findings in one place")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

        layout.addSpacing(24)

        review_layout = QHBoxLayout()

        self.findings_file_list = QListWidget()
        self.findings_file_list.setMinimumWidth(320)
        self.findings_file_list.setMaximumWidth(420)

        right_card = QFrame()
        right_card.setObjectName("card")
        right_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_layout = QVBoxLayout(right_card)

        findings_title = QLabel("Findings")
        findings_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        right_layout.addWidget(findings_title)

        self.findings_findings_output = QTextEdit()
        self.findings_findings_output.setReadOnly(True)
        self.findings_findings_output.setPlaceholderText("Select a file to view findings.")
        right_layout.addWidget(self.findings_findings_output, 1)

        extracted_title = QLabel("Extracted Text")
        extracted_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        right_layout.addWidget(extracted_title)

        self.findings_text_output = QTextEdit()
        self.findings_text_output.setReadOnly(True)
        self.findings_text_output.setPlaceholderText("Select a file to view extracted text.")
        right_layout.addWidget(self.findings_text_output, 2)

        def show_selected_finding():
            selected_items = self.findings_file_list.selectedItems()
            if not selected_items:
                return

            file_index = selected_items[0].data(Qt.ItemDataRole.UserRole)
            if file_index is None:
                return

            file_info = self.uploaded_files[file_index]
            self.display_finding_details(file_info)

        self.findings_file_list.itemSelectionChanged.connect(show_selected_finding)
        self.refresh_findings_page()

        review_layout.addWidget(self.findings_file_list)
        review_layout.addWidget(right_card, 1)
        layout.addLayout(review_layout)

        layout.addStretch()

    def display_finding_details(self, file_info: dict):
        """Render findings and extracted text for a selected uploaded file."""
        if self.findings_findings_output is None or self.findings_text_output is None:
            return

        if 'parse_error' in file_info:
            self.findings_findings_output.setPlainText(
                f"Could not process {file_info['name']}.\n\nError: {file_info['parse_error']}"
            )
            self.findings_text_output.clear()
            return

        content = file_info.get('parsed_content', '')
        findings = file_info.get('regex_findings', [])
        metadata = file_info.get('parsed_metadata', {})

        MAX_FINDINGS_PREVIEW = 100
        total_findings = len(findings)
        if total_findings > MAX_FINDINGS_PREVIEW:
            findings_to_show = findings[:MAX_FINDINGS_PREVIEW]
            truncated_msg = f"\n\n... and {total_findings - MAX_FINDINGS_PREVIEW} more findings (showing first {MAX_FINDINGS_PREVIEW})"
        else:
            findings_to_show = findings
            truncated_msg = ""

        findings_lines = []
        if findings_to_show:
            for index, finding in enumerate(findings_to_show, start=1):
                confidence = finding.get('consensus_score', finding.get('highest_confidence', finding.get('confidence')))
                confidence_parts = []
                if isinstance(confidence, (int, float)):
                    confidence_parts.append(f"confidence {confidence * 100:.1f}%" if confidence <= 1.0 else f"confidence {confidence:.1f}%")
                transformer_conf = finding.get('transformer_confidence')
                if isinstance(transformer_conf, (int, float)):
                    confidence_parts.append(f"transformer {transformer_conf * 100:.1f}%" if transformer_conf <= 1.0 else f"transformer {transformer_conf:.1f}%")
                confidence_str = f" [{', '.join(confidence_parts)}]" if confidence_parts else ""
                findings_lines.append(
                    f"{index}. {finding.get('type', 'Unknown')} ({finding.get('severity', 'medium')}) "
                    f"line {finding.get('line', '?')}: {finding.get('masked_value', finding.get('value', ''))}{confidence_str}\n"
                    f"   Context: {finding.get('context', '')}"
                )
        else:
            findings_lines.append("No findings detected.")

        metadata_lines = [
            f"{key}: {value}"
            for key, value in metadata.items()
            if value not in (None, "")
        ]

        findings_text = "\n\n".join([
            "Findings:",
            "\n".join(findings_lines) + truncated_msg,
            "Metadata:",
            "\n".join(metadata_lines) if metadata_lines else "No metadata available.",
        ])

        self.findings_findings_output.setPlainText(findings_text)
        self.findings_text_output.setPlainText(content or "No extracted text available.")

    def finalize_file_scan(self, file_info: dict) -> dict:
        """Compute risk, hash the file, and persist sensitive-data findings."""
        findings = file_info.get('findings', file_info.get('regex_findings', []))
        assessment = self._risk_manager.assess(findings)
        file_info['risk_assessment'] = assessment
        file_info['compliance_assessment'] = self._compliance_engine.assess(findings)
        file_info.setdefault('selected_finding_indices', list(range(len(findings))))
        try:
            self._sensitive_tracker.persist_file_scan(file_info, findings, assessment)
            if file_info.get("duplicate_of"):
                logger.info(
                    "Duplicate content detected: %s matches prior file %s",
                    file_info["name"],
                    file_info["duplicate_of"],
                )
        except Exception as exc:
            logger.warning("Sensitive data tracking skipped: %s", exc)
        return assessment

    def _complete_redaction(self, fi: dict, result: dict, strategy: str, mode: str, output) -> str:
        if result.get("skipped"):
            return "No findings matched the selected redaction mode."

        db = get_db()
        file_id = fi.get("file_id")
        if not file_id:
            row = db.get_file_by_path(fi["path"])
            file_id = row["id"] if row else None

        if file_id:
            self._sensitive_tracker.log_redaction(
                file_id=file_id,
                output_path=result.get("output_path", ""),
                strategy=strategy,
                mode=mode,
                findings_redacted=result.get("findings_redacted", 0),
            )
            db.update_file_risk_metadata(
                file_id=file_id,
                risk_score=fi.get("risk_assessment", {}).get("risk_score", 0),
                risk_level=fi.get("risk_assessment", {}).get("risk_level", "low"),
                redacted_path=result.get("output_path"),
                redaction_strategy=strategy,
            )
            fi["redacted_path"] = result.get("output_path")

            if self._app_settings.get("rescan_after_redaction", True) and result.get("output_path"):
                try:
                    pipeline = self._get_validation_pipeline()
                    if pipeline:
                        post = self._sensitive_tracker.rescan_redacted_file(
                            file_id, result["output_path"], self._risk_manager, pipeline,
                        )
                        fi["risk_assessment"]["post_redaction"] = post["assessment"]
                except Exception as exc:
                    logger.warning("Post-redaction rescan failed: %s", exc)

        post = fi.get("risk_assessment", {}).get("post_redaction", {})
        lines = [
            "Redaction complete.",
            f"Output folder: {self._redaction_output_dir}",
            f"Output file: {result.get('output_path')}",
            f"Findings redacted: {result.get('findings_redacted', 0)}",
            f"Strategy: {strategy} / Mode: {mode}",
        ]
        if post:
            lines.append(
                f"Post-redaction risk: {post.get('risk_level', 'unknown')} "
                f"({post.get('risk_score', 0)}) — {post.get('total_findings', 0)} findings remain"
            )
        if fi.get("duplicate_of"):
            lines.append(f"Note: duplicate of previously scanned file '{fi['duplicate_of']}'")
        return "\n".join(lines)

    def store_ocr_result_as_uploaded_file(self, result: dict):
        """Add or update an OCR image in the shared uploaded file list."""
        file_path = Path(result['file_path'])
        file_info = FileHandler.get_file_info(file_path)
        file_info['parsed_content'] = result['text']
        file_info['parsed_metadata'] = {
            'filename': result['file_name'],
            'file_size': file_info['size'],
            'source': 'ocr',
            'ocr_confidence': result['confidence'],
            'word_count': result['word_count'],
            'language': result['language'],
            'preprocessing_steps': result['preprocessing_steps'],
            'scan_mode': result.get('scan_mode'),
            'validation_tier': result.get('validation_tier'),
            'confidence_breakdown': result.get('confidence_breakdown'),
        }
        file_info['ocr_result'] = result
        file_info['findings'] = result.get('findings', result.get('regex_findings', []))
        file_info['findings_summary'] = result.get('findings_summary', result.get('regex_summary', {}))
        file_info['regex_findings'] = file_info['findings']
        file_info['regex_summary'] = file_info['findings_summary']

        self.finalize_file_scan(file_info)
        self._upsert_uploaded_file(file_info)
        return file_info

    def show_risk_dashboard(self):
        layout = self.content_layout
        layout.setContentsMargins(40, 40, 40, 40)

        title = QLabel("Privacy Risk Dashboard")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel("Risk scores, entity breakdown, and redaction coverage")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)
        layout.addSpacing(24)

        analytics = {}
        try:
            analytics = get_db().get_risk_analytics()
        except Exception as exc:
            logger.warning("Risk analytics unavailable: %s", exc)

        risk_dist = analytics.get("risk_distribution", {})
        high_count = risk_dist.get("high", 0)
        medium_count = risk_dist.get("medium", 0)
        redacted_count = analytics.get("total_redacted_documents", 0)
        tracking = {}
        try:
            tracking = get_db().get_entity_tracking_summary()
        except Exception as exc:
            logger.warning("Entity tracking summary unavailable: %s", exc)

        stats_layout = QHBoxLayout()
        for label_text, value, icon in [
            ("High-Risk Docs", str(high_count), "🔴"),
            ("Medium-Risk Docs", str(medium_count), "🟡"),
            ("Redacted Docs", str(redacted_count), "🛡️"),
            ("Tracked Entities", str(tracking.get("total_tracked_entities", 0)), "🔎"),
        ]:
            card = QFrame()
            card.setObjectName("card")
            cl = QVBoxLayout(card)
            cl.addWidget(QLabel(label_text))
            val_label = QLabel(value)
            val_label.setObjectName("stat-value")
            cl.addWidget(val_label)
            stats_layout.addWidget(card)
        layout.addLayout(stats_layout)
        layout.addSpacing(24)

        top_title = QLabel("Top Detected Entity Types")
        top_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(top_title)

        top_entities = analytics.get("top_entity_types", {})
        if top_entities:
            for entity, count in list(top_entities.items())[:8]:
                row = QHBoxLayout()
                row.addWidget(QLabel(entity))
                row.addStretch()
                row.addWidget(QLabel(f"{count} occurrences"))
                layout.addLayout(row)
        else:
            layout.addWidget(QLabel("No entity data available yet."))

        layout.addSpacing(20)
        tracking_title = QLabel("Sensitive Data Tracking")
        tracking_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(tracking_title)

        hashed_files = tracking.get("hashed_files", 0)
        cross_file = tracking.get("cross_file_entities", [])
        layout.addWidget(QLabel(f"Files with content hashes: {hashed_files}"))
        if cross_file:
            layout.addWidget(QLabel("Entities appearing in multiple files:"))
            for item in cross_file[:8]:
                row = QHBoxLayout()
                row.addWidget(QLabel(f"{item.get('entity_type', 'Unknown')}"))
                row.addStretch()
                row.addWidget(
                    QLabel(
                        f"{item.get('file_count', 0)} files · "
                        f"{item.get('occurrences', 0)} hits · "
                        f"hash {str(item.get('entity_hash', ''))[:10]}…"
                    )
                )
                layout.addLayout(row)
        else:
            layout.addWidget(QLabel("No cross-file entity matches recorded yet."))

        layout.addStretch()

    def show_settings_page(self):
        layout = self.content_layout
        layout.setContentsMargins(40, 40, 40, 40)

        title = QLabel("Settings")
        title.setObjectName("title")
        layout.addWidget(title)
        subtitle = QLabel("Defaults for scanning, redaction, and database session restore")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)
        layout.addSpacing(20)

        scan_combo = QComboBox()
        scan_combo.addItem("Quick Scan", "quick")
        scan_combo.addItem("Standard", "standard")
        scan_combo.addItem("Deep Analysis", "deep")
        current_scan = self._app_settings.get("default_scan_mode", "standard")
        scan_combo.setCurrentIndex(max(0, ["quick", "standard", "deep"].index(current_scan)))

        strategy_combo = QComboBox()
        strategy_combo.addItems(["blackout", "pixelate", "blur"])
        strategy_combo.setCurrentText(self._app_settings.get("default_redaction_strategy", "blackout"))

        mode_combo = QComboBox()
        mode_combo.addItems(["auto", "full", "partial"])
        mode_combo.setCurrentText(self._app_settings.get("default_redaction_mode", "auto"))

        output_input = QLineEdit(self._redaction_output_dir)
        reload_check = QCheckBox("Reload previous session from database on startup")
        reload_check.setChecked(bool(self._app_settings.get("reload_session_on_startup", True)))
        rescan_check = QCheckBox("Re-scan redacted files and update post-redaction risk")
        rescan_check.setChecked(bool(self._app_settings.get("rescan_after_redaction", True)))

        form = QVBoxLayout()
        for label, widget in [
            ("Default document scan mode", scan_combo),
            ("Default redaction strategy", strategy_combo),
            ("Default redaction mode", mode_combo),
            ("Default redaction output folder", output_input),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addWidget(widget, 1)
            form.addLayout(row)
        form.addWidget(reload_check)
        form.addWidget(rescan_check)
        layout.addLayout(form)

        status = QLabel("")
        layout.addWidget(status)

        def save_settings():
            out_dir = output_input.text().strip() or self._redaction_output_dir
            self._app_settings.save_many({
                "default_scan_mode": scan_combo.currentData(),
                "default_redaction_strategy": strategy_combo.currentText(),
                "default_redaction_mode": mode_combo.currentText(),
                "redaction_output_dir": out_dir,
                "reload_session_on_startup": reload_check.isChecked(),
                "rescan_after_redaction": rescan_check.isChecked(),
            })
            self._redaction_output_dir = out_dir
            self._redaction_engine.set_output_dir(out_dir)
            status.setText("Settings saved to database.")

        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(save_settings)
        layout.addWidget(save_btn)
        layout.addStretch()

    def show_redact_page(self):
        layout = self.content_layout
        layout.setContentsMargins(40, 40, 40, 40)

        title = QLabel("Redaction Studio")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel("Generate redacted copies sensitive data")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)
        layout.addSpacing(20)

        if not self.uploaded_files:
            layout.addWidget(QLabel("No files uploaded. Upload files first."))
            layout.addStretch()
            return

        file_list = QListWidget()
        file_list.setMinimumWidth(280)
        for fi in self.uploaded_files:
            risk = fi.get('risk_assessment', {}).get('risk_level', 'unknown')
            file_list.addItem(f"{fi['name']} [{risk.upper()}]")

        findings_list = QListWidget()
        findings_list.setMinimumWidth(360)

        def refresh_findings_panel():
            findings_list.clear()
            idx = file_list.currentRow()
            if idx < 0 or idx >= len(self.uploaded_files):
                return
            fi = self.uploaded_files[idx]
            findings = fi.get("findings", [])
            fi.setdefault("selected_finding_indices", list(range(len(findings))))
            for i, finding in enumerate(findings):
                checked = i in fi["selected_finding_indices"]
                label = (
                    f"{'☑' if checked else '☐'} {finding.get('type', '?')}: "
                    f"{finding.get('masked_value', finding.get('value', ''))}"
                )
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, i)
                findings_list.addItem(item)

        def toggle_finding(item: QListWidgetItem):
            idx = file_list.currentRow()
            if idx < 0:
                return
            fi = self.uploaded_files[idx]
            finding_idx = item.data(Qt.ItemDataRole.UserRole)
            selected = set(fi.get("selected_finding_indices", []))
            if finding_idx in selected:
                selected.discard(finding_idx)
            else:
                selected.add(finding_idx)
            fi["selected_finding_indices"] = sorted(selected)
            refresh_findings_panel()

        file_list.currentRowChanged.connect(lambda _row: refresh_findings_panel())
        findings_list.itemClicked.connect(toggle_finding)

        lists_row = QHBoxLayout()
        lists_row.addWidget(file_list)
        lists_row.addWidget(findings_list, 1)
        layout.addLayout(lists_row)

        strat_layout = QHBoxLayout()
        strat_layout.addWidget(QLabel("Strategy:"))
        strategy_combo = QComboBox()
        strategy_combo.addItems(["blackout", "pixelate", "blur"])
        strategy_combo.setCurrentText(self._app_settings.get("default_redaction_strategy", "blackout"))
        strat_layout.addWidget(strategy_combo)
        strat_layout.addSpacing(16)
        strat_layout.addWidget(QLabel("Mode:"))
        mode_combo = QComboBox()
        mode_combo.addItems(["auto", "full", "partial"])
        pending_mode = getattr(self, "_pending_redaction_mode", None)
        mode_combo.setCurrentText(pending_mode or self._app_settings.get("default_redaction_mode", "auto"))
        self._pending_redaction_mode = None
        strat_layout.addWidget(mode_combo)
        layout.addLayout(strat_layout)

        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(QLabel("Output folder:"))
        output_dir_input = QLineEdit(self._redaction_output_dir)
        output_dir_input.setPlaceholderText("Choose where redacted files are saved")
        output_dir_layout.addWidget(output_dir_input, 1)

        def browse_output_dir():
            start_dir = output_dir_input.text().strip() or self._redaction_output_dir
            chosen = QFileDialog.getExistingDirectory(
                self,
                "Select Redacted Output Folder",
                start_dir,
            )
            if chosen:
                output_dir_input.setText(chosen)

        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(browse_output_dir)
        output_dir_layout.addWidget(browse_btn)
        layout.addLayout(output_dir_layout)

        output = QTextEdit()
        output.setReadOnly(True)
        output.setPlaceholderText("Redaction output will appear here...")
        layout.addWidget(output, 1)

        def run_redaction_for_index(idx: int, strategy: str, mode: str) -> str | None:
            if idx < 0 or idx >= len(self.uploaded_files):
                return "Select a file first."
            fi = self.uploaded_files[idx]
            findings = self._get_selected_findings(fi)
            if not findings:
                return "No findings selected to redact."
            fmt = fi.get('format', '').lower()
            path = fi.get('path', '')
            output_dir = output_dir_input.text().strip()
            if not output_dir:
                return "Please choose an output folder for redacted files."
            self._redaction_output_dir = output_dir
            self._redaction_engine.set_output_dir(output_dir)
            self._app_settings.set("redaction_output_dir", output_dir)
            if fmt not in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.pdf', '.docx', '.txt', '.sql'}:
                return "Redaction for this file type is not yet supported."
            result = run_redaction_for_file(
                self._redaction_engine, fi, findings, strategy, mode,
            )
            return self._complete_redaction(fi, result, strategy, mode, output)

        def run_redaction():
            idx = file_list.currentRow()
            strategy = strategy_combo.currentText()
            mode = mode_combo.currentText()
            try:
                message = run_redaction_for_index(idx, strategy, mode)
                output.setText(message or "Redaction finished.")
            except Exception as exc:
                output.setText(f"Redaction failed: {exc}")

        def run_batch_redaction():
            strategy = strategy_combo.currentText()
            mode = mode_combo.currentText()
            lines = []
            for idx, fi in enumerate(self.uploaded_files):
                if not fi.get("findings"):
                    continue
                try:
                    msg = run_redaction_for_index(idx, strategy, mode)
                    if msg and "complete" in msg.lower():
                        lines.append(f"✓ {fi['name']}")
                    elif msg:
                        lines.append(f"– {fi['name']}: {msg}")
                except Exception as exc:
                    lines.append(f"✗ {fi['name']}: {exc}")
            output.setText("\n".join(lines) if lines else "No files with findings to redact.")

        self._run_batch_redaction = run_batch_redaction

        btn_layout = QHBoxLayout()
        redact_btn = QPushButton("Generate Redacted Copy")
        redact_btn.clicked.connect(run_redaction)
        btn_layout.addWidget(redact_btn)
        batch_btn = QPushButton("Redact All Files")
        batch_btn.clicked.connect(run_batch_redaction)
        btn_layout.addWidget(batch_btn)
        layout.addLayout(btn_layout)

        if self.uploaded_files:
            file_list.setCurrentRow(0)
            refresh_findings_panel()
        layout.addStretch()

    def show_reports_page(self):
        """Display reports page"""
        layout = self.content_layout
        layout.setContentsMargins(40, 40, 40, 40)
        
        title = QLabel("Reports")
        title.setObjectName("title")
        layout.addWidget(title)
        
        subtitle = QLabel("Scan history and analytics")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)
        
        layout.addSpacing(30)
        
        # Database retrieval buttons
        buttons_label = QLabel("Database Operations")
        buttons_label.setObjectName("subtitle")
        layout.addWidget(buttons_label)
        
        buttons_layout = QHBoxLayout()
        
        # View all files button
        btn_all_files = QPushButton("View All Files")
        btn_all_files.clicked.connect(self.show_all_files_report)
        buttons_layout.addWidget(btn_all_files)
        
        # View recent files button
        btn_recent_files = QPushButton("View Recent Files (7 days)")
        btn_recent_files.clicked.connect(self.show_recent_files_report)
        buttons_layout.addWidget(btn_recent_files)
        
        # View high-risk detections button
        btn_high_risk = QPushButton("View High-Risk Detections")
        btn_high_risk.clicked.connect(self.show_high_risk_report)
        buttons_layout.addWidget(btn_high_risk)
        
        # View statistics button
        btn_stats = QPushButton("View Statistics")
        btn_stats.clicked.connect(self.show_statistics_report)
        buttons_layout.addWidget(btn_stats)
        
        # Export data button
        btn_export = QPushButton("Export to JSON")
        btn_export.clicked.connect(self.export_json_report)
        buttons_layout.addWidget(btn_export)

        btn_redactions = QPushButton("Redaction History (DB)")
        btn_redactions.clicked.connect(self.show_redaction_history_report)
        buttons_layout.addWidget(btn_redactions)

        btn_reload = QPushButton("Reload Session from DB")
        btn_reload.clicked.connect(self.reload_session_report)
        buttons_layout.addWidget(btn_reload)
        
        layout.addLayout(buttons_layout)
        layout.addSpacing(20)
        
        # Results display area
        self.reports_display = QTextEdit()
        self.reports_display.setReadOnly(True)
        self.reports_display.setStyleSheet("""
            QTextEdit {
                background-color: #1a1f2e;
                color: #e0e6ed;
                border: 1px solid #2a3f5f;
                border-radius: 4px;
                padding: 10px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
            }
        """)
        layout.addWidget(self.reports_display)
        
        layout.addStretch()
    
    def _db_files_as_report_rows(self) -> list:
        db = get_db()
        rows = db.get_all_files(limit=500)
        report_files = []
        for row in rows:
            findings = db.get_detections_for_file(row["id"])
            high = sum(1 for f in findings if str(f.get("risk_level", "")).lower() == "high")
            report_files.append({
                "name": row.get("filename"),
                "path": row.get("file_path"),
                "size": row.get("file_size", 0),
                "format": row.get("file_format"),
                "file_hash": row.get("file_hash"),
                "parse_error": row.get("error_message") if row.get("status") == "error" else None,
                "parsed_content": row.get("parsed_content_preview"),
                "regex_findings": findings,
                "risk_assessment": {
                    "risk_level": row.get("risk_level"),
                    "risk_score": row.get("risk_score"),
                    "post_redaction": {
                        "risk_level": row.get("post_redaction_risk_level"),
                        "risk_score": row.get("post_redaction_risk_score"),
                        "total_findings": row.get("post_redaction_findings_count"),
                    } if row.get("post_redaction_risk_score") is not None else None,
                },
                "_high_count": high,
                "_finding_count": len(findings),
                "redacted_path": row.get("redacted_version_path"),
            })
        return report_files

    def reload_session_report(self):
        self._reload_session_from_db()
        self.refresh_upload_file_list()
        self.refresh_findings_page()
        self.reports_display.setText(
            f"Reloaded {len(self.uploaded_files)} file(s) from database into the current session."
        )

    def show_redaction_history_report(self):
        history = get_db().get_redaction_history(limit=100)
        if not history:
            self.reports_display.setText("No redaction history in database.")
            return
        report = "REDACTION HISTORY (DATABASE)\n" + "=" * 80 + "\n\n"
        for i, row in enumerate(history, 1):
            report += (
                f"{i}. {row.get('filename')} → {row.get('output_path')}\n"
                f"   Strategy: {row.get('strategy')} | Mode: {row.get('mode')} | "
                f"Redacted: {row.get('findings_redacted')}\n"
                f"   When: {row.get('redaction_timestamp')}\n\n"
            )
        self.reports_display.setText(report)

    def show_all_files_report(self):
        """Display all files stored in the database."""
        report_files = self._db_files_as_report_rows()
        if not report_files:
            self.reports_display.setText("No files in database.")
            return

        report = "ALL FILES (DATABASE)\n"
        report += "=" * 80 + "\n\n"
        report += self.build_files_report(report_files)
        report += f"\nTotal files: {len(report_files)}"
        self.reports_display.setText(report)
    
    def show_recent_files_report(self):
        """Display files uploaded in the last 7 days from the database."""
        db_files = get_db().get_recent_files(days=7, limit=200)
        if not db_files:
            self.reports_display.setText("No files uploaded in the last 7 days.")
            return

        report = "RECENT FILES (DATABASE — LAST 7 DAYS)\n"
        report += "=" * 80 + "\n\n"
        for i, row in enumerate(db_files, 1):
            report += (
                f"{i}. {row.get('filename')}\n"
                f"   Path: {row.get('file_path')}\n"
                f"   Uploaded: {row.get('upload_date')}\n"
                f"   Risk: {row.get('risk_level')} ({row.get('risk_score')})\n"
                f"   Findings: {row.get('total_findings_count', 0)}\n"
                f"   Redacted: {row.get('redacted_version_path') or 'No'}\n\n"
            )
        report += f"Total: {len(db_files)}"
        self.reports_display.setText(report)
    
    def show_high_risk_report(self):
        """Display high-risk detections from the database."""
        detections = get_db().get_high_risk_detections(limit=200)
        if not detections:
            self.reports_display.setText("No high-risk detections in database.")
            return

        report = "HIGH-RISK DETECTIONS (DATABASE)\n"
        report += "=" * 80 + "\n\n"
        for i, finding in enumerate(detections, 1):
            report += (
                f"{i}. File: {finding.get('filename')}\n"
                f"   Type: {finding.get('entity_type') or finding.get('pattern_matched')}\n"
                f"   Data: {finding.get('data_found')}\n"
                f"   Location: {finding.get('location_info')}\n"
                f"   When: {finding.get('detection_timestamp')}\n\n"
            )
        report += f"\nTotal: {len(detections)}"
        self.reports_display.setText(report)
    
    def show_statistics_report(self):
        """Display database-wide statistics."""
        stats = get_db().get_overall_stats()
        risk = get_db().get_risk_analytics()
        tracking = get_db().get_entity_tracking_summary()

        report = "DATABASE STATISTICS\n"
        report += "=" * 80 + "\n\n"
        report += f"Total Files: {stats.get('total_files', 0)}\n"
        report += f"Total Storage: {stats.get('total_size_bytes', 0):,} bytes\n"
        report += f"Total Detections: {stats.get('total_detections', 0)}\n"
        report += f"Redacted Documents: {risk.get('total_redacted_documents', 0)}\n"
        report += f"Tracked Entities: {tracking.get('total_tracked_entities', 0)}\n"
        report += f"Hashed Files: {tracking.get('hashed_files', 0)}\n"

        report += "\nRisk Distribution (files):\n"
        for level, count in risk.get("risk_distribution", {}).items():
            report += f"  {level}: {count}\n"

        report += "\nDetection Risk Distribution:\n"
        for level, count in stats.get("risk_distribution", {}).items():
            report += f"  {level}: {count}\n"

        report += "\nTop Entity Types:\n"
        for entity, count in list(risk.get("top_entity_types", {}).items())[:10]:
            report += f"  {entity}: {count}\n"

        self.reports_display.setText(report)

    def build_files_report(self, files: list) -> str:
        """Build a text report for uploaded file records."""
        report = ""
        for i, file_info in enumerate(files, 1):
            status = "Parsed/Scanned" if 'parsed_content' in file_info else "Parse Error" if 'parse_error' in file_info else "Pending"
            findings = file_info.get('regex_findings', [])
            high_findings = sum(1 for finding in findings if finding.get('severity') == 'high')

            report += f"{i}. {file_info['name']}\n"
            report += f"   Path: {file_info['path']}\n"
            report += f"   Size: {file_info['size']:,} bytes\n"
            report += f"   Format: {file_info['format']}\n"
            report += f"   Status: {status}\n"
            if file_info.get('file_hash'):
                report += f"   SHA-256: {file_info['file_hash']}\n"
            risk = file_info.get('risk_assessment', {})
            if risk:
                report += f"   Risk: {risk.get('risk_level', 'unknown')} ({risk.get('risk_score', 0)})\n"
            report += f"   Findings: {file_info.get('_finding_count', len(findings))} total, {file_info.get('_high_count', high_findings)} high risk\n"
            if file_info.get('redacted_path'):
                report += f"   Redacted copy: {file_info['redacted_path']}\n"
            post = (file_info.get('risk_assessment') or {}).get('post_redaction')
            if post and post.get('risk_score') is not None:
                report += (
                    f"   Post-redaction risk: {post.get('risk_level')} "
                    f"({post.get('risk_score')}) — {post.get('total_findings', 0)} findings remain\n"
                )
            if 'parse_error' in file_info:
                report += f"   Error: {file_info['parse_error']}\n"
            report += "\n"
        return report
    
    def export_json_report(self):
        """Export database records to JSON file"""
        try:
            import json
            from datetime import datetime

            db = get_db()
            data = []
            for row in db.get_all_files(limit=1000):
                file_id = row["id"]
                detections = db.get_detections_for_file(file_id)
                redactions = db.get_redactions_for_file(file_id)
                ocr = db.get_ocr_result(file_id)
                data.append({
                    'id': file_id,
                    'name': row.get('filename'),
                    'path': row.get('file_path'),
                    'size': row.get('file_size'),
                    'format': row.get('file_format'),
                    'file_hash': row.get('file_hash'),
                    'upload_date': row.get('upload_date'),
                    'risk_score': row.get('risk_score'),
                    'risk_level': row.get('risk_level'),
                    'post_redaction_risk_score': row.get('post_redaction_risk_score'),
                    'post_redaction_risk_level': row.get('post_redaction_risk_level'),
                    'post_redaction_findings_count': row.get('post_redaction_findings_count'),
                    'redacted_path': row.get('redacted_version_path'),
                    'findings_count': row.get('total_findings_count'),
                    'entity_type_counts': row.get('entity_type_counts'),
                    'detections': detections,
                    'redactions': redactions,
                    'ocr_summary': {
                        'confidence': ocr.get('confidence') if ocr else None,
                        'word_count': ocr.get('word_count') if ocr else None,
                    } if ocr else None,
                })

            json_data = json.dumps(data, indent=2)
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_file = Path(f"database/export_{timestamp}.json")
            export_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_file, 'w') as f:
                f.write(json_data)
            
            # Display result
            report = f"EXPORT SUCCESSFUL\n"
            report += "=" * 80 + "\n\n"
            report += f"File exported to: {export_file}\n"
            report += f"Export timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            report += f"\nExported Records:\n"
            report += f"  Total files: {len(data)}\n"
            
            total_detections = sum(len(record.get('detections', [])) for record in data)
            report += f"  Total detections: {total_detections}\n"
            
            report += f"\nData is ready for analysis and backup."
            
            self.reports_display.setText(report)
            logger.info(f"Exported data to {export_file}")
        except Exception as e:
            error_msg = f"Error exporting data: {str(e)}"
            self.reports_display.setText(error_msg)
            logger.error(error_msg)
    
    def on_files_uploaded(self, files: list):
        """Handle files dropped/selected"""
        logger.info(f"Files received: {len(files)}")
        self.file_list.add_files(files)
        if self.current_page == "findings" and self.findings_file_list is not None:
            self.refresh_findings_page()

# ============================================================================
# MAIN
# ============================================================================

def main():
    # Setup
    setup_project_structure()
    setup_logging()
    
    logger = logging.getLogger(__name__)
    logger.info("Starting IntelliSafe application...")
                    
    # Create and run application
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
