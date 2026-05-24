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

# Load Torch before Qt to avoid a Windows DLL initialization conflict where
# importing PyQt6 first can prevent torch\lib\c10.dll from initializing.
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
    
    SUPPORTED_FORMATS = {'.pdf', '.png', '.jpg', '.jpeg', '.docx', '.sql', '.txt'}
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
    
    @staticmethod
    def validate_file(file_path: Path) -> tuple:
        """Validate file format and size"""
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
        """Get file information"""
        stat = file_path.stat()
        return {
            'name': file_path.name,
            'path': str(file_path),
            'size': stat.st_size,
            'format': file_path.suffix.lower(),
            'created': stat.st_ctime,
            'modified': stat.st_mtime,
        }

# ============================================================================
# PYQT6 IMPORTS AND COMPONENTS
# ============================================================================

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QLabel, QFrame, QFileDialog, QListWidget, QListWidgetItem, QScrollArea,
    QTextEdit, QCheckBox, QComboBox
)
from PyQt6.QtCore import Qt, QSize, QMimeData, QTimer, pyqtSignal, QObject, QThread
from PyQt6.QtGui import QIcon, QFont, QDrag, QColor, QDropEvent, QPixmap

logger = logging.getLogger(__name__)

# ============================================================================
# COMPONENTS
# ============================================================================

class UploadDropZone(QFrame):
    """Drag and drop zone for file uploads"""
    
    files_dropped = pyqtSignal(list)  # Signal when files are dropped
    
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
        self.init_ui()
        self.refresh_list()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Uploaded Files")
        title.setObjectName("title")
        layout.addWidget(title)
        
        # File list
        self.list_widget = QListWidget()
        self.list_widget.setMaximumHeight(300)
        layout.addWidget(self.list_widget)
        
        # Clear button
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.clear_files)
        layout.addWidget(clear_btn)
        
        layout.addStretch()
    
    def add_files(self, files: list):
        """Add files to the list and parse them"""
        # Import parser
        try:
            from app.backend.file_parser import FileParser
            from app.backend.detection import ValidationPipeline
        except ImportError:
            logger.error("Could not import FileParser or ValidationPipeline")
            return
        
        for file_path in files:
            # Validate file
            valid, msg = FileHandler.validate_file(file_path)
            if not valid:
                logger.warning(f"Invalid file {file_path}: {msg}")
                continue
            
            info = FileHandler.get_file_info(file_path)
            self.files.append(info)
            
            # Parse the file
            logger.info(f"Parsing file: {file_path.name}")
            parse_result = FileParser.parse(file_path)
            
            if parse_result['status'] == 'success':
                # Store parsed content in file info
                info['parsed_content'] = parse_result['content']
                info['parsed_metadata'] = parse_result['metadata']
                validator = ValidationPipeline()
                validation = validator.run(parse_result['content'], mode="standard")
                info['findings'] = validation['findings']
                info['findings_summary'] = validation['summary']
                info['validation_tier'] = validation['validation_tier']
                info['confidence_breakdown'] = validation['confidence_breakdown']
                info['regex_findings'] = validation['findings']
                info['regex_summary'] = validation['summary']
                logger.info(f"Successfully parsed: {file_path.name}")
            else:
                # Add error indicator
                logger.error(f"Failed to parse {file_path.name}: {parse_result.get('error', 'Unknown error')}")
                info['parse_error'] = parse_result.get('error', 'Unknown error')
            
            self.add_file_item(info)
            
            logger.info(f"File added to list: {info['name']}")

    def refresh_list(self):
        """Render the current uploaded files."""
        self.list_widget.clear()
        for info in self.files:
            self.add_file_item(info)

    def add_file_item(self, info: dict):
        """Add one uploaded file row to the list widget."""
        size_mb = info['size'] / 1024 / 1024
        if 'parse_error' in info:
            item_text = f"✗ {info['name']} ({size_mb:.2f} MB) - Parse Error"
        elif 'parsed_content' in info:
            finding_count = len(info.get('findings', info.get('regex_findings', [])))
            item_text = f"✓ {info['name']} ({size_mb:.2f} MB) - {finding_count} findings"
        else:
            item_text = f"{info['name']} ({size_mb:.2f} MB) - {info['format']}"

        item = QListWidgetItem(item_text)
        self.list_widget.addItem(item)
    
    def clear_files(self):
        """Clear all files"""
        self.files.clear()
        self.list_widget.clear()
        logger.info("Files cleared")

# ============================================================================
# MAIN WINDOW
# ============================================================================

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IntelliSafe - Sensitive Data Detection")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet(STYLESHEET)
        self.current_page = "upload"
        self.uploaded_files = []
        self.ocr_image_path = None
        self.ocr_results = []
        self.ocr_step_checkboxes = {}
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Add sidebar
        self.sidebar = self.create_sidebar()
        main_layout.addWidget(self.sidebar)
        
        # Add content area
        self.content_area = QWidget()
        self.content_layout = QVBoxLayout(self.content_area)
        main_layout.addWidget(self.content_area, 1)
        
        # Show initial page
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
            ("OCR", "ocr"),
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
        layout.addWidget(settings_btn)
        
        return sidebar
    
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
        elif page_id == "ocr":
            self.show_ocr_page()
        elif page_id == "reports":
            self.show_reports_page()

    def clear_layout(self, layout):
        """Remove all widgets and nested layouts from a layout."""
        while layout.count() > 0:
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            elif child.layout():
                self.clear_layout(child.layout())
                child.layout().deleteLater()
    
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
        self.file_list = FileListWidget(self.uploaded_files)
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
                        finding_lines.append(
                            f"{index}. {finding['type']} ({finding['severity']}) "
                            f"line {finding['line']}: {finding['masked_value']}\n"
                            f"   Context: {finding['context']}"
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
        
        subtitle = QLabel("GDPR • HIPAA • DPA checks")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)
        
        layout.addSpacing(30)
        
        # Compliance cards
        for compliance, status in [("GDPR", "Not Checked"), ("HIPAA", "Not Checked"), ("DPA", "Not Checked")]:
            card = QFrame()
            card.setObjectName("card")
            card_layout = QHBoxLayout(card)
            
            label = QLabel(compliance)
            label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
            card_layout.addWidget(label)
            
            card_layout.addStretch()
            
            status_label = QLabel(status)
            status_label.setStyleSheet("color: #a0aec0;")
            card_layout.addWidget(status_label)
            
            layout.addWidget(card)
        
        layout.addStretch()
    
    def show_protection_page(self):
        """Display protection options page"""
        layout = self.content_layout
        layout.setContentsMargins(40, 40, 40, 40)
        
        title = QLabel("Protection Options")
        title.setObjectName("title")
        layout.addWidget(title)
        
        subtitle = QLabel("Mask • AE-256 Encrypt • Redact • Export Safe Copy")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)
        
        layout.addSpacing(30)
        
        # Protection options
        for option, desc in [
            ("Mask", "Hide sensitive data with asterisks"),
            ("AE-256 Encrypt", "Encrypt sensitive data with AES-256"),
            ("Redact", "Completely remove sensitive information"),
            ("Export Safe Copy", "Export sanitized version of files"),
        ]:
            card = QFrame()
            card.setObjectName("card")
            card_layout = QHBoxLayout(card)
            
            label = QLabel(option)
            label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
            card_layout.addWidget(label)
            
            card_layout.addStretch()
            
            btn = QPushButton("Configure")
            btn.setFixedWidth(120)
            card_layout.addWidget(btn)
            
            layout.addWidget(card)
        
        layout.addStretch()
    
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
            ("spaCy NER", "Named entity recognition - en-core-web-md"),
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
            ("spaCy en-core-web-md", "NER Pipeline - 80 MB, ~45ms avg"),
            ("MiniLM", "Sentence Embeddings - 45 MB, ~8ms avg"),
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

    def show_ocr_page(self):
        """Display OCR processing page"""
        layout = self.content_layout
        layout.setContentsMargins(40, 40, 40, 40)

        title = QLabel("OCR Processing")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel("Extract text from image files using Tesseract OCR")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

        layout.addSpacing(24)

        controls_layout = QHBoxLayout()

        select_btn = QPushButton("Select Image")
        select_btn.clicked.connect(self.select_ocr_image)
        controls_layout.addWidget(select_btn)

        run_btn = QPushButton("Run OCR")
        run_btn.clicked.connect(self.run_ocr_processing)
        controls_layout.addWidget(run_btn)

        language_label = QLabel("Language")
        controls_layout.addWidget(language_label)

        self.ocr_language_combo = QComboBox()
        for language in ["eng", "eng+spa", "spa", "fra", "deu"]:
            self.ocr_language_combo.addItem(language)
        controls_layout.addWidget(self.ocr_language_combo)

        scan_mode_label = QLabel("Scan Mode")
        controls_layout.addWidget(scan_mode_label)

        self.ocr_scan_mode_combo = QComboBox()
        self.ocr_scan_mode_combo.addItem("Quick Scan", "quick")
        self.ocr_scan_mode_combo.addItem("Standard", "standard")
        self.ocr_scan_mode_combo.addItem("Deep Analysis", "deep")
        self.ocr_scan_mode_combo.setCurrentIndex(1)
        controls_layout.addWidget(self.ocr_scan_mode_combo)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        self.ocr_status_label = QLabel("Select an image to begin.")
        self.ocr_status_label.setObjectName("subtitle")
        layout.addWidget(self.ocr_status_label)

        if self.ocr_image_path:
            self.ocr_status_label.setText(f"Selected: {Path(self.ocr_image_path).name}")

        layout.addSpacing(16)

        steps_title = QLabel("Preprocessing")
        steps_title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(steps_title)

        steps_layout = QHBoxLayout()
        self.ocr_step_checkboxes = {}
        default_steps = {"grayscale", "threshold_otsu", "sharpen"}
        for step, label in [
            ("grayscale", "Grayscale"),
            ("threshold_otsu", "Otsu Threshold"),
            ("threshold_adaptive", "Adaptive Threshold"),
            ("denoise_bilateral", "Denoise"),
            ("denoise_morphological", "Morphology"),
            ("sharpen", "Sharpen"),
            ("invert", "Invert"),
        ]:
            checkbox = QCheckBox(label)
            checkbox.setChecked(step in default_steps)
            self.ocr_step_checkboxes[step] = checkbox
            steps_layout.addWidget(checkbox)

        steps_layout.addStretch()
        layout.addLayout(steps_layout)

        layout.addSpacing(20)

        content_layout = QHBoxLayout()

        preview_card = QFrame()
        preview_card.setObjectName("card")
        preview_layout = QVBoxLayout(preview_card)

        preview_title = QLabel("Image Preview")
        preview_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        preview_layout.addWidget(preview_title)

        self.ocr_image_preview = QLabel("No image selected")
        self.ocr_image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ocr_image_preview.setMinimumHeight(320)
        self.ocr_image_preview.setStyleSheet("border: 1px solid #334155; color: #a0aec0;")
        preview_layout.addWidget(self.ocr_image_preview)

        content_layout.addWidget(preview_card, 1)

        output_card = QFrame()
        output_card.setObjectName("card")
        output_layout = QVBoxLayout(output_card)

        output_title = QLabel("Extracted Text")
        output_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        output_layout.addWidget(output_title)

        self.ocr_text_output = QTextEdit()
        self.ocr_text_output.setReadOnly(True)
        self.ocr_text_output.setPlaceholderText("OCR output will appear here.")
        output_layout.addWidget(self.ocr_text_output)

        content_layout.addWidget(output_card, 1)
        layout.addLayout(content_layout)

        if self.ocr_image_path:
            self.update_ocr_image_preview()

        if self.ocr_results:
            latest = self.ocr_results[-1]
            self.display_ocr_result(latest)

        layout.addStretch()

    def select_ocr_image(self):
        """Select an image for OCR processing."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select image for OCR",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;All Files (*)"
        )
        if not file_path:
            return

        self.ocr_image_path = file_path
        self.ocr_status_label.setText(f"Selected: {Path(file_path).name}")
        self.ocr_text_output.clear()
        self.update_ocr_image_preview()

    def update_ocr_image_preview(self):
        """Refresh the OCR image preview."""
        if not self.ocr_image_path:
            return

        pixmap = QPixmap(self.ocr_image_path)
        if pixmap.isNull():
            self.ocr_image_preview.setText("Could not preview selected image.")
            return

        scaled = pixmap.scaled(
            self.ocr_image_preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.ocr_image_preview.setPixmap(scaled)

    def run_ocr_processing(self):
        """Run OCR and store results in the shared file state."""
        if not self.ocr_image_path:
            self.ocr_status_label.setText("Select an image before running OCR.")
            return

        selected_steps = [
            step
            for step, checkbox in self.ocr_step_checkboxes.items()
            if checkbox.isChecked()
        ]
        if not selected_steps:
            selected_steps = ["grayscale"]

        language = self.ocr_language_combo.currentText()

        try:
            from app.backend.ocr_processor import OCRPipeline
            from app.backend.detection import ValidationPipeline

            self.ocr_status_label.setText("Running OCR...")
            QApplication.processEvents()

            pipeline = OCRPipeline(language=language)
            result = pipeline.process(self.ocr_image_path, selected_steps)
            scan_mode = self.ocr_scan_mode_combo.currentData()
            if not hasattr(self, "validation_pipeline"):
                self.validation_pipeline = ValidationPipeline()
            validation = self.validation_pipeline.run(result.get('text', ''), mode=scan_mode)
            findings = validation['findings']

            ocr_record = {
                'file_path': self.ocr_image_path,
                'file_name': Path(self.ocr_image_path).name,
                'text': result.get('text', ''),
                'confidence': result.get('confidence', 0),
                'word_count': result.get('word_count', 0),
                'engine': result.get('engine', 'tesseract'),
                'fallback_used': result.get('fallback_used', False),
                'preprocessing_steps': result.get('preprocessing_steps_used', selected_steps),
                'language': language,
                'scan_mode': scan_mode,
                'validation_tier': validation['validation_tier'],
                'confidence_breakdown': validation['confidence_breakdown'],
                'findings': findings,
                'findings_summary': validation['summary'],
                'regex_findings': findings,
                'regex_summary': validation['summary'],
            }
            self.ocr_results.append(ocr_record)
            self.store_ocr_result_as_uploaded_file(ocr_record)
            self.display_ocr_result(ocr_record)

            self.ocr_status_label.setText(
                f"OCR complete: {ocr_record['word_count']} words, "
                f"{ocr_record['confidence']:.1f}% confidence, "
                f"{len(findings)} findings ({scan_mode})"
            )
            logger.info(f"OCR completed for {ocr_record['file_name']}")
        except Exception as e:
            error_msg = f"OCR failed: {str(e)}"
            self.ocr_status_label.setText(error_msg)
            self.ocr_text_output.setPlainText(error_msg)
            logger.error(error_msg)

    def display_ocr_result(self, result: dict):
        """Display OCR result details."""
        finding_lines = []
        findings = result.get('findings', result.get('regex_findings', []))
        for index, finding in enumerate(findings, start=1):
            finding_lines.append(
                f"{index}. {finding['type']} ({finding['severity']}) "
                f"line {finding['line']}: {finding['masked_value']}\n"
                f"   Context: {finding['context']}"
            )

        findings_text = "\n".join(finding_lines) if finding_lines else "No findings detected."
        text = (
            f"File: {result['file_name']}\n"
            f"Language: {result['language']}\n"
            f"Confidence: {result['confidence']:.1f}%\n"
            f"Word Count: {result['word_count']}\n"
            f"Preprocessing: {', '.join(result['preprocessing_steps'])}\n\n"
            f"Scan Mode: {result.get('scan_mode', 'standard')}\n"
            f"Validation Tier: {result.get('validation_tier', 'standard')}\n\n"
            f"Findings:\n{findings_text}\n\n"
            f"Extracted Text:\n\n{result['text']}"
        )
        self.ocr_text_output.setPlainText(text)

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

        existing_index = next(
            (
                index
                for index, uploaded in enumerate(self.uploaded_files)
                if uploaded.get('path') == file_info['path']
            ),
            None
        )
        if existing_index is None:
            self.uploaded_files.append(file_info)
        else:
            self.uploaded_files[existing_index] = file_info
    
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
    
    def show_all_files_report(self):
        """Display all files uploaded during this session."""
        if not self.uploaded_files:
            self.reports_display.setText("No files uploaded in this session.")
            return

        report = "ALL UPLOADED FILES\n"
        report += "=" * 80 + "\n\n"
        report += self.build_files_report(self.uploaded_files)
        report += f"\nTotal files: {len(self.uploaded_files)}"

        self.reports_display.setText(report)
        logger.info(f"Displayed {len(self.uploaded_files)} uploaded files")
    
    def show_recent_files_report(self):
        """Display files uploaded during this session."""
        if not self.uploaded_files:
            self.reports_display.setText("No files uploaded in this session.")
            return

        report = "RECENT FILES (Current Session)\n"
        report += "=" * 80 + "\n\n"
        report += self.build_files_report(self.uploaded_files)
        report += f"\nTotal session files: {len(self.uploaded_files)}"

        self.reports_display.setText(report)
        logger.info(f"Displayed {len(self.uploaded_files)} recent session files")
    
    def show_high_risk_report(self):
        """Display high-risk regex detections from uploaded files."""
        high_risk_findings = []
        for file_info in self.uploaded_files:
            for finding in file_info.get('regex_findings', []):
                if finding.get('severity') == 'high':
                    high_risk_findings.append((file_info, finding))

        if not high_risk_findings:
            self.reports_display.setText("No high-risk regex detections found in uploaded files.")
            return

        report = "HIGH-RISK REGEX DETECTIONS\n"
        report += "=" * 80 + "\n\n"

        for i, (file_info, finding) in enumerate(high_risk_findings, 1):
            report += f"{i}. File: {file_info['name']}\n"
            report += f"   Detection Type: {finding['type']}\n"
            report += f"   Risk Level: {finding['severity'].upper()}\n"
            report += f"   Data Found: {finding['masked_value']}\n"
            report += f"   Line: {finding['line']}\n"
            report += f"   Context: {finding['context']}\n"
            report += "\n"

        report += f"\nTotal high-risk detections: {len(high_risk_findings)}"
        self.reports_display.setText(report)
        logger.info(f"Displayed {len(high_risk_findings)} high-risk regex detections")
    
    def show_statistics_report(self):
        """Display current session statistics."""
        total_files = len(self.uploaded_files)
        parsed_files = sum(1 for file_info in self.uploaded_files if 'parsed_content' in file_info)
        parse_errors = sum(1 for file_info in self.uploaded_files if 'parse_error' in file_info)
        total_size = sum(file_info.get('size', 0) for file_info in self.uploaded_files)
        all_findings = [
            finding
            for file_info in self.uploaded_files
            for finding in file_info.get('regex_findings', [])
        ]

        severity_counts = {}
        type_counts = {}
        for finding in all_findings:
            severity = finding.get('severity', 'unknown')
            finding_type = finding.get('type', 'Unknown')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            type_counts[finding_type] = type_counts.get(finding_type, 0) + 1

        report = "SESSION STATISTICS\n"
        report += "=" * 80 + "\n\n"
        report += f"Total Files Uploaded: {total_files}\n"
        report += f"Files Parsed/Scanned: {parsed_files}\n"
        report += f"Files With Parse Errors: {parse_errors}\n"
        report += f"Total Storage Used: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)\n"
        report += f"Total Regex Detections: {len(all_findings)}\n"

        report += "\nSeverity Distribution:\n"
        if severity_counts:
            for severity, count in sorted(severity_counts.items()):
                percentage = (count / len(all_findings) * 100) if all_findings else 0
                report += f"  {severity.upper()}: {count} ({percentage:.1f}%)\n"
        else:
            report += "  No detections yet\n"

        report += "\nDetection Type Distribution:\n"
        if type_counts:
            for finding_type, count in sorted(type_counts.items()):
                report += f"  {finding_type}: {count}\n"
        else:
            report += "  No detections yet\n"

        self.reports_display.setText(report)
        logger.info("Displayed session statistics")

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
            report += f"   Regex Findings: {len(findings)} total, {high_findings} high risk\n"
            if 'parse_error' in file_info:
                report += f"   Error: {file_info['parse_error']}\n"
            report += "\n"
        return report
    
    def export_json_report(self):
        """Export uploaded session data to JSON file"""
        try:
            import json
            from datetime import datetime

            data = []
            for file_info in self.uploaded_files:
                data.append({
                    'name': file_info.get('name'),
                    'path': file_info.get('path'),
                    'size': file_info.get('size'),
                    'format': file_info.get('format'),
                    'created': file_info.get('created'),
                    'modified': file_info.get('modified'),
                    'status': 'parsed' if 'parsed_content' in file_info else 'parse_error' if 'parse_error' in file_info else 'pending',
                    'parse_error': file_info.get('parse_error'),
                    'metadata': file_info.get('parsed_metadata', {}),
                    'regex_summary': file_info.get('regex_summary', {}),
                    'regex_findings': file_info.get('regex_findings', []),
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
            
            total_detections = sum(record.get('regex_summary', {}).get('total', 0) for record in data)
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
