"""
IntelliSafe - Database Management Module
Handles SQLite database operations for files, detections, and logs
"""

import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DatabaseManager:
    """SQLite database manager for IntelliSafe"""
    
    def __init__(self, db_path: str = "database/intellisafe.db"):
        """
        Initialize database manager
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = None
        self.init_database()
    
    def connect(self) -> sqlite3.Connection:
        """Get or create database connection"""
        if self.connection is None:
            self.connection = sqlite3.connect(str(self.db_path))
            self.connection.row_factory = sqlite3.Row
        return self.connection
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def init_database(self):
        """Initialize database schema"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            # Files table - stores uploaded file information
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL UNIQUE,
                    file_size INTEGER,
                    file_format TEXT,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    file_hash TEXT,
                    status TEXT DEFAULT 'success',
                    error_message TEXT,
                    parsed_content_preview TEXT
                )
            ''')
            
            # Detections table - stores detection results
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL,
                    detection_type TEXT NOT NULL,
                    pattern_matched TEXT,
                    data_found TEXT,
                    location_info TEXT,
                    risk_level TEXT,
                    detection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
                )
            ''')
            
            # Logs table - stores application logs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    log_level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    source TEXT,
                    log_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # OCR results table - stores OCR extraction results
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ocr_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL,
                    extracted_text TEXT,
                    confidence REAL,
                    preprocessing_steps TEXT,
                    word_count INTEGER,
                    ocr_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ocr_language TEXT DEFAULT 'eng',
                    validation_tier TEXT,
                    confidence_breakdown TEXT,
                    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
                )
            ''')

            self._ensure_column(cursor, "ocr_results", "validation_tier", "TEXT")
            self._ensure_column(cursor, "ocr_results", "confidence_breakdown", "TEXT")

            # Risk-assessment columns for files table
            self._ensure_column(cursor, "files", "risk_score", "REAL DEFAULT 0")
            self._ensure_column(cursor, "files", "risk_level", "TEXT DEFAULT 'low'")
            self._ensure_column(cursor, "files", "document_classification", "TEXT DEFAULT 'unclassified'")
            self._ensure_column(cursor, "files", "redacted_version_path", "TEXT")
            self._ensure_column(cursor, "files", "redaction_strategy", "TEXT DEFAULT 'blackout'")
            self._ensure_column(cursor, "files", "total_findings_count", "INTEGER DEFAULT 0")
            self._ensure_column(cursor, "files", "entity_type_counts", "TEXT DEFAULT '{}'")

            # Sensitive-entity tracking columns on detections
            self._ensure_column(cursor, "detections", "entity_type", "TEXT")
            self._ensure_column(cursor, "detections", "entity_hash", "TEXT")
            self._ensure_column(cursor, "detections", "char_start", "INTEGER")
            self._ensure_column(cursor, "detections", "char_end", "INTEGER")
            self._ensure_column(cursor, "detections", "confidence", "REAL")
            self._ensure_column(cursor, "detections", "entity_value", "TEXT")
            self._ensure_column(cursor, "files", "post_redaction_risk_score", "REAL")
            self._ensure_column(cursor, "files", "post_redaction_risk_level", "TEXT")
            self._ensure_column(cursor, "files", "post_redaction_findings_count", "INTEGER DEFAULT 0")
            self._ensure_column(cursor, "ocr_results", "word_boxes_json", "TEXT")

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Redactions log table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS redactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL,
                    redaction_type TEXT NOT NULL,
                    entity_type TEXT,
                    mode TEXT DEFAULT 'full',
                    strategy TEXT DEFAULT 'blackout',
                    output_path TEXT,
                    findings_redacted INTEGER DEFAULT 0,
                    redaction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
                )
            ''')

            # Create indexes for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_date ON files(upload_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_format ON files(file_format)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_risk_level ON files(risk_level)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_file ON detections(file_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_type ON detections(detection_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_risk ON detections(risk_level)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_level ON logs(log_level)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(log_timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ocr_file ON ocr_results(file_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ocr_confidence ON ocr_results(confidence)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_redactions_file ON redactions(file_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_entity_hash ON detections(entity_hash)')
            
            conn.commit()
            logger.info(f"Database initialized: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            conn.rollback()
            raise

    def _ensure_column(self, cursor: sqlite3.Cursor, table: str, column: str, definition: str):
        cursor.execute(f"PRAGMA table_info({table})")
        columns = {row[1] for row in cursor.fetchall()}
        if column not in columns:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
    
    # ========================================================================
    # FILES TABLE OPERATIONS
    # ========================================================================
    
    def add_file(self, filename: str, file_path: str, file_size: int, 
                 file_format: str, file_hash: str = None, 
                 status: str = 'success', error_message: str = None,
                 parsed_preview: str = None) -> int:
        """
        Add uploaded file record to database
        
        Args:
            filename: Name of the file
            file_path: Full path to the file
            file_size: File size in bytes
            file_format: File format (.pdf, .docx, etc.)
            file_hash: Optional hash of file content
            status: File status (success/error)
            error_message: Error message if any
            parsed_preview: Preview of parsed content
            
        Returns:
            File ID (row id)
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO files 
                (filename, file_path, file_size, file_format, file_hash, 
                 status, error_message, parsed_content_preview)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (filename, file_path, file_size, file_format, file_hash,
                  status, error_message, parsed_preview))
            
            conn.commit()
            file_id = cursor.lastrowid
            logger.info(f"File added to database: {filename} (ID: {file_id})")
            return file_id
        except sqlite3.IntegrityError:
            logger.debug("File already exists in database: %s — updating record.", file_path)
            cursor.execute('SELECT id FROM files WHERE file_path = ?', (file_path,))
            result = cursor.fetchone()
            if not result:
                return None
            file_id = result[0]
            cursor.execute('''
                UPDATE files
                SET filename = ?, file_size = ?, file_format = ?, file_hash = ?,
                    status = ?, error_message = ?, parsed_content_preview = ?,
                    upload_date = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (filename, file_size, file_format, file_hash, status, error_message, parsed_preview, file_id))
            conn.commit()
            return file_id
        except sqlite3.Error as e:
            logger.error(f"Error adding file to database: {e}")
            raise
    
    def get_file(self, file_id: int) -> Optional[Dict]:
        """Get file information by ID"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT * FROM files WHERE id = ?', (file_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        except sqlite3.Error as e:
            logger.error(f"Error retrieving file: {e}")
            return None
    
    def get_all_files(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get all uploaded files with pagination"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT * FROM files 
                ORDER BY upload_date DESC 
                LIMIT ? OFFSET ?
            ''', (limit, offset))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error retrieving files: {e}")
            return []
    
    def get_files_by_format(self, file_format: str) -> List[Dict]:
        """Get all files of a specific format"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT * FROM files 
                WHERE file_format = ? 
                ORDER BY upload_date DESC
            ''', (file_format,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error retrieving files by format: {e}")
            return []
    
    def get_recent_files(self, days: int = 7, limit: int = 50) -> List[Dict]:
        """Get files uploaded in last N days"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT * FROM files 
                WHERE upload_date > datetime('now', '-' || ? || ' days')
                ORDER BY upload_date DESC
                LIMIT ?
            ''', (days, limit))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error retrieving recent files: {e}")
            return []
    
    def get_file_count(self) -> int:
        """Get total number of files in database"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT COUNT(*) FROM files')
            return cursor.fetchone()[0]
        except sqlite3.Error as e:
            logger.error(f"Error getting file count: {e}")
            return 0
    
    def delete_file(self, file_id: int) -> bool:
        """Delete file and associated detections"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            # Detections will cascade delete due to foreign key
            cursor.execute('DELETE FROM files WHERE id = ?', (file_id,))
            conn.commit()
            logger.info(f"File deleted: ID {file_id}")
            return True
        except sqlite3.Error as e:
            logger.error(f"Error deleting file: {e}")
            return False
    
    # ========================================================================
    # DETECTIONS TABLE OPERATIONS
    # ========================================================================
    
    def add_detection(
        self,
        file_id: int,
        detection_type: str,
        pattern_matched: str,
        data_found: str,
        location_info: str = None,
        risk_level: str = 'medium',
        entity_type: str | None = None,
        entity_hash: str | None = None,
        char_start: int | None = None,
        char_end: int | None = None,
        confidence: float | None = None,
        entity_value: str | None = None,
    ) -> int:
        """Add a single detection result to the database."""
        conn = self.connect()
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT INTO detections
                (file_id, detection_type, pattern_matched, data_found,
                 location_info, risk_level, entity_type, entity_hash,
                 char_start, char_end, confidence, entity_value)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                file_id, detection_type, pattern_matched, data_found,
                location_info, risk_level, entity_type, entity_hash,
                char_start, char_end, confidence, entity_value,
            ))

            conn.commit()
            detection_id = cursor.lastrowid
            logger.info(f"Detection added: File {file_id}, Type {detection_type} (ID: {detection_id})")
            return detection_id
        except sqlite3.Error as e:
            logger.error(f"Error adding detection: {e}")
            raise

    def replace_detections_for_file(self, file_id: int, detections: List[Dict]) -> int:
        """Replace all detections for a file (used on re-scan)."""
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute('DELETE FROM detections WHERE file_id = ?', (file_id,))
            inserted = 0
            for row in detections:
                cursor.execute('''
                    INSERT INTO detections
                    (file_id, detection_type, pattern_matched, data_found,
                     location_info, risk_level, entity_type, entity_hash,
                     char_start, char_end, confidence, entity_value)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row.get("file_id", file_id),
                    row.get("detection_type"),
                    row.get("pattern_matched"),
                    row.get("data_found"),
                    row.get("location_info"),
                    row.get("risk_level", "medium"),
                    row.get("entity_type"),
                    row.get("entity_hash"),
                    row.get("char_start"),
                    row.get("char_end"),
                    row.get("confidence"),
                    row.get("entity_value"),
                ))
                inserted += 1
            conn.commit()
            return inserted
        except sqlite3.Error as e:
            logger.error(f"Error replacing detections for file {file_id}: {e}")
            raise

    def get_file_by_path(self, file_path: str) -> Optional[Dict]:
        """Get file record by absolute path."""
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute('SELECT * FROM files WHERE file_path = ?', (file_path,))
            row = cursor.fetchone()
            return dict(row) if row else None
        except sqlite3.Error as e:
            logger.error(f"Error retrieving file by path: {e}")
            return None

    def get_file_by_hash(self, file_hash: str, exclude_path: str | None = None) -> Optional[Dict]:
        conn = self.connect()
        cursor = conn.cursor()
        if exclude_path:
            cursor.execute(
                'SELECT * FROM files WHERE file_hash = ? AND file_path != ? ORDER BY upload_date DESC LIMIT 1',
                (file_hash, exclude_path),
            )
        else:
            cursor.execute('SELECT * FROM files WHERE file_hash = ? ORDER BY upload_date DESC LIMIT 1', (file_hash,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_entity_tracking_summary(self, limit: int = 25) -> Dict:
        """Summarize tracked sensitive entities across all files."""
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute('SELECT COUNT(*) FROM detections')
            total_entities = cursor.fetchone()[0]

            cursor.execute('''
                SELECT entity_type, COUNT(*) as count
                FROM detections
                WHERE entity_type IS NOT NULL
                GROUP BY entity_type
                ORDER BY count DESC
                LIMIT ?
            ''', (limit,))
            by_type = {row[0]: row[1] for row in cursor.fetchall()}

            cursor.execute('''
                SELECT entity_hash, entity_type, COUNT(*) as occurrences,
                       COUNT(DISTINCT file_id) as file_count
                FROM detections
                WHERE entity_hash IS NOT NULL
                GROUP BY entity_hash
                HAVING file_count > 1
                ORDER BY occurrences DESC
                LIMIT ?
            ''', (limit,))
            cross_file = [
                {
                    "entity_hash": row[0],
                    "entity_type": row[1],
                    "occurrences": row[2],
                    "file_count": row[3],
                }
                for row in cursor.fetchall()
            ]

            cursor.execute('''
                SELECT COUNT(*) FROM files
                WHERE file_hash IS NOT NULL AND file_hash != ''
            ''')
            hashed_files = cursor.fetchone()[0]

            return {
                "total_tracked_entities": total_entities,
                "entities_by_type": by_type,
                "cross_file_entities": cross_file,
                "hashed_files": hashed_files,
            }
        except sqlite3.Error as e:
            logger.error(f"Error building entity tracking summary: {e}")
            return {}
    
    def get_detections_for_file(self, file_id: int) -> List[Dict]:
        """Get all detections for a specific file"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT * FROM detections 
                WHERE file_id = ? 
                ORDER BY detection_timestamp DESC
            ''', (file_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error retrieving detections: {e}")
            return []
    
    def get_detections_by_type(self, detection_type: str, limit: int = 100) -> List[Dict]:
        """Get detections by type"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT * FROM detections 
                WHERE detection_type = ? 
                ORDER BY detection_timestamp DESC 
                LIMIT ?
            ''', (detection_type, limit))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error retrieving detections by type: {e}")
            return []
    
    def get_high_risk_detections(self, limit: int = 100) -> List[Dict]:
        """Get all high-risk detections"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT d.*, f.filename FROM detections d
                JOIN files f ON d.file_id = f.id
                WHERE d.risk_level = 'high'
                ORDER BY d.detection_timestamp DESC
                LIMIT ?
            ''', (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error retrieving high-risk detections: {e}")
            return []
    
    def get_detection_stats(self, file_id: int) -> Dict:
        """Get detection statistics for a file"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            # Total detections
            cursor.execute('SELECT COUNT(*) FROM detections WHERE file_id = ?', (file_id,))
            total = cursor.fetchone()[0]
            
            # By type
            cursor.execute('''
                SELECT detection_type, COUNT(*) as count 
                FROM detections 
                WHERE file_id = ? 
                GROUP BY detection_type
            ''', (file_id,))
            by_type = {row[0]: row[1] for row in cursor.fetchall()}
            
            # By risk level
            cursor.execute('''
                SELECT risk_level, COUNT(*) as count 
                FROM detections 
                WHERE file_id = ? 
                GROUP BY risk_level
            ''', (file_id,))
            by_risk = {row[0]: row[1] for row in cursor.fetchall()}
            
            return {
                'total': total,
                'by_type': by_type,
                'by_risk': by_risk
            }
        except sqlite3.Error as e:
            logger.error(f"Error getting detection stats: {e}")
            return {'total': 0, 'by_type': {}, 'by_risk': {}}
    
    def get_overall_stats(self) -> Dict:
        """Get overall system statistics"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            # File stats
            cursor.execute('SELECT COUNT(*) FROM files')
            total_files = cursor.fetchone()[0]
            
            cursor.execute('SELECT SUM(file_size) FROM files')
            total_size = cursor.fetchone()[0] or 0
            
            # Detection stats
            cursor.execute('SELECT COUNT(*) FROM detections')
            total_detections = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT risk_level, COUNT(*) as count 
                FROM detections 
                GROUP BY risk_level
            ''')
            risk_dist = {row[0]: row[1] for row in cursor.fetchall()}
            
            return {
                'total_files': total_files,
                'total_size_bytes': total_size,
                'total_detections': total_detections,
                'risk_distribution': risk_dist
            }
        except sqlite3.Error as e:
            logger.error(f"Error getting overall stats: {e}")
            return {'total_files': 0, 'total_size_bytes': 0, 'total_detections': 0}
    
    # ========================================================================
    # LOGS TABLE OPERATIONS
    # ========================================================================
    
    def add_log(self, log_level: str, message: str, source: str = None):
        """
        Add log entry to database
        
        Args:
            log_level: Level (info, warning, error)
            message: Log message
            source: Source module/function
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO logs (log_level, message, source)
                VALUES (?, ?, ?)
            ''', (log_level, message, source))
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error adding log: {e}")
    
    def get_logs(self, log_level: str = None, limit: int = 100) -> List[Dict]:
        """Get logs, optionally filtered by level"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            if log_level:
                cursor.execute('''
                    SELECT * FROM logs 
                    WHERE log_level = ? 
                    ORDER BY log_timestamp DESC 
                    LIMIT ?
                ''', (log_level, limit))
            else:
                cursor.execute('''
                    SELECT * FROM logs 
                    ORDER BY log_timestamp DESC 
                    LIMIT ?
                ''', (limit,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error retrieving logs: {e}")
            return []
    
    def get_recent_errors(self, hours: int = 24, limit: int = 50) -> List[Dict]:
        """Get error logs from last N hours"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT * FROM logs 
                WHERE log_level = 'error' 
                AND log_timestamp > datetime('now', '-' || ? || ' hours')
                ORDER BY log_timestamp DESC
                LIMIT ?
            ''', (hours, limit))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error retrieving error logs: {e}")
            return []
    
    def clear_old_logs(self, days: int = 30) -> int:
        """Delete logs older than N days"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                DELETE FROM logs 
                WHERE log_timestamp < datetime('now', '-' || ? || ' days')
            ''', (days,))
            conn.commit()
            count = cursor.rowcount
            logger.info(f"Cleared {count} old log entries")
            return count
        except sqlite3.Error as e:
            logger.error(f"Error clearing old logs: {e}")
            return 0
    
    # ========================================================================
    # UTILITY OPERATIONS
    # ========================================================================
    
    def export_file_history(self, file_format: str = 'json') -> str:
        """Export file history as JSON or CSV"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT f.*, 
                       COUNT(d.id) as detection_count,
                       GROUP_CONCAT(DISTINCT d.risk_level) as risk_levels
                FROM files f
                LEFT JOIN detections d ON f.id = d.file_id
                GROUP BY f.id
                ORDER BY f.upload_date DESC
            ''')
            
            rows = cursor.fetchall()
            
            if file_format == 'json':
                import json
                data = [dict(row) for row in rows]
                return json.dumps(data, indent=2, default=str)
            else:
                import csv
                from io import StringIO
                output = StringIO()
                if rows:
                    writer = csv.DictWriter(output, fieldnames=dict(rows[0]).keys())
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(dict(row))
                return output.getvalue()
        except Exception as e:
            logger.error(f"Error exporting file history: {e}")
            return ""
    
    def get_database_info(self) -> Dict:
        """Get database information"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            db_size = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM files')
            files_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM detections')
            detections_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM logs')
            logs_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM ocr_results')
            ocr_count = cursor.fetchone()[0]
            
            return {
                'database_path': str(self.db_path),
                'database_size_bytes': db_size,
                'files_count': files_count,
                'detections_count': detections_count,
                'logs_count': logs_count,
                'ocr_count': ocr_count
            }
        except sqlite3.Error as e:
            logger.error(f"Error getting database info: {e}")
            return {}
    
    # ========================================================================
    # REDACTION TABLE OPERATIONS
    # ========================================================================

    def add_redaction(
        self,
        file_id: int,
        redaction_type: str,
        entity_type: str | None = None,
        mode: str = "full",
        strategy: str = "blackout",
        output_path: str | None = None,
        findings_redacted: int = 0,
    ) -> int:
        """Log a redaction operation."""
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO redactions
                (file_id, redaction_type, entity_type, mode, strategy, output_path, findings_redacted)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (file_id, redaction_type, entity_type, mode, strategy, output_path, findings_redacted))
            conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Error logging redaction: {e}")
            raise

    def get_redactions_for_file(self, file_id: int) -> List[Dict]:
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM redactions
            WHERE file_id = ?
            ORDER BY redaction_timestamp DESC
        ''', (file_id,))
        return [dict(row) for row in cursor.fetchall()]

    def update_file_risk_metadata(
        self,
        file_id: int,
        risk_score: float,
        risk_level: str,
        classification: str = "unclassified",
        redacted_path: str | None = None,
        redaction_strategy: str | None = None,
        findings_count: int | None = None,
        entity_counts: Dict[str, int] | None = None,
    ) -> bool:
        """Update risk and redaction metadata for a file."""
        conn = self.connect()
        cursor = conn.cursor()
        try:
            sets = ["risk_score = ?", "risk_level = ?", "document_classification = ?"]
            params: list = [risk_score, risk_level, classification]
            if redacted_path:
                sets.append("redacted_version_path = ?")
                params.append(redacted_path)
            if redaction_strategy:
                sets.append("redaction_strategy = ?")
                params.append(redaction_strategy)
            if findings_count is not None:
                sets.append("total_findings_count = ?")
                params.append(findings_count)
            if entity_counts is not None:
                import json
                sets.append("entity_type_counts = ?")
                params.append(json.dumps(entity_counts))
            params.append(file_id)
            cursor.execute(f"UPDATE files SET {', '.join(sets)} WHERE id = ?", params)
            conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.error(f"Error updating file risk metadata: {e}")
            return False

    def get_risk_analytics(self) -> Dict:
        """Get risk-level distribution and top entity types across all files."""
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                SELECT risk_level, COUNT(*) as count
                FROM files
                WHERE risk_level IS NOT NULL AND risk_level != 'low'
                GROUP BY risk_level
            ''')
            risk_dist = {row[0]: row[1] for row in cursor.fetchall()}

            cursor.execute('''
                SELECT entity_type_counts FROM files
                WHERE entity_type_counts IS NOT NULL AND entity_type_counts != '{}'
            ''')
            entity_agg: Dict[str, int] = {}
            import json
            for row in cursor.fetchall():
                counts = json.loads(row[0])
                for k, v in counts.items():
                    entity_agg[k] = entity_agg.get(k, 0) + int(v)

            top_entities = sorted(entity_agg.items(), key=lambda kv: kv[1], reverse=True)[:10]

            cursor.execute('''
                SELECT COUNT(*) FROM files
                WHERE redacted_version_path IS NOT NULL AND redacted_version_path != ''
            ''')
            redacted_count = cursor.fetchone()[0]

            return {
                "risk_distribution": risk_dist,
                "top_entity_types": dict(top_entities),
                "total_redacted_documents": redacted_count,
            }
        except sqlite3.Error as e:
            logger.error(f"Error getting risk analytics: {e}")
            return {}

    def set_setting(self, key: str, value: str) -> None:
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO app_settings (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP
        ''', (key, value))
        conn.commit()

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute('SELECT value FROM app_settings WHERE key = ?', (key,))
        row = cursor.fetchone()
        return row[0] if row else default

    def get_all_settings(self) -> Dict:
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute('SELECT key, value FROM app_settings')
        result = {}
        for row in cursor.fetchall():
            raw = row[1]
            try:
                result[row[0]] = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                if raw in ("True", "False"):
                    result[row[0]] = raw == "True"
                else:
                    result[row[0]] = raw
        return result

    def update_post_redaction_metadata(
        self,
        file_id: int,
        risk_score: float,
        risk_level: str,
        findings_count: int,
    ) -> bool:
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                UPDATE files
                SET post_redaction_risk_score = ?,
                    post_redaction_risk_level = ?,
                    post_redaction_findings_count = ?
                WHERE id = ?
            ''', (risk_score, risk_level, findings_count, file_id))
            conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.error(f"Error updating post-redaction metadata: {e}")
            return False

    def get_redaction_history(self, limit: int = 100) -> List[Dict]:
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT r.*, f.filename, f.file_path
            FROM redactions r
            JOIN files f ON f.id = r.file_id
            ORDER BY r.redaction_timestamp DESC
            LIMIT ?
        ''', (limit,))
        return [dict(row) for row in cursor.fetchall()]

        # ========================================================================
    # OCR RESULTS TABLE OPERATIONS
    # ========================================================================
    
    def upsert_ocr_result(
        self,
        file_id: int,
        extracted_text: str,
        confidence: float,
        preprocessing_steps: list | None = None,
        word_count: int = 0,
        language: str = 'eng',
        validation_tier: str | None = None,
        confidence_breakdown: dict | None = None,
        word_boxes: list | None = None,
    ) -> int:
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute('DELETE FROM ocr_results WHERE file_id = ?', (file_id,))
            steps_json = json.dumps(preprocessing_steps or [])
            confidence_json = json.dumps(confidence_breakdown) if confidence_breakdown else None
            boxes_json = json.dumps(word_boxes) if word_boxes else None
            cursor.execute('''
                INSERT INTO ocr_results
                (file_id, extracted_text, confidence, preprocessing_steps, word_count,
                 ocr_language, validation_tier, confidence_breakdown, word_boxes_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                file_id, extracted_text, confidence, steps_json, word_count,
                language, validation_tier, confidence_json, boxes_json,
            ))
            conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Error upserting OCR result: {e}")
            raise

    def add_ocr_result(
        self,
        file_id: int,
        extracted_text: str,
        confidence: float,
        preprocessing_steps: list = None,
        word_count: int = 0,
        language: str = 'eng',
        validation_tier: str | None = None,
        confidence_breakdown: dict | None = None,
    ) -> int:
        """
        Add OCR extraction result to database.
        
        Args:
            file_id: ID of the file being processed
            extracted_text: Text extracted by OCR
            confidence: OCR confidence score (0-100)
            preprocessing_steps: List of preprocessing steps applied
            word_count: Number of words extracted
            language: OCR language used
            
        Returns:
            OCR result ID
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            import json
            steps_json = json.dumps(preprocessing_steps) if preprocessing_steps else json.dumps([])
            confidence_json = json.dumps(confidence_breakdown) if confidence_breakdown else None
            
            cursor.execute('''
                INSERT INTO ocr_results 
                (file_id, extracted_text, confidence, preprocessing_steps, word_count,
                 ocr_language, validation_tier, confidence_breakdown)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                file_id,
                extracted_text,
                confidence,
                steps_json,
                word_count,
                language,
                validation_tier,
                confidence_json,
            ))
            
            conn.commit()
            result_id = cursor.lastrowid
            logger.info(f"OCR result saved (ID: {result_id}, file_id: {file_id}, confidence: {confidence:.1f}%)")
            return result_id
        except sqlite3.Error as e:
            logger.error(f"Error adding OCR result: {e}")
            raise
    
    def get_ocr_result(self, file_id: int) -> Optional[Dict]:
        """
        Get OCR result for a specific file.
        
        Args:
            file_id: File ID
            
        Returns:
            OCR result dict or None
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT * FROM ocr_results 
                WHERE file_id = ?
                ORDER BY ocr_timestamp DESC
                LIMIT 1
            ''', (file_id,))
            row = cursor.fetchone()
            
            if row:
                result = dict(row)
                result['preprocessing_steps'] = json.loads(result['preprocessing_steps'])
                if result.get('confidence_breakdown'):
                    result['confidence_breakdown'] = json.loads(result['confidence_breakdown'])
                if result.get('word_boxes_json'):
                    result['word_boxes'] = json.loads(result['word_boxes_json'])
                else:
                    result['word_boxes'] = []
                return result
            return None
        except sqlite3.Error as e:
            logger.error(f"Error retrieving OCR result: {e}")
            return None
    
    def get_all_ocr_results(self, limit: int = 100) -> List[Dict]:
        """
        Get all OCR results.
        
        Args:
            limit: Maximum number of results
            
        Returns:
            List of OCR result dicts
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT ocr_results.*, files.filename 
                FROM ocr_results 
                JOIN files ON ocr_results.file_id = files.id
                ORDER BY ocr_results.ocr_timestamp DESC
                LIMIT ?
            ''', (limit,))
            rows = cursor.fetchall()
            
            import json
            results = []
            for row in rows:
                result = dict(row)
                result['preprocessing_steps'] = json.loads(result['preprocessing_steps'])
                if result.get('confidence_breakdown'):
                    result['confidence_breakdown'] = json.loads(result['confidence_breakdown'])
                results.append(result)
            return results
        except sqlite3.Error as e:
            logger.error(f"Error retrieving OCR results: {e}")
            return []
    
    def get_ocr_statistics(self) -> Dict:
        """
        Get OCR-related statistics.
        
        Returns:
            Dict with OCR stats
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            # Total OCR results
            cursor.execute('SELECT COUNT(*) FROM ocr_results')
            total_ocr = cursor.fetchone()[0]
            
            # Average confidence
            cursor.execute('SELECT AVG(confidence) FROM ocr_results')
            avg_confidence = cursor.fetchone()[0] or 0
            
            # Total text extracted
            cursor.execute('SELECT SUM(word_count) FROM ocr_results')
            total_words = cursor.fetchone()[0] or 0
            
            # Most used languages
            cursor.execute('''
                SELECT ocr_language, COUNT(*) as count 
                FROM ocr_results 
                GROUP BY ocr_language
                ORDER BY count DESC
                LIMIT 5
            ''')
            languages = {row[0]: row[1] for row in cursor.fetchall()}
            
            return {
                'total_ocr_results': total_ocr,
                'average_confidence': avg_confidence,
                'total_words_extracted': total_words,
                'languages_used': languages
            }
        except sqlite3.Error as e:
            logger.error(f"Error getting OCR statistics: {e}")
            return {}
    
    def delete_ocr_result(self, ocr_id: int) -> bool:
        """
        Delete OCR result.
        
        Args:
            ocr_id: OCR result ID
            
        Returns:
            Success status
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('DELETE FROM ocr_results WHERE id = ?', (ocr_id,))
            conn.commit()
            logger.info(f"Deleted OCR result: {ocr_id}")
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.error(f"Error deleting OCR result: {e}")
            return False

    def cleanup_repetitive_data(self) -> Dict:
        """Remove duplicate detections and redactions."""
        conn = self.connect()
        cursor = conn.cursor()
        cleaned = {}

        try:
            cursor.execute('''
                DELETE FROM detections
                WHERE id NOT IN (
                    SELECT MIN(id) FROM detections
                    WHERE data_found IS NOT NULL AND data_found != ''
                    GROUP BY file_id, data_found
                ) AND data_found IS NOT NULL AND data_found != ''
            ''')
            cleaned['detections_duplicates_removed'] = cursor.rowcount
            conn.commit()

            cursor.execute('''
                DELETE FROM redactions
                WHERE id NOT IN (
                    SELECT MIN(id) FROM redactions
                    GROUP BY file_id, redaction_type, findings_redacted
                )
            ''')
            cleaned['redactions_duplicates_removed'] = cursor.rowcount
            conn.commit()

            cursor.execute('VACUUM')
            conn.commit()

            cleaned['status'] = 'completed'
            logger.info(f"Cleanup completed: {cleaned}")
            return cleaned
        except sqlite3.Error as e:
            logger.error(f"Error cleaning repetitive data: {e}")
            conn.rollback()
            return {'error': str(e)}

    def cleanup_duplicates_by_hash(self) -> Dict:
        """Remove duplicate detections sharing the same entity_hash."""
        conn = self.connect()
        cursor = conn.cursor()
        cleaned = {}

        try:
            cursor.execute('''
                DELETE FROM detections
                WHERE id NOT IN (
                    SELECT MIN(id) FROM detections
                    WHERE entity_hash IS NOT NULL AND entity_hash != ''
                    GROUP BY entity_hash, file_id
                ) AND entity_hash IS NOT NULL AND entity_hash != ''
            ''')
            cleaned['hash_duplicates_removed'] = cursor.rowcount
            conn.commit()

            cursor.execute('VACUUM')
            conn.commit()

            cleaned['status'] = 'completed'
            logger.info(f"Hash cleanup completed: {cleaned}")
            return cleaned
        except sqlite3.Error as e:
            logger.error(f"Error cleaning hash duplicates: {e}")
            conn.rollback()
            return {'error': str(e)}



# Global database instance
_db_manager = None


def get_db() -> DatabaseManager:
    """Get global database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
