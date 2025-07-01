import os

# Face recognition settings
FACE_RECOGNITION_THRESHOLD = 0.6  # Cosine similarity threshold
MTCNN_CONFIDENCE_THRESHOLD = 0.9  # Face detection confidence
FACE_EMBEDDING_SIZE = 512

# File paths
STUDENTS_DB_PATH = "data/students.csv"
ATTENDANCE_DB_PATH = "data/attendance.csv"

# UI Settings
APP_TITLE = "Face-Based Attendance System"
SIDEBAR_TITLE = "Navigation"

# Validation rules
MIN_NAME_LENGTH = 2
MATRIC_NUMBER_PATTERN = r"^[A-Z0-9]{6,15}$"  # Adjust based on your institution

# Time settings
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DATE_FORMAT = "%Y-%m-%d"
