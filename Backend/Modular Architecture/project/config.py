import os
from dotenv import load_dotenv

load_dotenv()

# General Flask App settings
FLASK_ENV = os.getenv('FLASK_ENV', 'development')
PORT = int(os.getenv('PORT', 8000))
DEBUG_MODE = FLASK_ENV != 'production'

# AWS Configuration
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

# Roboflow Configuration
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
ROBOFLOW_PROJECT_ID = os.getenv('ROBOFLOW_PROJECT_ID')
ROBOFLOW_VERSION_NUMBER = os.getenv('ROBOFLOW_VERSION_NUMBER')
ROBOFLOW_SDK_AVAILABLE = False # Default, will be updated in main.py

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Poppler Path
POPPLER_PATH = os.getenv('POPPLER_PATH_OVERRIDE', r"C:\Users\khura\OneDrive\Documents\Agentic AI\1. Projects\3. AI Page Comparison\Ref_Files\poppler-24.08.0\Library\bin") # Adjust as necessary or ensure it's in PATH

# Temp Uploads Directory
TEMP_UPLOADS_DIR = "temp_uploads"

# Logging Format
LOG_FORMAT = '%(asctime)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s'
LOG_LEVEL = 'DEBUG' if DEBUG_MODE else 'INFO'

# Ensure critical variables are present (optional, add more checks as needed)
if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME]):
    print("Warning: AWS credentials or S3_BUCKET_NAME are not fully configured.")

if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not found in environment variables.")

if not all([ROBOFLOW_API_KEY, ROBOFLOW_PROJECT_ID, ROBOFLOW_VERSION_NUMBER]):
    print("Warning: Roboflow API Key, Project ID, or Version Number not fully configured.")
