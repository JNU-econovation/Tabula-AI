"""
result-sdk 설정 관리 모듈
환경 변수, API 키, 경로 등 애플리케이션 전반의 설정을 관리함
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Google AI 유형 임포트 시도 (안전 설정용)
try:
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    print("Warning: 'google.generativeai.types' could not be found. Using default values for SAFETY_SETTINGS.")
    HarmCategory = None
    HarmBlockThreshold = None

# --- 경로 설정 ---
RESULT_SDK_ROOT = Path(__file__).parent.resolve()
COMMON_SDK_CONFIG_DIR = RESULT_SDK_ROOT.parent.parent / 'common-sdk' / 'common_sdk'

# --- .env 파일 로드 ---
# APP_ENV 환경 변수를 기반으로 적절한 .env 파일 로드
APP_ENV = os.getenv("APP_ENV", "development").lower()

dotenv_path = None
if APP_ENV == "development":
    dotenv_path = COMMON_SDK_CONFIG_DIR / '.env.dev'
elif APP_ENV == "test":
    dotenv_path = COMMON_SDK_CONFIG_DIR / '.env.test'
elif APP_ENV == "production":
    # 프로덕션 환경에서는 보통 환경 변수를 직접 설정
    dotenv_path = COMMON_SDK_CONFIG_DIR / '.env' 
else:
    print(f"Warning: Unknown APP_ENV '{APP_ENV}'. Defaulting to '.env.dev'.")
    dotenv_path = COMMON_SDK_CONFIG_DIR / '.env.dev'

if dotenv_path and dotenv_path.exists():
    load_dotenv(dotenv_path)
    print(f"INFO: Loaded environment variables from: {dotenv_path}")
else:
    print(f"WARNING: Dotenv file not found at {dotenv_path}. Relying on system environment variables.")

class Settings:
    """애플리케이션 설정을 담는 클래스"""
    def __init__(self):
        # --- API 키 및 외부 서비스 설정 ---
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE_PATH")
        
        # LLM 모델 이름을 config.py에서 직접 관리 (사용자 요청)
        self.LLM_MODEL_NAME = "gemini-2.5-flash-lite-preview-06-17" 

        # --- 경로 설정 ---
        # APP_ENV에 따라 다른 기본 경로 설정
        if APP_ENV == "production":
            self.BASE_INPUT_DIR = os.getenv("PROD_INPUT_DIR", "/srv/app/inputs")
            self.BASE_OUTPUT_DIR = os.getenv("PROD_OUTPUT_DIR", "/srv/app/outputs")
            self.BASE_TEMP_DIR = os.getenv("PROD_TEMP_DIR", "/tmp/processing_temp")
        else: # development, test 등 다른 환경
            self.BASE_INPUT_DIR = os.getenv("DEV_INPUT_DIR", str(RESULT_SDK_ROOT / "sample_inputs"))
            self.BASE_OUTPUT_DIR = os.getenv("DEV_OUTPUT_DIR", str(RESULT_SDK_ROOT / "local_outputs"))
            self.BASE_TEMP_DIR = os.getenv("DEV_TEMP_DIR", str(RESULT_SDK_ROOT / "local_temp"))
            # 개발 환경용 서비스 계정 파일 경로 (필요 시 .env 파일에 정의)
            if not self.SERVICE_ACCOUNT_FILE:
                self.SERVICE_ACCOUNT_FILE = os.getenv("LOCAL_DEV_SERVICE_ACCOUNT_FALLBACK")

        # --- LLM 생성 설정 ---
        self.GENERATION_CONFIG = {
            "temperature": 0.2,
            # "max_output_tokens": 8192, # 필요 시 설정
        }

        # --- LLM 안전 설정 ---
        if HarmCategory and HarmBlockThreshold:
            self.SAFETY_SETTINGS = [
                {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
                {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
                {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
                {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
            ]
        else:
            self.SAFETY_SETTINGS = None
        
        # --- 기타 기본값 ---
        self.DEFAULT_PDF_CONVERT_TEMP_SUBDIR_NAME = "pdf_pages"
        self.DEFAULT_UNDERLINE_OUTPUT_SUBDIR_NAME = "visualized_outputs"

# 설정 객체 인스턴스화
settings = Settings()

# --- 애플리케이션 시작 시 디렉토리 생성 ---
# 설정된 경로가 유효한 경우, 기본 출력 및 임시 디렉토리를 미리 생성
if settings.BASE_OUTPUT_DIR:
    os.makedirs(settings.BASE_OUTPUT_DIR, exist_ok=True)
if settings.BASE_TEMP_DIR:
    os.makedirs(settings.BASE_TEMP_DIR, exist_ok=True)
