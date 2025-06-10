import os
from dotenv import load_dotenv
from pathlib import Path # Add Path import

# Attempt to import Google AI types for safety settings
try:
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    print("Warning: 'google.generativeai.types'를 찾을 수 없습니다. SAFETY_SETTINGS에 대한 기본값을 문자열로 사용하거나 라이브러리를 확인하세요.")
    HarmCategory = None
    HarmBlockThreshold = None

# 경로 설정: result-sdk
RESULT_SDK_ROOT = Path(os.path.dirname(__file__)) # Use Path for easier manipulation

# Determine APP_ENV to load the correct .env file from common-sdk
# This APP_ENV is primarily for selecting the .env file.
# The Settings class will also read it for other path configurations.
APP_ENV_FOR_DOTENV = os.getenv("APP_ENV", "development").lower()

# 경로 설정: common-sdk
COMMON_SDK_CONFIG_DIR = RESULT_SDK_ROOT.parent.parent / 'common-sdk' / 'common_sdk'

dotenv_path_to_load = None
if APP_ENV_FOR_DOTENV == "development":
    dotenv_path_to_load = COMMON_SDK_CONFIG_DIR / '.env.dev'
elif APP_ENV_FOR_DOTENV == "test":
    dotenv_path_to_load = COMMON_SDK_CONFIG_DIR / '.env.test'
elif APP_ENV_FOR_DOTENV == "production":
    # For production, env vars are often set directly in the environment.
    # If a specific .env.prod file is used, its path should be specified here.
    # Attempting to load a generic .env from common-sdk as a fallback.
    dotenv_path_to_load = COMMON_SDK_CONFIG_DIR / '.env' 
else:  # Staging or unknown environment
    print(f"Warning: Unknown APP_ENV '{APP_ENV_FOR_DOTENV}' for selecting .env file. Defaulting to load '.env.dev' from common-sdk.")
    dotenv_path_to_load = COMMON_SDK_CONFIG_DIR / '.env.dev' # Default to .env.dev

if dotenv_path_to_load and dotenv_path_to_load.exists():
    load_dotenv(dotenv_path_to_load)
    print(f"INFO: Loaded environment variables from: {dotenv_path_to_load}")
else:
    # If the specific .env file (e.g. .env.dev for development) is not found,
    # try loading a generic .env from common-sdk as a general fallback.
    generic_common_sdk_dotenv_path = COMMON_SDK_CONFIG_DIR / '.env'
    if dotenv_path_to_load != generic_common_sdk_dotenv_path and generic_common_sdk_dotenv_path.exists():
        load_dotenv(generic_common_sdk_dotenv_path)
        print(f"INFO: Loaded environment variables from generic common-sdk .env: {generic_common_sdk_dotenv_path}")
    else:
        if dotenv_path_to_load:
            print(f"WARNING: Environment file not found at {dotenv_path_to_load}. Also, generic common-sdk .env not found or already checked.")
        else: # Should not happen with current logic, but as a safeguard
            print(f"WARNING: dotenv_path_to_load was not set. Generic common-sdk .env not found.")
        print("Relying on system environment variables. Ensure GOOGLE_API_KEY and other critical variables are set if not found.")

class Settings:
    def __init__(self):

        # Pinecone 설정
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.INDEX_NAME_KOR_DEN_CONTENTS = os.getenv("INDEX_NAME_KOR_DEN_CONTENTS")
        self.INDEX_NAME_ENG_DEN_CONTENTS = os.getenv("INDEX_NAME_ENG_DEN_CONTENTS")
        self.INDEX_NAME_KOR_SPA_CONTENTS = os.getenv("INDEX_NAME_KOR_SPA_CONTENTS")
        self.INDEX_NAME_ENG_SPA_CONTENTS = os.getenv("INDEX_NAME_ENG_SPA_CONTENTS")

        # OpenAI 설정
        self.OPENAI_API_KEY_J = os.getenv("OPENAI_API_KEY_J")
        self.OPENAI_API_KEY_K = os.getenv("OPENAI_API_KEY_K")
        self.OPENAI_API_KEY_B = os.getenv("OPENAI_API_KEY_B")

        # --- Settings from former kiwi/config.py ---
        self.APP_ENV = os.getenv("APP_ENV", "development").lower()
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE_PATH")
        # LLM 모델 이름을 config.py에서 직접 관리 (사용자 요청)
        self.LLM_MODEL_NAME = "gemini-2.5-flash-preview-05-20" 

        # Path configurations based on APP_ENV
        # Using RESULT_SDK_ROOT as the base for development paths.
        # For production, paths are typically absolute and set via environment variables.
        project_root_for_dev_paths = RESULT_SDK_ROOT

        if self.APP_ENV == "production":
            self.BASE_INPUT_DIR = os.getenv("PROD_INPUT_DIR", "/srv/app/ocr_inputs")
            self.BASE_OUTPUT_DIR = os.getenv("PROD_OUTPUT_DIR", "/srv/app/ocr_outputs")
            self.BASE_TEMP_DIR = os.getenv("PROD_TEMP_DIR", "/tmp/ocr_processing_temp")
            if not self.SERVICE_ACCOUNT_FILE:
                print("경고: 프로덕션 환경 SERVICE_ACCOUNT_FILE_PATH 환경 변수가 설정되지 않았습니다.")
        elif self.APP_ENV == "development":
            self.BASE_INPUT_DIR = os.getenv("DEV_INPUT_DIR", os.path.join(project_root_for_dev_paths, "sample_inputs"))
            self.BASE_OUTPUT_DIR = os.getenv("DEV_OUTPUT_DIR", os.path.join(project_root_for_dev_paths, "local_outputs"))
            self.BASE_TEMP_DIR = os.getenv("DEV_TEMP_DIR", os.path.join(project_root_for_dev_paths, "local_temp"))
            if not self.SERVICE_ACCOUNT_FILE:
                # Fallback for local development, ideally this path should also be in .env
                self.SERVICE_ACCOUNT_FILE = os.getenv("LOCAL_DEV_SERVICE_ACCOUNT_FALLBACK", "/Users/ki/Desktop/Google Drive/Dev/Ecode/ecode-458109-73d063ae5f2a.json")
        else:  # Staging or unknown environment
            print(f"경고: 알 수 없는 APP_ENV '{self.APP_ENV}'. 개발 환경 설정을 사용합니다.")
            self.BASE_INPUT_DIR = os.getenv("DEV_INPUT_DIR", os.path.join(project_root_for_dev_paths, "sample_inputs"))
            self.BASE_OUTPUT_DIR = os.getenv("DEV_OUTPUT_DIR", os.path.join(project_root_for_dev_paths, "local_outputs"))
            self.BASE_TEMP_DIR = os.getenv("DEV_TEMP_DIR", os.path.join(project_root_for_dev_paths, "local_temp"))
            if not self.SERVICE_ACCOUNT_FILE:
                self.SERVICE_ACCOUNT_FILE = os.getenv("LOCAL_DEV_SERVICE_ACCOUNT_FALLBACK", "/Users/ki/Desktop/Google Drive/Dev/Ecode/ecode-458109-73d063ae5f2a.json")

        self.GENERATION_CONFIG = {
            "temperature": 0.2,
            #"max_output_tokens": 8192  # Gemini 1.5 Flash max
        }

        if HarmCategory and HarmBlockThreshold:
            self.SAFETY_SETTINGS = [
                {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
                {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
                {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
                {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
            ]
        else:
            print("Warning: HarmCategory 또는 HarmBlockThreshold를 import 할 수 없어 SAFETY_SETTINGS를 기본값(None)으로 설정합니다.")
            self.SAFETY_SETTINGS = None
        
        self.DEFAULT_PDF_CONVERT_TEMP_SUBDIR_NAME = "pdf_pages"
        self.DEFAULT_UNDERLINE_OUTPUT_SUBDIR_NAME = "visualized_outputs"

settings = Settings()

# 애플리케이션 실행 시 필요한 기본 디렉토리 자동 생성 (선택 사항)
# Ensure paths are valid before creating
if hasattr(settings, 'BASE_OUTPUT_DIR') and settings.BASE_OUTPUT_DIR:
    os.makedirs(settings.BASE_OUTPUT_DIR, exist_ok=True)
if hasattr(settings, 'BASE_TEMP_DIR') and settings.BASE_TEMP_DIR:
    os.makedirs(settings.BASE_TEMP_DIR, exist_ok=True)

# 인덱스 목록 확인 (commented out as in original)
# from pinecone import Pinecone
# pc = Pinecone(api_key=settings.PINECONE_API_KEY)
# indexes = pc.list_indexes()
# print([idx.name for idx in indexes.indexes])
