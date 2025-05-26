# 파일명: config.py

import os
from dotenv import load_dotenv
# google.generativeai.types는 genai 객체를 통해 접근하거나,
# 직접 해당 타입이 필요한 곳에서 import 하는 것이 좋습니다.
# 여기서는 SAFETY_SETTINGS 정의를 위해 직접 import 합니다.
try:
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    print("Warning: 'google.generativeai.types'를 찾을 수 없습니다. SAFETY_SETTINGS에 대한 기본값을 문자열로 사용하거나 라이브러리를 확인하세요.")
    # fallback 또는 에러 처리 (여기서는 일단 None으로 설정 후 아래에서 처리)
    HarmCategory = None
    HarmBlockThreshold = None


# .env 파일에서 환경 변수 로드 (예: GOOGLE_API_KEY, SERVICE_ACCOUNT_FILE_PATH 등)
load_dotenv()

# --- 기본 환경 설정 ---
# APP_ENV 환경 변수를 통해 현재 실행 환경을 구분 (기본값: "development")
# VM에 배포 시 해당 VM의 환경 변수에 APP_ENV=production 등으로 설정
APP_ENV = os.getenv("APP_ENV", "development").lower()

# --- API 키 및 인증 파일 경로 ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# 서비스 계정 파일 경로는 환경 변수에서 절대 경로로 받는 것이 좋음
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE_PATH") # 환경변수 이름은 일관성 있게 사용

# --- LLM 모델 설정 ---
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash-latest")

# --- 경로 설정 ---
# 이 파일(config.py)의 위치를 기준으로 프로젝트 루트를 추정
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if APP_ENV == "production":
    # 프로덕션(VM) 환경 경로 예시
    # 이 경로들은 VM에 실제로 존재해야 합니다.
    BASE_INPUT_DIR = os.getenv("PROD_INPUT_DIR", "/srv/app/ocr_inputs")
    BASE_OUTPUT_DIR = os.getenv("PROD_OUTPUT_DIR", "/srv/app/ocr_outputs")
    BASE_TEMP_DIR = os.getenv("PROD_TEMP_DIR", "/tmp/ocr_processing_temp") # VM의 임시 디렉토리
    if not SERVICE_ACCOUNT_FILE:
        print("경고: 프로덕션 환경 SERVICE_ACCOUNT_FILE_PATH 환경 변수가 설정되지 않았습니다.")
        # 필요시 여기서 에러를 발생시키거나 기본 동작을 정의할 수 있습니다.
elif APP_ENV == "development":
    # 로컬 개발 환경 경로 예시 (config.py 파일 위치 기준)
    BASE_INPUT_DIR = os.path.join(PROJECT_ROOT, "sample_inputs")
    BASE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "local_outputs")
    BASE_TEMP_DIR = os.path.join(PROJECT_ROOT, "local_temp")
    # 로컬 개발 시 .env에 SERVICE_ACCOUNT_FILE_PATH가 없으면 하드코딩된 경로 사용 (테스트용)
    if not SERVICE_ACCOUNT_FILE:
        SERVICE_ACCOUNT_FILE = "/Users/ki/Desktop/Google Drive/Dev/Ecode/ecode-458109-73d063ae5f2a.json" # 사용자 로컬 경로 예시
else:
    # 기타 환경 (예: staging) - 필요에 따라 추가
    print(f"경고: 알 수 없는 APP_ENV '{APP_ENV}'. 개발 환경 설정을 사용합니다.")
    BASE_INPUT_DIR = os.path.join(PROJECT_ROOT, "sample_inputs")
    BASE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "local_outputs")
    BASE_TEMP_DIR = os.path.join(PROJECT_ROOT, "local_temp")
    if not SERVICE_ACCOUNT_FILE:
        SERVICE_ACCOUNT_FILE = "/Users/ki/Desktop/Google Drive/Dev/Ecode/ecode-458109-73d063ae5f2a.json"


# --- LLM 생성 및 안전 설정 ---
GENERATION_CONFIG = {
    "temperature": 0.2,
    "max_output_tokens": 8192 # Gemini 1.5 Flash의 경우 최대 8192
}

# 안전 설정 (HarmCategory, HarmBlockThreshold가 정상적으로 import 되었는지 확인)
if HarmCategory and HarmBlockThreshold:
    SAFETY_SETTINGS = [
        {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
        {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
        {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
        {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    ]
else:
    print("Warning: HarmCategory 또는 HarmBlockThreshold를 import 할 수 없어 SAFETY_SETTINGS를 기본값(None)으로 설정합니다.")
    SAFETY_SETTINGS = None # 또는 적절한 기본값

# --- 기타 일반 설정 ---
DEFAULT_PDF_CONVERT_TEMP_SUBDIR_NAME = "pdf_pages" # PDF 변환 시 생성될 하위 폴더명
DEFAULT_UNDERLINE_OUTPUT_SUBDIR_NAME = "visualized_outputs" # 밑줄 친 이미지 저장 하위 폴더명

# 애플리케이션 실행 시 필요한 기본 디렉토리 자동 생성 (선택 사항)
# os.makedirs(BASE_INPUT_DIR, exist_ok=True) # 입력은 보통 존재해야 함
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(BASE_TEMP_DIR, exist_ok=True)

# 설정값 로드 확인용 (테스트 시)
# if __name__ == '__main__':
#     print(f"APP_ENV: {APP_ENV}")
#     print(f"GOOGLE_API_KEY: {'Set' if GOOGLE_API_KEY else 'Not Set'}")
#     print(f"SERVICE_ACCOUNT_FILE: {SERVICE_ACCOUNT_FILE}")
#     print(f"LLM_MODEL_NAME: {LLM_MODEL_NAME}")
#     print(f"BASE_TEMP_DIR: {BASE_TEMP_DIR}")
#     print(f"SAFETY_SETTINGS: {SAFETY_SETTINGS}")