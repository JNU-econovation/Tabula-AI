import os
from dotenv import load_dotenv
from pathlib import Path

# 1. 환경 구분
env = os.getenv("ENV", "test")

# 2. NOTE_SDK_ROOT 설정: note_sdk
NOTE_SDK_ROOT = Path(__file__).resolve().parent

# 3. 프로젝트 루트 설정
env_project_root_mapping = {
    "test": NOTE_SDK_ROOT.parent.parent.parent,
    "dev": Path("/app"),
    "prod": Path("/app")
}
PROJECT_ROOT = env_project_root_mapping.get(env, NOTE_SDK_ROOT.parent.parent.parent)

class Settings:

    # 기본 설정
    if env == "test":
        SPACE_DIR = os.path.join(NOTE_SDK_ROOT, "space")
    else:
        SPACE_DIR = os.path.join(PROJECT_ROOT, "space")

    # 파일 최대 크기 설정
    MAX_TEXT_TOKEN = 100000
    MAX_IMAGE_TOKEN = 100000

    # 작업 디렉토리 구조
    @staticmethod
    def get_space_dir(space_id: str) -> Path:
        """사용자별 작업 디렉토리 경로 반환"""
        return Path(Settings.SPACE_DIR) / space_id

    @staticmethod
    def get_markdown_path(space_id: str, document_id: str) -> Path:
        """마크다운 파일 경로 반환"""
        return Path(Settings.SPACE_DIR) / space_id / "md" / f"{document_id}.md"

    @staticmethod
    def get_image_dir(space_id: str) -> Path:
        """이미지 디렉토리 경로 반환"""
        return Path(Settings.SPACE_DIR) / space_id / "images"

    @staticmethod
    def get_keyword_path(space_id: str, document_id: str) -> Path:
        """키워드 파일 경로 반환"""
        return Path(Settings.SPACE_DIR) / space_id / "keyword" / f"{document_id}.json"
    
    @staticmethod
    def get_origin_dir(space_id: str) -> Path:
        """원본 파일 디렉토리 경로 반환"""
        return Path(Settings.SPACE_DIR) / space_id / "origin"

    @staticmethod
    def get_temp_dir(space_id: str) -> Path:
        """임시 디렉토리 경로 반환"""
        temp_dir = Path(Settings.SPACE_DIR) / space_id / "temp"
        return temp_dir

    @staticmethod
    def ensure_user_directories(space_id: str) -> None:
        """사용자별 필요한 디렉토리 생성"""
        directories = [
            Path(Settings.SPACE_DIR) / space_id / "md",
            Path(Settings.SPACE_DIR) / space_id / "images",
            Path(Settings.SPACE_DIR) / space_id / "keyword",
            Path(Settings.SPACE_DIR) / space_id / "origin",
            Path(Settings.SPACE_DIR) / space_id / "temp"
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

# Settings 인스턴스 생성
settings = Settings() 