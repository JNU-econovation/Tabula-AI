import os
import yaml

from typing import Dict, Any
from .config import settings

from common_sdk.exceptions import FileNotFoundError
from common_sdk.get_logger import get_logger

# 로거 설정
logger = get_logger()

class PromptLoader:
    def __init__(self):
        self.prompts: Dict[str, Any] = {}
        self._load_prompts()

    def _load_prompts(self):
        prompts_dir = settings.PROMPT_BASE_PATH
        
        for filename in os.listdir(prompts_dir):
            if filename.endswith(".yaml"):
                with open(os.path.join(prompts_dir, filename), "r", encoding="utf-8") as f:
                    prompt_data = yaml.safe_load(f)
                    # 파일 이름에서 확장자를 제거하고 키로 사용
                    prompt_key = os.path.splitext(filename)[0].lower()
                    self.prompts[prompt_key] = prompt_data

    def load_prompt(self, prompt_key: str) -> Any:
        if prompt_key not in self.prompts:
            logger.error(f"[load_prompt] Prompt key '{prompt_key}' not found")
            raise FileNotFoundError(prompt_key)
        
        return self.prompts[prompt_key]