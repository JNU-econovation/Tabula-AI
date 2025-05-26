import os
import yaml

from typing import Dict, Any
from .config import settings

class PromptLoader:
    def __init__(self):
        self._prompts: Dict[str, Any] = {}
        self._load_prompts()

    def _load_prompts(self):
        """Load all YAML files from the prompts directory."""
        prompts_dir = settings.PROMPT_BASE_PATH
        
        for filename in os.listdir(prompts_dir):
            if filename.endswith(".yaml"):
                with open(os.path.join(prompts_dir, filename), "r", encoding="utf-8") as f:
                    prompt_data = yaml.safe_load(f)
                    # 파일 이름에서 확장자를 제거하고 키로 사용
                    prompt_key = os.path.splitext(filename)[0].lower()
                    self._prompts[prompt_key] = prompt_data

    def load_prompt(self, prompt_key: str) -> Any:
        """Get a prompt by its key."""
        if prompt_key not in self._prompts:
            raise KeyError(f"Prompt key '{prompt_key}' not found")
        return self._prompts[prompt_key]

    def get_all_prompts(self) -> Dict[str, Any]:
        """Get all loaded prompts."""
        return self._prompts.copy() 