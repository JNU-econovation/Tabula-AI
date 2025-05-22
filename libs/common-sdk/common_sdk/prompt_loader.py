import os
import yaml
from typing import Dict, Any

class PromptLoader:
    def __init__(self):
        self._prompts: Dict[str, Any] = {}
        self._load_prompts()

    def _load_prompts(self):
        """Load all YAML files from the prompts directory."""
        prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
        
        for filename in os.listdir(prompts_dir):
            if filename.endswith(".yaml"):
                with open(os.path.join(prompts_dir, filename), "r", encoding="utf-8") as f:
                    prompts = yaml.safe_load(f)
                    self._prompts.update(prompts)

    def get_prompt(self, prompt_key: str) -> str:
        """Get a prompt by its key."""
        if prompt_key not in self._prompts:
            raise KeyError(f"Prompt key '{prompt_key}' not found")
        return self._prompts[prompt_key]

    def get_all_prompts(self) -> Dict[str, Any]:
        """Get all loaded prompts."""
        return self._prompts.copy() 