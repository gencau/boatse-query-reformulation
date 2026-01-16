import json
import re
from typing import Dict, Any, Optional, List, Union
from tenacity import wait_random_exponential, retry, stop_after_attempt

from langchain_ollama import ChatOllama
from src.baselines.backbones.agent.prompts.agent_context_prompt import AgentContextPrompt
from src.baselines.backbones.base_backbone import BaseBackbone
from src.utils.tokenization_utils import TokenizationUtils as tk
from src.baselines.backbones.agent.utils.json_utils import parse_json_safe


class InfoExtractionBackbone(BaseBackbone):
    def __init__(self, name: str, model_name: str, prompt: AgentContextPrompt, experiment: Optional[str] = None):
        super().__init__(name)
        self._model_name = model_name
        self._prompt = prompt
        self._experiment = experiment
        self._tokenizer = tk(self._model_name)
        self._max_tokens = self._tokenizer._context_size


    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def chat_completion_request(self, messages, tools=None):
        try:
            model = ChatOllama(model=self._model_name,
                                temperature=0.0,
                                num_predict=2048,
                                seed=42) 
            
            response = model.invoke(messages)
            return response
        except Exception as e:
            print("Unable to generate chat completion response")
            print(f"Exception: {e}")
            return e
    
    def _extract_info_from_issue(self, issue_description: str):
        sys = self._prompt.get_system_prompt()
        user = self._prompt.get_base_extract_prompt(issue_description=issue_description)
        messages = [
            {"role": "system", "content": sys},
            {"role": "user", "content": user}
        ]
        print(f"Messages in prompt: {messages}")
        txt = self.chat_completion_request(messages)
        print(f"Response from model: {txt}")
        try:
            summarized_info = parse_json_safe(txt.content)
        except Exception:
            print("Failed to parse JSON from model response")
            summarized_info = {[]}

        return summarized_info

    def localize_bugs(self, issue_description, repo_content, **kwargs):
        pass
    def extract_info_from_issue(self, issue_description: str,  **kwargs) -> Dict[str, Any]:
        summarized_info = self._extract_info_from_issue(issue_description)
        print(f"Extracted summarized info: {summarized_info}")

        return {
            "summarized_info": summarized_info
        }
