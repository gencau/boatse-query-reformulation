from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseBackbone(ABC):

    def __init__(self, name: str, **kwargs):
        self.name = name

    @abstractmethod
    def localize_bugs(self, issue_description: str, repo_content: Dict[str, str], **kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    def extract_info_from_issue(self, issue_description: str, **kwargs) -> Dict[str, Any]:
        pass
