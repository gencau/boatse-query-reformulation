import os
from pathlib import Path
from typing import List, Optional

from datasets import get_dataset_config_names, load_dataset
from src.utils.file_utils import get_repo_content
from src.utils.git_utils import parse_changed_files_from_diff
from src.baselines.data_sources.base_data_source import BaseDataSource

class SWEDataSource(BaseDataSource):

    def __init__(
            self,
            hub_name: str,
            repos_dir: str,
            configs: Optional[List[str]] = None,
            split: Optional[str] = None,
            cache_dir: Optional[str] = None
    ):
        self._hub_name = hub_name
        self._cache_dir = cache_dir
        self._repos_dir = repos_dir

        if configs:
            self._configs = configs
        else:
            self._configs = get_dataset_config_names(self._hub_name)
        self._split = split

    def __iter__(self):
        if Path(self._hub_name).exists():
            dataset = load_dataset(self._hub_name, split="test")
        else:
            print(f"Couldn't find dataset {self._hub_name}")
        for dp in dataset:
            owner, name = dp["repo"].split("/", 1)

            # Repo layout convention: <repo_root>/<owner>__<name>__commithash
            repo_name = owner + "__" + name + "__" + dp["base_commit"]
            repo_path = os.path.join(self._repos_dir, repo_name)
            try:
                # Get repo content on commit  
                repo_content = get_repo_content(
                    repo_path,
                    extensions=['.py'], 
                    ignore_tests=True
                ) 
                # Get changed files between commits
                changed_files = parse_changed_files_from_diff(dp['patch'])

                yield dp, repo_content, changed_files
            except Exception as e:
                print(f"Failed to get repo content for {repo_name}", e)
                continue
