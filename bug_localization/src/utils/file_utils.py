import os
import re
from typing import Dict, List, Optional
from collections import Counter

from omegaconf import DictConfig, OmegaConf


def get_file_ext(filepath: str):
    return os.path.splitext(filepath)[-1].lower()


def get_file_exts(files: list[str]) -> dict[str, int]:
    return dict(Counter([get_file_ext(file) for file in files]))


def create_dir(dir_path: str) -> str:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    return dir_path


def create_run_directory(baseline_results_path: str) -> tuple[str, int]:
    run_index = 0
    while os.path.exists(os.path.join(baseline_results_path, f'run_{run_index}')):
        run_index += 1

    run_path = create_dir(os.path.join(baseline_results_path, f'run_{run_index}'))

    return run_path, run_index


def save_config(config: DictConfig, path: str):
    with open(os.path.join(path, 'config.yamls'), 'w') as f:
        f.write(OmegaConf.to_yaml(config))


def is_test_file(file_path: str):
    test_phrases = ["test", "tests", "testing"]
    words = set(re.split(r" |_|\/|\.", file_path.lower()))
    return any(word in words for word in test_phrases)

def get_repo_content(
    repo_path: str,
    extensions: Optional[List[str]] = None,
    ignore_tests: bool = True
) -> Dict[str, Optional[str]]:
    """
    Read every file in the working tree (excluding .git) and return
    a map from its path (relative to repo_path) to its text content.
    Assumes that the repo at repo_path is already checked out at the
    desired commit SHA.
    """
    file_contents: Dict[str, Optional[str]] = {}
    if not os.path.exists(repo_path):
        print(f"Repo path {repo_path} does not exist")
        return file_contents

    for dirpath, dirnames, filenames in os.walk(repo_path):
        # never descend into .git
        if ".git" in dirnames:
            dirnames.remove(".git")

        for fname in filenames:
            # filter by extension
            if extensions and not any(fname.endswith(ext) for ext in extensions):
                continue

            rel_path = os.path.relpath(os.path.join(dirpath, fname), repo_path)

            # skip test files if requested
            if ignore_tests and is_test_file(rel_path):
                continue

            full_path = os.path.join(dirpath, fname)
            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    file_contents[rel_path] = f.read()
            except Exception:
                file_contents[rel_path] = None

    return file_contents