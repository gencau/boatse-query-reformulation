# Generate BM25 results for each data point of the SWE-Bench dataset
# Include dependencies
# Compare the baseline top-k with a K equal to the one that includes dependencies.
# Goal: check if including dependencies is beneficial.
import csv
import os, sys
import shutil
import time
import sys
import subprocess
import ast
from pathlib import Path

import hydra
from datasets import config
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from backbones.base_backbone import BaseBackbone
from configs.baseline_configs import BaselineConfig
from data_sources.base_data_source import BaseDataSource
from utils.bm25_utils import build_json_files


@hydra.main(version_base="1.1", config_path="../../configs/baselines", config_name="bm25_swe")
def main(cfg: BaselineConfig) -> None:
    os.environ['HYDRA_FULL_ERROR'] = '1'
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Insert the project root at the beginning of sys.path so it has priority
    sys.path.insert(0, project_root)

    backbone: BaseBackbone = hydra.utils.instantiate(cfg.backbone)
    data_src: BaseDataSource = hydra.utils.instantiate(cfg.data_source)
    recreate_index = cfg.reindex

    output_path = HydraConfig.get().run.dir
    os.makedirs(output_path, exist_ok=True)
    results_csv_path = os.path.join(output_path, "results.csv")

    # changed files is the ground truth
    for dp, repo_content, changed_files in data_src:
        owner, name = dp["repo"].split("/", 1)

        # Repo layout convention: <repo_root>/<owner>__<name>__commithash
        repo_name = owner + "__" + name + "__" + dp["base_commit"]
        repo_dir = os.path.join(cfg.data_source.repos_dir, repo_name)

        if recreate_index:
            # We need to first create the json representations of our corpus
            build_json_files(Path(repo_dir), repo_dir + "/json")

            start_time = time.time()
            # Indexing command
            command = [
                sys.executable,  # Uses the same Python interpreter running this script
                "-m",
                "pyserini.index.lucene",
                "--collection", "JsonCollection",
                "--input", repo_dir + "/json",
                "--index", repo_dir + "/indexes",
                "--generator", "DefaultLuceneDocumentGenerator",
                "--threads", "1",
                "--storePositions",
                "--storeDocvectors",
                "--storeRaw"
            ]

            # then we create the index
            try:
                subprocess.run(command, check=True)
                print("Indexing completed successfully!")
            except subprocess.CalledProcessError as e:
                print(f"Error during indexing: {e}")
            end_time = time.time()

            print(f"Created index for {dp['base_commit']} in {end_time - start_time} seconds")
            dp['indexing_time'] = (end_time - start_time)

        issue_description = dp['problem_statement']
        start_time = time.time()

        language = '.py'
        results_dict = backbone.localize_bugs(issue_description, repo_content, repo_dir, language)
        end_time = time.time()
        dp.update(results_dict)
        dp['time_s'] = (end_time - start_time) * 1000000
        dp['changed_files'] = changed_files
        dp['changed_files_count'] = len(changed_files)

        with open(results_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(dp.keys())
            writer.writerow(dp.values())

# Make sure to load the python environment and the conda pyserini environment before running!
# Call like this: python run_bm25.py
if __name__ == '__main__':
    cache_dir = config.HF_DATASETS_CACHE
    shutil.rmtree(cache_dir, ignore_errors=True)
    load_dotenv()
    main()
