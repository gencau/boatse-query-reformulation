import csv
import os, sys
import shutil
import time
import sys

import hydra
from datasets import config
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from backbones.base_backbone import BaseBackbone
from configs.baseline_configs import BaselineConfig
from data_sources.base_data_source import BaseDataSource


@hydra.main(version_base="1.1", config_path="../../configs/baselines", config_name="qwen-extract")
def main(cfg: BaselineConfig) -> None:
    os.environ['HYDRA_FULL_ERROR'] = '1'
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Insert the project root at the beginning of sys.path so it has priority
    sys.path.insert(0, project_root)

    backbone: BaseBackbone = hydra.utils.instantiate(cfg.backbone)
    data_src: BaseDataSource = hydra.utils.instantiate(cfg.data_source)

    output_path = HydraConfig.get().run.dir
    os.makedirs(output_path, exist_ok=True)
    results_csv_path = os.path.join(output_path, "results.csv")

    # changed files is the ground truth
    for dp, repo_content, changed_files in data_src:
        issue_description = dp['issue_title'] + '\n' + dp['issue_body']
        start_time = time.time()
        results_dict = backbone.extract_info_from_issue(issue_description)

        end_time = time.time()
        dp.update(results_dict)
        dp['time_s'] = (end_time - start_time) * 1000000

        with open(results_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(dp.keys())
            writer.writerow(dp.values())

# Call like this: python extract_lca.py
if __name__ == '__main__':
    cache_dir = config.HF_DATASETS_CACHE
    shutil.rmtree(cache_dir, ignore_errors=True)
    load_dotenv()
    main()
