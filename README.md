# Reformulate, Retrieve, Localize: Agents for Repository-Level Bug Localization

Running the experiments requires installing Pyserini. Follow detailed installation instructions at: https://github.com/castorini/pyserini/blob/master/docs/installation.md
All experiments were tested with Python 3.11 on Mac OS and Ollama 0.11.10.

Experiments are run on Long Code Arena (LCA) and SWE-bench Lite (SWE). The datasets are extracted to .csv files and included in the datasets/ folder, which can be used by the agent. However, the BM25 experiments require a local copy of the datasets so that the files can be indexed within the repositories. Both can be found on HuggingFace:

- LCA: JetBrains-Research/long-code-arena
- SWE-bench Lite: princeton-nlp/SWE-bench_Lite

Configuration can be found under bug_localization/configs/baselines.
Running scripts can be found under bug_localization/src/baselines.

## Repository content
-- configs/  -- all configuration files required to run BM25 variations and extracting information using a LLM  
-- datasets/ -- LCA and SWE datasets exported in .csv format  
-- output/   -- extracted information for each dataset  
-- results/  -- experiment results by dataset/model   
&nbsp;&nbsp;&nbsp;&nbsp;-- lca/   -- results on LCA  
&nbsp;&nbsp;&nbsp;&nbsp;-- swe/   -- results on SWE  
&nbsp;&nbsp;&nbsp;&nbsp;-- matching_logs/ -- logs of agent runs  
&nbsp;&nbsp;&nbsp;&nbsp;-- statistics/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- sigtests_outputs -- statistical analysis results  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- tool_usage_and_error_stats.pdf -- tool usage and error statistics  
&nbsp;&nbsp;&nbsp;&nbsp;-- scripts/  -- statistical analysis, running time analysis
&nbsp;&nbsp;&nbsp;&nbsp;-- src/      -- all source code for BM25 experiments, information extraction and agent  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- baselines/  -- contains the code for different types of baselines. BM25 and agents are supported  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- data/       -- scripts for retrieving data from hugging face  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- modelfiles/ -- use those to create custom models with Ollama  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- utils/      -- different utilities (file, git, hf, tokenization)  

## Running BM25 Experiments
### Baseline
BM25 requires setting a configuration file. There is one configuration file per dataset.
See the bug_localization/config/baselines directory:
- bm25.yaml: run the baseline experiment on LCA
- bm25_swe.yaml: run the baseline experiment on SWE-bench Lite

For the first run, make sure to enable indexing for both datasets in each configuration file by setting reindex to True.

The experiments are run by executing the python scripts:
- run_bm25.py and run_bm25_swe.py

### Extracting relevant information with LLMs (query reformulation)
Similar to the previous experiment, setup the configuration files first:
- qwen_extract.yaml (LCA)
- swe_extract.yaml (SWE)

Then run:
- extract_lca.py (LCA)
- extract_swe.py (SWE)

Finally, re-run BM25, but this time with the extracted information. Set the path to the file that contains the extracted information in the configuration files for each dataset:
- bm25_extracted.yaml (LCA)
- bm25_extracted_swe.yaml (SWE)

Then run:
- run_bm25_extracted.py (LCA)
- run_bm25_extracted_swe.py (SWE)

## Running the LLM-based agent
The agent is essentially run with the script agent_workflow_defined_set.py with command-line options. This is an example for running the agent with LCA and the Qwen3 30B model:

python agent_workflow_defined_set.py --dataset=lca --dataset_path=datasets/lca_dataset.csv --model_name=qwen3-coder-16k

While running, it creates 2 directories:
- logs, which contains detailed logs of each run, organized by dataset
- results, which contains the results in .csv format, organized by dataset

## Evaluation
There is a single evaluation script for both BM25 and agent approaches located under src/baselines/metrics.

Example usage:
python compute_metrics.py --input=results/your_results_file.csv --topk=5

The evaluation results (precision, recall, MAP, Hit@K, F1, MRR) are added as new columns to the results file.
