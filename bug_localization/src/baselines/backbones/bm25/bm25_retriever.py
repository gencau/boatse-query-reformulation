from typing import Dict, Any
import json
import numpy as np
from tenacity import retry, stop_after_attempt, wait_random_exponential
from src.baselines.backbones.base_backbone import BaseBackbone

from pyserini.search import LuceneSearcher
from pyserini.index.lucene import IndexReader

class BM25Backbone(BaseBackbone):

    def __init__(self,
                 name: str,
                 data_path: str,
                 topk: int,
                 percent: int,
                 k1: int,
                 b : int,
                 **kwargs):
        self.name = name
        self._data_path = data_path
        self._topk = topk
        self._percent = percent
        self._k1 = k1
        self._b = b

    def _search_index(self, query: str, repo_dir: str, language: str):
        index_dir = repo_dir + "/indexes"

        topk_retrieval = self._topk
        if (topk_retrieval == 0):
            reader = IndexReader(index_dir)
            stats = reader.stats()
            print("Documents in index:", stats['documents'])
            topk_retrieval = round((self._percent/100) * stats['documents'])
            print(f"Will retrieve {topk_retrieval} documents.")

        searcher = LuceneSearcher(index_dir)
        searcher.set_bm25(k1=self._k1, b=self._b) 
        hits = searcher.search(query, k=topk_retrieval)

        print(f"Repo dir path is: {repo_dir} with language {language}")

        files = []
        scores = []
        callers = []
        callees = []

        print(f"Top {self._topk} results for '{query}':")
        for i, hit in enumerate(hits):
            doc = searcher.doc(hit.docid)
            doc = hit.lucene_document.get('raw')
            doc_dict = json.loads(doc)
            print(f"{i+1:2d}. {hit.docid:4}\n Document: {doc_dict.get('path')}\n(score: {hit.score:.4f})")
            files.append(doc_dict.get('path'))
            scores.append(hit.score)

        callers = []
        callees = []
        print(f"Final files contains: {files} with {len(files)} hits.")
        return files, scores, callers, callees

    def _iterate_on_context(self, issue : str, repo_dir: str, language: str):

        # Pass issue description, paths and file content
        predicted_files, scores, callers, callees = self._search_index(issue, repo_dir, language)

        return predicted_files, scores, callers, callees
    
    def extract_info_from_issue(self, issue_description: str, **kwargs) -> Dict[str, Any]:
        return NotImplementedError

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def localize_bugs(self, issue_description: str, repo_content: dict[str, str], 
                      repo_dir: str, language: str, **kwargs) -> Dict[str, Any]:

        retrieved_files, scores, callers, callees = self._iterate_on_context(issue_description, repo_dir, language)

        print(f"Final files contains: {retrieved_files} with {len(retrieved_files)} hits.")
        all_files = []
        all_files.extend(retrieved_files)
        if len(callers) > 0:
            all_files.extend(callers)
        if len(callees) > 0:
            all_files.extend(callees)

        return {
            "final_files": list(retrieved_files),
            "rank_scores": list(scores),
            "callers": list(callers),
            "callees": list(callees),
            "all_files": list(all_files)
        }
