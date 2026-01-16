import json
import re
from typing import Dict, Any, Optional
from tenacity import wait_random_exponential, retry, stop_after_attempt

from langchain_ollama import ChatOllama
from src.baselines.backbones.agent.prompts.agent_context_prompt import AgentContextPrompt
from src.baselines.backbones.base_backbone import BaseBackbone
from src.utils.tokenization_utils import TokenizationUtils as tk


class OllamaAgentBackbone(BaseBackbone):
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
                                top_k=20,
                                top_p=0.5) 
            
            response = model.invoke(messages)
            return response
        except Exception as e:
            print("Unable to generate chat completion response")
            print(f"Exception: {e}")
            return e

    def _extract_json(self, result) -> list:
        # Try to extract JSON from a markdown code block
        match = re.search(r"```json\s*(\{.*\})\s*```", result, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Try to extract JSON after a </think> marker
            json_matches = re.findall(r'</think>\s*(\{.*\})', result, re.DOTALL)
            if json_matches:
                json_str = json_matches[-1]
            else:
                # Fallback: extract the last JSON object from the text
                json_matches = re.findall(r"(\{.*\})", result, re.DOTALL)
                json_str = json_matches[-1] if json_matches else ""
        try:
            data = json.loads(json_str)
            return data.get("files", [])
        except json.JSONDecodeError:
            print("#### No JSON content found.")
        return []

    def _rerank_results(self, issue: str, file_list: list):
        valid_files = [f for f in file_list if isinstance(f, str)]
        if len(valid_files) <= 1:
            return valid_files, []

        system_prompt = self._prompt.get_rerank_system_prompt()
        all_results = []
        all_metadata = []

        chunk = []
        for f in valid_files:
            # Attempt to add f to current chunk
            prospective = chunk + [f]
            user_prompt = self._prompt.get_rerank_prompt(issue, "\n".join(prospective))

            # Count tokens for system + user
            n_sys   = self._tokenizer.count_text_tokens(system_prompt)
            n_user  = self._tokenizer.count_text_tokens(user_prompt)
            if n_sys + n_user > self._max_tokens and chunk:
                # flush existing chunk
                full = self._prompt.chat(system_prompt, self._prompt.get_rerank_prompt(issue, "\n".join(chunk)))
                resp = self.chat_completion_request(full)
                all_results.extend(self._extract_json(resp.content))
                all_metadata.append(resp.response_metadata)
                chunk = [f]  # start new chunk
            else:
                chunk = prospective

        # flush the last chunk
        if chunk:
            full = self._prompt.chat(system_prompt, self._prompt.get_rerank_prompt(issue, "\n".join(chunk)))
            resp = self.chat_completion_request(full)
            all_results.extend(self._extract_json(resp.content))
            all_metadata.append(resp.response_metadata)

        # dedupe in case of overlap
        seen = set()
        deduped = []
        for fn in all_results:
            try:
                if fn in seen:          # will raise TypeError on unhashable items
                    continue
                deduped.append(fn)
                seen.add(fn)
            except TypeError:
                # fn is a dict, list, etc. – ignore it
                continue

        return deduped, all_metadata

    def _iterate_on_context(self, issue : str, repo_content: Dict[str, str]):
        """
            Calculate # tokens in base prompt (without project content) to get remaining tokens for project content
            Split the project content by this amount of tokens until all are split
            Call the LLM, saving each returned file list in memory
        """
        system_prompt = self._prompt.get_system_prompt()
        base_prompt = self._prompt.base_prompt(issue)
        
        # Now count the tokens in the static parts of the prompt + issue description
        # This uses the model's specific tokenizer
        tokenizer = tk(self._model_name)
        max_tokens = tokenizer._context_size
        tk_count = tokenizer.count_text_tokens(base_prompt) + tokenizer.count_text_tokens(system_prompt)
        
        print(f"Static prompt + issue description has {tk_count} tokens")

        # Define the files tags and calculate token count.
        # Issue tags are already counted with the base prompt
        start_tag = "<FILES>"
        end_tag = "</FILES>"
        start_tag_token_count = tokenizer.count_text_tokens(start_tag)
        end_tag_token_count = tokenizer.count_text_tokens(end_tag)
        tokens_per_chunk = max_tokens - tk_count - start_tag_token_count - end_tag_token_count
        
        print(f"{tokens_per_chunk} tokens remaining for the project context")

        chunks = []
        current_chunk = ""

        # this is to make sure that we don't cut in the middle of a path
        for file_path in repo_content.keys():
            candidate = file_path if not current_chunk else current_chunk + '\n' + file_path
            if tokenizer.count_text_tokens(candidate) <= tokens_per_chunk:
                current_chunk  = candidate
            else:
                # if over the max token count, create a chunk with previous content
                # and start a new one
                chunks.append(current_chunk)
                current_chunk = file_path
        
        if current_chunk:
            chunks.append(current_chunk)

        results_list = []
        files_list = []
        response_metadata = []
        for chunk_text in chunks:
            # Call LLM with this chunk
            full_prompt = self._prompt.chat(self._prompt.get_system_prompt(), self._prompt.full_prompt(issue, chunk_text))
            print(f"Full prompt {full_prompt}")

            # Ensure the chunk ends with the required closing tag.
            for message in full_prompt:
                if (message["role"] == "user"):
                    if not message['content'].rstrip().endswith(end_tag):
                        message['content'] = message['content'].rstrip() + " " + end_tag

            result = self.chat_completion_request(full_prompt)
            print(f"Chat results: {result}")
            results_list.append(result.content)
            response_metadata.append(result.response_metadata)

        for result in results_list:
            data = self._extract_json(result)
            if len(data) > 0:
                files_list.extend(data)

        print(f"Final files list: {files_list}")
        reranked_results, rerank_metadata = self._rerank_results(issue, files_list)

        print(f"RERANKED Final files list: {reranked_results}")
        return files_list, reranked_results, results_list, response_metadata, rerank_metadata
    

    def localize_bugs(self, issue_description: str, repo_content: Dict[str, str], **kwargs) -> Dict[str, Any]:
        generated_files, reranked_files, results_list, response_metadata, rerank_metadata = self._iterate_on_context(issue_description, repo_content)

        # Find valid files from generated files
        final_files = [f for f in generated_files if not isinstance(f, dict) and f in repo_content]
        return {
            "all_generated_files": generated_files,
            "final_files": final_files,
            "reranked_files": reranked_files,
            "raw_responses": results_list,
            "response_metadata": response_metadata,
            "rerank_metadata": rerank_metadata
        }
