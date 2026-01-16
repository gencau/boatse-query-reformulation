import json
import re
from typing import List, Dict, Any, Optional

from src.baselines.backbones.chat.prompts.chat_base_prompt import ChatBasePrompt
from src.utils.tokenization_utils import TokenizationUtils


def check_match_context_size(tokenization_utils: TokenizationUtils,
                             prompt: ChatBasePrompt,
                             issue_description: str,
                             project_content: Dict[str, str],
                             is_chat: bool):
    if is_chat:
        messages = prompt.chat(issue_description, project_content)
        return tokenization_utils.messages_match_context_size(messages)

    text = prompt.complete(issue_description, project_content)
    return tokenization_utils.text_match_context_size(text)

def split_into_chunks(text: str,
                      token_utils: TokenizationUtils,
                      max_tokens: int) -> List[str]:
    """
    Tokenizes `text`, then splits tokens into consecutive
    slices of length ≤ max_tokens, and detokenizes each slice.
    """
    # turn the text into a flat list of tokens
    tokens = token_utils.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        slice_tokens = tokens[i : i + max_tokens]
        chunks.append(token_utils.detokenize(slice_tokens))
    return chunks

def batch_project_context(model: str,
                          prompt: ChatBasePrompt,
                          issue_description: str,
                          project_content: Dict[str, str],
                          is_chat: bool) -> List[Dict[str, str]]:
    """
    Greedily group files so each batch fits in the model context,
    using check_match_context_size() to test each candidate batch.
    """
    token_utils = TokenizationUtils(model)
    file_paths = list(project_content.keys())

    if not file_paths:
        print(f"###### WARNING: Got 0 files for project content")
        return []

    batches: List[Dict[str, str]] = []
    current_batch_paths: List[str] = []

    for f in file_paths:
        # try adding this file onto the current batch
        candidate_paths = current_batch_paths + [f]
        candidate_content = {p: project_content[p] for p in candidate_paths}

        if check_match_context_size(token_utils,
                                    prompt,
                                    issue_description,
                                    candidate_content,
                                    is_chat):
            # still fits—keep growing
            current_batch_paths = candidate_paths
        else:
            # flush the previous batch (if any)
            if current_batch_paths:
                batches.append({
                    p: project_content[p]
                    for p in current_batch_paths
                })
            current_batch_paths = [f]

    # flush any trailing batch
    if current_batch_paths:
        batches.append({
            p: project_content[p]
            for p in current_batch_paths
        })

    return batches

def deduplicate_files(files):
    """
        Removes duplicates and invalid formats (dict) inserted within JSON objects.
    """
    unique = []
    seen = set()
    
    for item in files:
        # For dictionaries, convert to a tuple of sorted (key, value) pairs.
        if isinstance(item, dict):
            #skip it
            continue
        else:
            # Assuming item is already hashable (like a string)
            key = item
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique

def parse_json_response(response: str) -> list:
    files = []
    patterns = [
        r"```json\s*(\{.*?\})\s*```",   # JSON inside markdown code blocks
        r'</think>\s*(\{.*?\})',         # JSON after a </think> marker
        r"(\{.*?\})"                    # Fallback: any JSON-like object
    ]
    
    # Extract and process JSON objects
    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        for json_str in matches:
            try:
                data = json.loads(json_str)
                if "files" in data:
                    files.extend(data["files"])
            except json.JSONDecodeError:
                print("#### JSON decoding error for match:", json_str)
    
    # Remove markdown code blocks before inline extraction to avoid double matching.
    cleaned_result = re.sub(r"```.*?```", "", response, flags=re.DOTALL)
    # Also remove <think> tags so that we don't match files mentioned there
    cleaned_result = re.sub(r"<think>.*?</think>", "", cleaned_result, flags=re.DOTALL)
    if len(files) == 0:
        inline_matches = re.findall(r'`([^`]+)`', cleaned_result)
        # Filter inline matches: accept only candidates that appear to contain files (i.e. contain '.')
        valid_inline_matches = [m for m in inline_matches if '.' in m]
        files.extend(valid_inline_matches)

    # Get rid of any duplicates
    files = deduplicate_files(files)

    return files

def parse_list_files_completion(raw_completion: str,) -> List[str]:
    return parse_json_response(raw_completion)