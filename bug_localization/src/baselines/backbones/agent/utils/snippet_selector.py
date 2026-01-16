import re
from typing import List, Tuple

_OP_PROXIES = {
    "sum": {"sum", "+=", "accum", "total", "reduce"},
    "round": {"round", "ceil", "floor", "decimal", "BigDecimal"},
    "mod": {"%", "mod"},
    "index": {"len", "size", "index", "i<", "i<=", "i>", "i>="},
    "sort": {"sort", "sorted", "compare", "comparator"},
}

def _extract_cues(text: str) -> dict:
    nums = re.findall(r"\b\d+\b", text)
    words = set(w.lower() for w in re.findall(r"[A-Za-z_]{2,}", text))
    # expand with proxies
    proxy_terms = set()
    for k, vals in _OP_PROXIES.items():
        if k in words:
            proxy_terms |= vals
    return {
        "nums": set(nums),
        "words": words | proxy_terms,
        "err": set(w for w in words if w.endswith("error") or w.endswith("exception")),
    }

def _windows(lines: List[str], win: int = 40, stride: int = 20) -> List[Tuple[int,int]]:
    out = []
    n = len(lines)
    i = 0
    while i < n:
        j = min(n, i + win)
        # avoid ultra-empty windows
        if any(t.strip() for t in lines[i:j]):
            out.append((i, j))
        if j == n: break
        i += stride
    return out

def _score_window(text: str, cues: dict) -> float:
    # cheap tokenization
    toks = re.findall(r"[A-Za-z_]+|[%*/+\-<>=]+|\d+", text)
    toks_low = [t.lower() for t in toks]
    toks_set = set(toks_low)

    s = 0.0
    # numeric literal hits (strong)
    num_hits = sum(1 for n in cues["nums"] if n in toks_set)
    s += 3.0 * num_hits

    # error tokens
    err_hits = sum(1 for e in cues["err"] if e in toks_set)
    s += 2.0 * err_hits

    # word/proxy overlap
    word_hits = sum(1 for w in cues["words"] if w in toks_set)
    s += 1.0 * word_hits

    # structure bonus: looks like a function/method body?
    if re.search(r"\b(def|class|public|private|protected|static|void|int|float|double|fun)\b", text):
        s += 0.5

    # penalize windows that are mostly braces or imports
    if re.search(r"^\s*(import|from\s+\S+\s+import)\b", text, flags=re.M):
        s -= 1.0
    brace_ratio = sum(1 for t in toks if t in {"{","}","(",")",";","[","]"}) / (len(toks)+1e-6)
    if brace_ratio > 0.35:  # mostly scaffolding
        s -= 0.5

    return s

def _mmr_select(cands: List[Tuple[float, Tuple[int,int]]],
                lines: List[str], k: int = 2, lambda_=0.7) -> List[Tuple[int,int]]:
    """Max Marginal Relevance on window bag-of-words overlap."""
    selected = []
    def bow(idx):
        i,j = cands[idx][1]
        return set(re.findall(r"[A-Za-z_]{2,}", "\n".join(lines[i:j]).lower()))
    bows = [bow(i) for i in range(len(cands))]

    while cands and len(selected) < k:
        if not selected:
            best = max(range(len(cands)), key=lambda i: cands[i][0])
        else:
            def mmr(i):
                sim_to_sel = max((len(bows[i] & bows[s]) / (len(bows[i] | bows[s]) + 1e-6)
                                  for s in selected), default=0.0)
                return lambda_ * cands[i][0] - (1 - lambda_) * sim_to_sel
            best = max(range(len(cands)), key=mmr)
        selected.append(best)
        # mark taken
        cands[best], bows[best] = (-1e9, (0,0)), set()
    # return spans
    return [c[1] for i,c in enumerate(cands) if i in selected]

def select_snippets(file_text: str, bug_text: str,
                    max_windows: int = 5, win: int = 60, stride: int = 20) -> List[str]:
    lines = file_text.splitlines()
    cues = _extract_cues(bug_text)
    spans = _windows(lines, win=win, stride=stride)
    scored = []
    for (i,j) in spans:
        txt = "\n".join(lines[i:j])
        scored.append((_score_window(txt, cues), (i,j)))
    if not scored:
        return []

    # sort, then diversify with MMR
    scored.sort(key=lambda x: x[0], reverse=True)
    top_spans = _mmr_select(scored[:10], lines, k=max_windows)  # shortlist 10 for speed/diversity

    snippets = []
    for (i,j) in top_spans:
        # expand to function boundaries a bit (cheap heuristic)
        a = max(0, i-3); b = min(len(lines), j+3)
        snippets.append("\n".join(lines[a:b]))
    return snippets
