# inference_worker.py
import sys, json, warnings, os
from contextlib import redirect_stderr

def main():
    # Hide noisy UserWarnings from libs (optional)
    warnings.filterwarnings("ignore", category=UserWarning)
    # Silence stderr from libs that write banners to stderr (optional)
    devnull = open(os.devnull, "w")
    with redirect_stderr(devnull):
        from langchain_ollama import ChatOllama

        payload = json.load(sys.stdin)
        messages = payload["messages"]
        model_kwargs = payload["model_kwargs"]

        model = ChatOllama(**model_kwargs)
        resp = model.invoke(messages)
        content = getattr(resp, "content", str(resp))

    json.dump({"ok": True, "content": content}, sys.stdout)
    sys.stdout.flush()

if __name__ == "__main__":
    main()
