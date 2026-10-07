"""Ask questions from the terminal. Run from the project root.

    python scripts/ask.py                 # interactive; type "stop" to quit
    python scripts/ask.py "your question" # one-shot
"""
import os
import sys
import logging

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)

from app.rag.retrieval import Retriever
from app.services.answer_service import AnswerService, openai_llm

DEFAULT_QUESTIONS = {
    "1": "What does a severe thunderstorm warning mean?",
    "2": "What is the difference between a watch and a warning?",
    "3": "How much rain is considered extremely heavy rain in IMD?",
    "4": "What causes dust storms in Delhi?",
    "5": "What is heat index?",
}


def show(service: AnswerService, question: str) -> None:
    print(service.answer(question).answer)


def run_interactive(service: AnswerService) -> None:
    print("Choose a question by number, or type your own question:")
    for number, question in DEFAULT_QUESTIONS.items():
        print(f"{number}. {question}")
    print("Type 'stop' to quit.")

    while True:
        try:
            selection = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if selection.casefold() == "stop":
            return
        if selection in DEFAULT_QUESTIONS:
            show(service, DEFAULT_QUESTIONS[selection])
        elif selection:
            show(service, selection)


def main() -> None:
    service = AnswerService(Retriever(backend="local"), openai_llm())
    if len(sys.argv) > 1:
        show(service, " ".join(sys.argv[1:]))
    else:
        run_interactive(service)


if __name__ == "__main__":
    main()
