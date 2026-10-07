from types import SimpleNamespace

from scripts.ask import run_interactive


class FakeAnswerService:
    def __init__(self):
        self.questions = []

    def answer(self, question):
        self.questions.append(question)
        return SimpleNamespace(answer=f"Answer to: {question}")


def test_interactive_cli_prints_answers_and_stops(monkeypatch, capsys):
    service = FakeAnswerService()
    questions = iter(["1", "What is a tornado warning?", "STOP", "ignored"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(questions))

    run_interactive(service)

    assert service.questions == [
        "What does a severe thunderstorm warning mean?",
        "What is a tornado warning?",
    ]
    output = capsys.readouterr().out
    assert "1. What does a severe thunderstorm warning mean?" in output
    assert "Answer to: What does a severe thunderstorm warning mean?" in output
    assert "Answer to: What is a tornado warning?" in output
