from app.generation.prompts import SYSTEM_PROMPT, user_prompt


def test_system_prompt_requires_grounding() -> None:
    assert "only from the provided context" in SYSTEM_PROMPT
    assert "do not know" in SYSTEM_PROMPT.lower()


def test_user_prompt_includes_question_and_context() -> None:
    prompt = user_prompt("Does RingCentral integrate with Salesforce?", "[1] Salesforce is supported.")
    assert "Does RingCentral integrate with Salesforce?" in prompt
    assert "[1] Salesforce is supported." in prompt


def test_user_prompt_handles_empty_context() -> None:
    assert "No retrieved context." in user_prompt("What is Nextiva?", "")
