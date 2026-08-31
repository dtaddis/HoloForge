from holoforge.app import clean_optional_prompt


def test_optional_prompt_accepts_untouched_gradio_value() -> None:
    assert clean_optional_prompt(None) is None
    assert clean_optional_prompt("") is None
    assert clean_optional_prompt("   ") is None
    assert clean_optional_prompt("  continue the blue door  ") == "continue the blue door"
