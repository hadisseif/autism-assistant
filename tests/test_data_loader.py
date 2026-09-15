from pathlib import Path

from data_loader import _merge_fields, extract_label_text, load_samples_fa


def test_merge_fields_combines_non_empty_values():
    row = {"instruction": "Speak clearly", "input": "", "output": "Response"}
    text = _merge_fields(row, ("instruction", "input", "output"))

    assert "instruction: Speak clearly" in text
    assert "output: Response" in text
    assert "input:" not in text


def test_merge_fields_returns_empty_for_no_text():
    row = {"instruction": "", "input": None, "output": "  "}
    assert _merge_fields(row, ("instruction", "input", "output")) == ""


def test_extract_label_text_uses_patient_input():
    row = {
        "instruction": (
            "You are a helpful mental health counselling assistant, "
            "please answer the mental health questions."
        ),
        "input": "My child gets upset when the daily plan changes suddenly.",
        "output": "Support and therapy can improve quality of life.",
    }

    assert extract_label_text(row, mode="patient") == row["input"]

    all_text = extract_label_text(row, mode="all")
    assert "instruction:" in all_text
    assert "output:" in all_text


def test_patient_mode_skips_when_input_is_missing():
    row = {
        "instruction": "A non-generic instruction that used to be a fallback.",
        "output": "A counseling response.",
    }

    assert extract_label_text(row, mode="patient") == ""


def test_patient_mode_skips_when_input_is_empty_even_if_tagged_text_exists():
    row = {
        "text": (
            "instruction: احساس اضطراب دارم.\n"
            "input: \n"
            "output: ساختار و پیش‌بینی‌پذیری معمولاً کمک‌کننده است."
        )
    }

    assert extract_label_text(row, mode="patient") == ""


def test_patient_mode_can_extract_input_from_tagged_text():
    row = {
        "text": (
            "instruction: پاسخ مناسب بده.\n"
            "input: وقتی برنامه روزانه‌ام عوض می‌شود خیلی مضطرب می‌شوم.\n"
            "output: می‌توانیم راهبردهایی بررسی کنیم."
        )
    }

    text = extract_label_text(row, mode="patient")
    assert "برنامه روزانه" in text
    assert "راهبردهایی" not in text


def test_load_samples_fa_from_sample_file():
    path = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "mentalchat16k_fa.sample.jsonl"
    )

    # این تست فقط وقتی fixture دارای input باشد معنی دارد.
    samples = load_samples_fa(path=path, limit=3)

    assert len(samples) <= 3
    for sample in samples:
        assert sample.text
        assert "output:" not in sample.text.lower()
