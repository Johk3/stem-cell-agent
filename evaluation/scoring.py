import re


_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = _ARTICLES.sub(" ", text)
    text = _PUNCT.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def _normalize_number(text: str) -> str:
    """Try to parse as a number and return a canonical string."""
    cleaned = re.sub(r"[,\s]", "", text)
    try:
        val = float(cleaned)
        return str(int(val)) if val == int(val) else str(val)
    except ValueError:
        return text


def is_correct(predicted: str, expected: str) -> bool:
    pred = normalize(predicted)
    exp = normalize(expected)

    if pred == exp:
        return True

    # Numeric comparison: "42.0" == "42", "1,000" == "1000"
    pred_num = _normalize_number(pred)
    exp_num = _normalize_number(exp)
    if pred_num == exp_num and pred_num != pred:
        return True

    # Accept if the expected answer appears as a whole-word match inside the response
    pattern = r"\b" + re.escape(exp) + r"\b"
    if re.search(pattern, pred):
        return True

    return False
