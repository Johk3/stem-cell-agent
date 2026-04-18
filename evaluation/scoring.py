import re


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_correct(predicted: str, expected: str) -> bool:
    pred = normalize(predicted)
    exp = normalize(expected)

    if pred == exp:
        return True

    # Accept if the expected answer appears as a whole-word match inside the response
    pattern = r"\b" + re.escape(exp) + r"\b"
    if re.search(pattern, pred):
        return True

    return False
