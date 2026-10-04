import re

def clean_text(text):
    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)

    text = text.translate(str.maketrans({
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t"
    }))

    text = re.sub(r"\b(?:[a-z]\s+){2,}[a-z]\b",
                  lambda m: m.group(0).replace(" ", ""),
                  text)

    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    text = re.sub(r"[^\w\s\u0C00-\u0C7F]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()