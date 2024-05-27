import fasttext
from resiliparse import parse
from resiliparse.extract import html2text


default_model = fasttext.load_model("/home/shared/lid.176.bin")


def extract_text(inp: bytes) -> str:
    encoding = parse.encoding.detect_encoding(inp)
    decoded = inp.decode(encoding, errors="replace")
    return html2text.extract_plain_text(decoded)


def identify_language(text: str, model: str | None = None) -> tuple[str, float]:
    if model is None:
        model = default_model
    else:
        model = fasttext.load_model(model)
    results = model.predict(text)
    results = [(l.replace("__label__", ""), v) for l, v in zip(*results)]
    return sorted(results, key=lambda x: x[1], reverse=True)[0]