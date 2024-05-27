import re
import fasttext
from resiliparse import parse
from resiliparse.extract import html2text


#default_model = fasttext.load_model("/home/shared/lid.176.bin")


def extract_text(inp: bytes) -> str:
    encoding = parse.encoding.detect_encoding(inp)
    decoded = inp.decode(encoding, errors="replace")
    return html2text.extract_plain_text(decoded)


def identify_language(text: str, model: str | None = None) -> tuple[str, float]:
    if model is None:
        model = default_model
    else:
        model = fasttext.load_model(model)
    results = model.predict(text.replace("\n", " "), k=1)
    results = [(l, v) for l, v in zip(*results)]
    return sorted(results, key=lambda x: x[1], reverse=True)[0]


def mask_emails(text: str) -> tuple[str, int]:
    masked_text, num_masks = re.subn(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "|||EMAIL_ADDRESS|||", text)
    return masked_text, num_masks


def mask_phone_numbers(text: str) -> tuple[str, int]:
    masked_text, num_masks = re.subn(
        r"(\b\d{3}[-.]?\d{3}[-.]?\d{4}\b)|(\+\d{1,2}\s?\(?\d{2,3}\)?\s?\d{3,4}[-.]?\d{4})|(\b\d{2,4}[-.]?\d{2,4}[-.]?\d{2,4}\b)", "|||PHONE_NUMBER|||", text)
    return masked_text, num_masks

def mask_ipv4(text: str) -> tuple[str, int]:
    masked_text, num_masks = re.subn(r"\b(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b", "|||IP_ADDRESS|||", text)
    return masked_text, num_masks