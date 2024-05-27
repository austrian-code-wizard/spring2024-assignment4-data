from resiliparse import parse
from resiliparse.extract import html2text


def extract_text(inp: bytes) -> str:
    encoding = parse.encoding.detect_encoding(inp)
    decoded = inp.decode(encoding)
    return html2text.extract_plain_text(decoded)
