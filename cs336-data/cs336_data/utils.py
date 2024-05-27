import re
import nltk
import mmh3
import fasttext
from resiliparse import parse
from resiliparse.extract import html2text

nltk.download('punkt')


default_lang_model = fasttext.load_model("/home/shared/lid.176.bin")
default_nsfw_model = fasttext.load_model("/home/shared/dolma-jigsaw-fasttext-bigrams-nsfw.bin")
default_toxic_model = fasttext.load_model("/home/shared/dolma-jigsaw-fasttext-bigrams-hatespeech.bin")


def extract_text(inp: bytes) -> str:
    encoding = parse.encoding.detect_encoding(inp)
    decoded = inp.decode(encoding, errors="replace")
    return html2text.extract_plain_text(decoded)


def identify_language(text: str, model: str | None = None) -> tuple[str, float]:
    if model is None:
        model = default_lang_model
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


def classify_nsfw(text: str, model: str | None = None) -> tuple[str, float]:
    if model is None:
        model = default_nsfw_model
    else:
        model = fasttext.load_model(model)
    results = model.predict(text.replace("\n", " "), k=1)
    results = [(l, v) for l, v in zip(*results)]
    return sorted(results, key=lambda x: x[1], reverse=True)[0]


def classify_toxic_speech(text: str, model: str | None = None) -> tuple[str, float]:
    if model is None:
        model = default_toxic_model
    else:
        model = fasttext.load_model(model)
    results = model.predict(text.replace("\n", " "), k=1)
    results = [(l, v) for l, v in zip(*results)]
    return sorted(results, key=lambda x: x[1], reverse=True)[0]


def gopher_filters(text: str) -> bool:
    """Returns bool indicating whether to keep the document"""
    words = nltk.word_tokenize(text)

    # Filter out documents with less than 50 words or more than 100,000 words
    if len(words) < 50 or len(words) > 100000:
        return False
    
    # Filter out docs with mean word length outside of 3-10 char range
    if not 3 <= sum(len(word) for word in words) / len(words) <= 10:
        return False
    
    # Filter out docs with more than 30% of lines ending with "..."
    lines = text.splitlines()
    if sum(line.endswith("...") for line in lines) / len(lines) > 0.3:
        return False
    
    # Filter out docs where less than 80% of words have an alphabetic character
    if sum(any(c.isalpha() for c in word) for word in words) / len(words) < 0.8:
        return False
    
    return True


def deduplicate_lines(paths: list[str], output_dir: str):
    line_counts = {}
    for path in paths:
        with open(path, "r") as f:
            for line in f:
                h = mmh3.hash(line)
                if h not in line_counts:
                    line_counts[h] = 0
                line_counts[h] += 1
    
    for path in paths:
        with open(path, "r") as f:
            with open(f"{output_dir}/{path.split('/')[-1]}", "w") as out:
                for line in f:
                    h = mmh3.hash(line)
                    if line_counts[h] <= 1:
                        out.write(line)
