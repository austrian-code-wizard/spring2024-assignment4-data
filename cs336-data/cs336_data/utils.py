import re
import os
import re
import nltk
import mmh3
import random
import platform
import fasttext
import unicodedata
from resiliparse import parse
from resiliparse.extract import html2text

nltk.download('punkt')


model_path_prefix = "/home/shared/" if platform.system() != "Darwin" else "./models/"
default_lang_model = fasttext.load_model(model_path_prefix + "lid.176.bin")
default_nsfw_model = fasttext.load_model(model_path_prefix + "dolma-jigsaw-fasttext-bigrams-nsfw.bin")
default_toxic_model = fasttext.load_model(model_path_prefix + "dolma-jigsaw-fasttext-bigrams-hatespeech.bin")


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


def deduplicate_lines(paths: list[os.PathLike], output_dir: os.PathLike):
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
            with open(output_dir / path.name, "w") as out:
                for line in f:
                    h = mmh3.hash(line)
                    if line_counts[h] <= 1:
                        out.write(line)


def compute_minhash(ngrams: list[str], seed: int):
    minhash = float("inf")
    for ngram in ngrams:
        h = mmh3.hash(ngram, seed)
        if h < minhash:
            minhash = h
    return minhash


def normalize_text(text: str):
    text = text.lower()
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # NFD unicode normalization
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')
    return text


def compute_ngrams(text: str, ngram_length: int):
    return [text[i:i+ngram_length] for i in range(len(text) - ngram_length + 1)]


def jaccard_distance(set1: set, set2: set):
    return len(set1.intersection(set2)) / len(set1.union(set2))


def dfs(edges: dict, node: os.PathLike, visited: set) -> set[os.PathLike]:
    visited.add(node)
    cluster = {node}
    for neighbor in edges[node]:
        if neighbor not in visited:
            cluster |= dfs(edges, neighbor, visited)
    return cluster

def deduplicate_fuzzy(all_paths: list[os.PathLike], num_hashes: int, num_bands: int, ngram_length: int, output_dir: os.PathLike, threshold: float):
    clusters = {i: {} for i in range(num_bands)}
    for path in all_paths:
        with open(path, "r") as f:
            text = f.read()
            text = normalize_text(text)
            ngrams = compute_ngrams(text, ngram_length)
            min_hashes = [compute_minhash(ngrams, i) for i in range(num_hashes)]
            for band in range(num_bands):
                band_hashes = tuple(min_hashes[band * (num_hashes // num_bands):(band + 1) * (num_hashes // num_bands)])
                if band_hashes not in clusters[band]:
                    clusters[band][band_hashes] = []
                clusters[band][band_hashes].append(path)

    edges = {}
    for band in range(num_bands):
        for paths in clusters[band].values():
            if len(paths) == 1:
                continue
            ngram_cache = {}
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    if paths[i] not in ngram_cache:
                        with open(paths[i], "r") as f:
                            ngram_cache[paths[i]] = set(compute_ngrams(normalize_text(f.read()), ngram_length))
                    if paths[j] not in ngram_cache:
                        with open(paths[j], "r") as f:
                            ngram_cache[paths[j]] = set(compute_ngrams(normalize_text(f.read()), ngram_length))
                    if jaccard_distance(ngram_cache[paths[i]], ngram_cache[paths[j]]) >= threshold:
                        if paths[i] not in edges:
                            edges[paths[i]] = set()
                        edges[paths[i]].add(paths[j])
                        if paths[j] not in edges:
                            edges[paths[j]] = set()
                        edges[paths[j]].add(paths[i])

    visited = set()
    keep = set(all_paths)
    while len(edges) > 0:
        node = next(iter(edges))
        cluster = dfs(edges, node, visited)
        edges = {k: v for k, v in edges.items() if k not in cluster}
        keep_idx = random.randint(0, len(cluster) - 1)
        cluster = list(cluster)
        cluster.pop(keep_idx)
        keep -= set(cluster)

    for path in keep:
        with open(path, "r") as f:
            with open(output_dir / path.name, "w") as out:
                out.write(f.read())
