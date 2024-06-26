#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Any


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    from cs336_data.utils import extract_text
    return extract_text(html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    from cs336_data.utils import identify_language
    res = identify_language(text)
    return (res[0].replace("__label__", ""), res[1])


def run_mask_emails(text: str) -> tuple[str, int]:
    from cs336_data.utils import mask_emails
    return mask_emails(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    from cs336_data.utils import mask_phone_numbers
    return mask_phone_numbers(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    from cs336_data.utils import mask_ipv4
    return mask_ipv4(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    from cs336_data.utils import classify_nsfw
    res = classify_nsfw(text)
    return (res[0].replace("__label__", ""), res[1])


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    from cs336_data.utils import classify_toxic_speech
    res = classify_toxic_speech(text)
    return (res[0].replace("__label__", ""), res[1])


def run_classify_quality(text: str) -> tuple[Any, float]:
    from cs336_data.utils import identify_quality
    res = identify_quality(text)
    res = (res[0].replace("__label__", ""), res[1])
    res = ("cc" if res[0] == "negative" else "wiki", res[1])
    return res


def run_gopher_quality_filter(text: str) -> bool:
    from cs336_data.utils import gopher_filters
    return gopher_filters(text)


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    from cs336_data.utils import deduplicate_lines
    deduplicate_lines(input_files, output_directory)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    from cs336_data.utils import deduplicate_fuzzy
    deduplicate_fuzzy(
        input_files,
        num_hashes,
        num_bands,
        ngrams,
        output_directory,
        jaccard_threshold,
    )