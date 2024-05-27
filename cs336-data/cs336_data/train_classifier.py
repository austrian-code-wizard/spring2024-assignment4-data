import os
import gzip
import random
import asyncio
random.seed(42)

from tqdm import tqdm
from fastwarc import ArchiveIterator

#from cs336_data.utils import extract_text, identify_language, mask_emails, classify_nsfw, classify_toxic_speech, gopher_filters


NUM_SAMPLES = 64000
URL_PATH = "enwiki-20240420-extracted_urls.txt.gz"
OUTPUT_PATH = "./data"


def sample_urls_sharded(num_samples: int, num_shards: int = 64) -> list[str]:
    with gzip.open(URL_PATH, "rt") as f:
        urls = [l for l in tqdm(f.readlines())]
    urls = random.sample(urls, num_samples)
    print(f"Sampled {len(urls)} URLs.")
    shards = [urls[i::num_shards] for i in range(num_shards)]
    for i, shard in enumerate(shards):
        with open(f"{OUTPUT_PATH}/subsampled_positive_urls-{i}.txt", "w") as f:
            f.writelines(shard)
    print(f"Shards written to {OUTPUT_PATH}.")


async def download_shard(shard_index: int) -> None:
    cmd = f"wget --timeout=5 -i {OUTPUT_PATH}/subsampled_positive_urls-{shard_index}.txt --warc-file={OUTPUT_PATH}/warcs/subsampled_positive_urls-{shard_index}.warc -O /dev/null"
    process = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await process.communicate()
    if process.returncode == 0:
        print(f"Shard {shard_index} downloaded successfully.")
    else:
        print(f"Error downloading shard {shard_index}: {stderr.decode()}")


async def download_all_shards(num_shards: int = 64) -> None:
    tasks = [download_shard(i) for i in range(num_shards)]
    await asyncio.gather(*tasks)


def filter_documents(warc_file_path: str) -> list[str]:
    documents = []
    for record in tqdm(ArchiveIterator(open(warc_file_path, 'rb')), desc="Processing records"):
        content = record.reader.read()
        content = extract_text(content)
        lang, conf = identify_language(content)
        if lang != '__label__en' or conf < 0.65:
            continue
        nsf, conf = classify_nsfw(content)
        if nsf != '__label__non-nsfw' or conf < 0.9:
            continue
        toxic, conf = classify_toxic_speech(content)
        if toxic != '__label__non-toxic' or conf < 0.9:
            continue
        filter_gopher = gopher_filters(content)
        if not filter_gopher:
            continue
        documents.append(content)
    return documents



if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    #sample_urls_sharded(NUM_SAMPLES)
    asyncio.run(download_all_shards())