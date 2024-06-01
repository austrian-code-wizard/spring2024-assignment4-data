import os
import json
import gzip
import random
import asyncio
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
random.seed(42)

from tqdm import tqdm
from fastwarc import ArchiveIterator

from cs336_data.utils import extract_text, identify_language, classify_nsfw, classify_toxic_speech, gopher_filters


NUM_SAMPLES = 64000
URL_PATH = "/home/shared/enwiki-20240420-extracted_urls.txt.gz"
OUTPUT_PATH = "./data"


def sample_urls_sharded(num_samples: int, num_shards: int = 16) -> list[str]:
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
    cmd = f"wget --timeout=5 --tries=2 -i {OUTPUT_PATH}/subsampled_positive_urls-{shard_index}.txt --warc-file={OUTPUT_PATH}/warcs/subsampled_positive_urls-{shard_index}.warc -O /dev/null"
    process = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await process.communicate()
    if process.returncode == 0:
        print(f"Shard {shard_index} downloaded successfully.")
    else:
        print(f"Error downloading shard {shard_index}: {stderr.decode()}")


async def download_all_shards(num_shards: int = 16) -> None:
    tasks = [download_shard(i) for i in range(num_shards)]
    await asyncio.gather(*tasks)


def filter_content(content: str) -> bool:
    lang, conf = identify_language(content)
    if lang != '__label__en' or conf < 0.65:
        return False
    nsf, conf = classify_nsfw(content)
    if nsf != '__label__non-nsfw' or conf < 0.9:
        return False
    toxic, conf = classify_toxic_speech(content)
    if toxic != '__label__non-toxic' or conf < 0.9:
        return False
    filter_gopher = gopher_filters(content)
    if not filter_gopher:
        return False
    return True


def filter_documents(warc_file_path: str) -> list[str]:
    documents = []
    for record in tqdm(ArchiveIterator(open(warc_file_path, 'rb')), desc="Processing records"):
        content = record.reader.read()
        content = extract_text(content)
        if not filter_content(content):
            continue
        documents.append(content)
    return documents


def filter_paloma_documents(paloma_path: str = "/home/shared/paloma_c4_100_domains_val/") -> list[str]:
    documents = []
    for file in os.listdir(paloma_path):
        if not file.endswith(".jsonl.gz"):
            continue
        for line in tqdm(gzip.open(f"{paloma_path}/{file}", "rb")):
            content = json.loads(line.decode())["text"].replace("\n", " ")
            if not filter_content(content):
                continue
            documents.append(content)
    return documents


def filter_warc_file(file_path: str, num_samples: int, total_records: int, idx: int):
    count = 0
    with open(f"{OUTPUT_PATH}/train_neg_{idx}.txt", "w+") as f:
        for record in tqdm(ArchiveIterator(open(file_path, 'rb')), desc="Processing records"):
            if random.random() > num_samples / total_records:
                continue
            if count >= num_samples:
                print(f"Found {count} negative samples for shard {idx}.")
                return
            content = record.reader.read()
            content = extract_text(content).replace("\n", " ")
            f.write(f"__label__negative {content}\n")
            count += 1
        print(f"Found {count} negative samples for shard {idx}.")
        return
    
def get_negative_samples(num_total: int, warc_folder_path: int = "/home/shared/CC-MAIN-2023-50-warc-filtered") -> list[str]:
    warc_files = [os.path.join(warc_folder_path, f) for f in os.listdir(warc_folder_path) if f.endswith(".warc.filtered.gz")]
    count_per_file = num_total // len(warc_files)
    total_records = 3000
    indices = list(range(len(warc_files)))
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        _ = list(executor.map(filter_warc_file, warc_files, [count_per_file] * len(warc_files), [total_records] * len(warc_files), indices))


def create_training_data() -> None:
    with open(f"{OUTPUT_PATH}/train_pos.txt", "w+") as f:
        positive_samples = filter_paloma_documents()
        positive_samples = [f"__label__positive {doc}\n" for doc in positive_samples]
        f.writelines(positive_samples)
    print(f"Found {len(positive_samples)} positive samples.")
    get_negative_samples(len(positive_samples) * 3)
    print(f"Training data written.")


if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    #sample_urls_sharded(NUM_SAMPLES)
    #asyncio.run(download_all_shards())
    create_training_data()