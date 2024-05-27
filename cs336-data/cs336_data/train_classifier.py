import os
import gzip
import random
random.seed(42)


NUM_SAMPLES = 64000
URL_PATH = "/home/shared/enwiki-20240420-extracted_urls.txt.gz"
OUTPUT_PATH = "./data"


def sample_urls_sharded(num_samples: int, num_shards: int = 64) -> list[str]:
    with gzip.open(URL_PATH, "rt") as f:
        urls = f.readlines()
    urls = random.sample(urls, num_samples)
    shards = [urls[i::num_shards] for i in range(num_shards)]
    for i, shard in shards:
        with open(f"{OUTPUT_PATH}/subsampled_positive_urls-{i}.txt", "w") as f:
            f.writelines(shard)


if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    sample_urls_sharded(NUM_SAMPLES)