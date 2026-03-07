import os
from datasets import load_dataset_builder

CACHE_DIR = os.environ.get("RESULT_CACHE_DIR")
print("---")
print(CACHE_DIR)
print("---")
# MAX_SIZE = 1_000_000_000  # 1 GB
MAX_SIZE = 500_000_000  # 500 MB
# Tmp only small datasets
# MAX_SIZE = 10_000_000  # 10 MB


def get_models():
    models = [
        "sentence-transformers__all-MiniLM-L6-v2",
        "BAAI__bge-small-en-v1.5",
        "thenlper__gte-small",
        "Snowflake__snowflake-arctic-embed-xs",
        "sentence-transformers__all-mpnet-base-v2",
        "BAAI__bge-base-en-v1.5",
        "thenlper__gte-base",
        "Snowflake__snowflake-arctic-embed-m",
        # "jinaai__jina-embeddings-v2-base-en",
        "BAAI__bge-large-en-v1.5",
        "Snowflake__snowflake-arctic-embed-l",
        # "NovaSearch__stella_en_400M_v5",
        "Qwen__Qwen3-Embedding-0.6B",
        "Alibaba-NLP__gte-Qwen2-1.5B-instruct",
        # "nvidia__NV-Embed-v2",
    ]
    # Restrict models temprorary
    models = [
        "sentence-transformers__all-MiniLM-L6-v2",
        "BAAI__bge-small-en-v1.5",
        "thenlper__gte-small",
        "Snowflake__snowflake-arctic-embed-xs",
        "sentence-transformers__all-mpnet-base-v2",
        "BAAI__bge-base-en-v1.5",
        "thenlper__gte-base",
        "Snowflake__snowflake-arctic-embed-m",
        "BAAI__bge-large-en-v1.5",
    ]
    return models


def dataset_too_large(task, max_size=MAX_SIZE):
    try:
        path = task.metadata.dataset["path"]
        revision = task.metadata.dataset.get("revision", None)
        builder = load_dataset_builder(path, "corpus", revision=revision)
        info = builder.info
        # print("INFO", info)
        # Added num_bytes to handle grid
        size = 0
        try:
            size = info.download_size or info.dataset_size or 0
        except Exception:
            pass
        if size == 0:
            try:
                size = sum(split.num_bytes for split in info.splits.values()) or 0
            except Exception:
                pass
        print(f"{path}: {size / 1e6:.1f} MB")
        return size > max_size
    except Exception as e:
        print("Size check failed:", task.metadata.name, e)
        return False


TARGET_DOMAINS_UNUSED = [
    "legal",
    "medical",
    "chemistry",
    "engineering",
    "programming",
    "financial",
    "fiction",
]

TARGET_DOMAINS = [
    "legal",
    "medical",
    # "chemistry",  # These 2 have large overlap with legal
    # "engineering",
    "programming",
    "financial",
    # "fiction", # Not enoguh items to make sense to use
]


def good_task(t):
    # Only allow t2t
    if t.metadata.category != "t2t":
        return False
    if t.is_aggregate == True:
        return False
    # Find if there exists language that is 3 letters long and not eng
    for lang in t.metadata.languages:
        if len(lang) == 3 and lang != "eng":
            return False
    return True


def get_task_domains(task):
    if not hasattr(task.metadata, "domains") or not task.metadata.domains:
        return []
    domains = []
    for domain in task.metadata.domains:
        dl = domain.lower()
        # Basically only show the interesting domains
        # if dl in TARGET_DOMAINS_UNUSED:
        domains.append(dl)
    return domains


def task_has_target_domain(task, target_domains):
    domains = get_task_domains(task)
    for td in target_domains:
        # print(td, target_domains, domains)
        if td in domains:
            return True
    return False


def mteb_to_hf_repo(model_name: str) -> str:
    if "__" not in model_name:
        return model_name  # already correct
    org, rest = model_name.split("__", 1)
    return f"{org}/{rest}"


def hf_repo_to_mteb(repo_name: str) -> str:
    if "/" not in repo_name:
        return repo_name  # already correct
    org, rest = repo_name.split("/", 1)
    return f"{org}__{rest}"
