import os
import sys
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import mteb
from datasets import load_dataset_builder

CACHE_DIR = os.environ.get("RESULT_CACHE_DIR")


def thesis_file(filename):
    thesis_path = os.environ.get("THESIS_PATH")
    if not thesis_path:
        sys.exit("THESIS_PATH not set")
    return os.path.join(thesis_path, filename)


print("---")
print(CACHE_DIR)
print("---")
# MAX_SIZE = 1_000_000_000  # 1 GB
MAX_SIZE = 500_000_000  # 500 MB
# Tmp only small datasets
# MAX_SIZE = 10_000_000  # 10 MB


def get_english_models():
    models = [
        "sentence-transformers__all-MiniLM-L12-v2",
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


def get_models():
    models = get_english_models()
    # Add multilingual which i can run
    # Switch order back later TODO
    models = [
        "intfloat__multilingual-e5-base",
        "intfloat__multilingual-e5-small",
        "sentence-transformers__paraphrase-multilingual-mpnet-base-v2",
    ] + models
    return models


def get_english_full_models():
    models = get_english_models()
    models = models + [
        "jinaai__jina-embeddings-v2-base-en",  # Weird architecture
        "Snowflake__snowflake-arctic-embed-l",  # Size constraint was the limting factor
        "NovaSearch__stella_en_400M_v5",
        "Qwen__Qwen3-Embedding-0.6B",
        "Alibaba-NLP__gte-Qwen2-1.5B-instruct",
        "nvidia__NV-Embed-v2",
        "intfloat__e5-large-v2",
        "intfloat__e5-base-v2",
        "intfloat__e5-mistral-7b-instruct",
        "mixedbread-ai__mxbai-embed-large-v1",
        "jinaai__jina-embeddings-v4",
        "jinaai__jina-embeddings-v3",  # Closed source next ones
        "openai__text-embedding-3-large",
        "openai__text-embedding-3-small",
        # "voyageai__voyage-4",
        # "voyageai__voyage-3.5",
        # "voyageai__voyage-large-2",
        "google__text-embedding-005",
        "ibm-granite__granite-embedding-english-r2",
    ]
    return models


def get_multilingual_models():
    # Base sources
    models = get_models() + get_english_models() + get_english_full_models()

    # Add strong multilingual-specific models
    multilingual_extra = [
        "Alibaba-NLP__gte-multilingual-base",
        "nomic-ai__nomic-embed-text-v1.5",  # Weird stuff
        "intfloat__multilingual-e5-large",  # stronger than base/small
        "BAAI__bge-m3",  # Multlingual but too large
        "Cohere__Cohere-embed-multilingual-v3.0",
        # "voyageai__voyage-multilingual-2",
        # "google__text-multilingual-embedding-002",
    ]

    models += multilingual_extra

    # Deduplicate while preserving order
    seen = set()
    unique_models = []
    for m in models:
        if m not in seen:
            seen.add(m)
            unique_models.append(m)

    return unique_models


_SIZE_CACHE_FILE = os.path.join(os.path.dirname(__file__), "dataset_size_cache.json")
_size_cache = None


def _load_size_cache():
    global _size_cache
    if _size_cache is None:
        if os.path.exists(_SIZE_CACHE_FILE):
            with open(_SIZE_CACHE_FILE) as f:
                _size_cache = json.load(f)
        else:
            _size_cache = {}
    return _size_cache


def _save_size_cache():
    with open(_SIZE_CACHE_FILE, "w") as f:
        json.dump(_size_cache, f, indent=2)


def dataset_too_large(task, max_size=MAX_SIZE):
    path = task.metadata.dataset.get("path", task.metadata.name)
    revision = task.metadata.dataset.get("revision", None)
    cache_key = f"{path}@{revision}@{max_size}"
    try:
        cache = _load_size_cache()
        # cache = {}
        if cache_key in cache:
            return cache[cache_key]
        builder = load_dataset_builder(path, "corpus", revision=revision)
        info = builder.info
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
        result = size > max_size
        cache[cache_key] = result
        # _save_size_cache()
        return result
    except Exception as e:
        print("Size check failed:", task.metadata.name, e)
        cache = _load_size_cache()
        cache[cache_key] = False
        # _save_size_cache()
        return False


def dataset_too_large_multi(task, max_size=MAX_SIZE):
    path = task.metadata.dataset.get("path", task.metadata.name)
    revision = task.metadata.dataset.get("revision", None)
    cache_key = f"{path}@{revision}@{max_size}"
    try:
        cache = _load_size_cache()
        # cache = {}
        if cache_key in cache:
            return cache[cache_key]
        builder = load_dataset_builder(path, "corpus", revision=revision)
        info = builder.info
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
        result = size > max_size
        cache[cache_key] = result
        # _save_size_cache()
        return result
    except Exception as e:
        print("Size check failed:", task.metadata.name, e)
        cache = _load_size_cache()
        cache[cache_key] = False
        # _save_size_cache()
        return True


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
    "chemistry",
    # "engineering", # Only 1 item
    "programming",
    "financial",
    "fiction",
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


CODE_LANGS = {
    "python",
    "java",
    "javascript",
    "typescript",
    "c",
    "c++",
    "go",
    "php",
    "ruby",
    "rust",
    "scala",
    "swift",
    "shell",
    "sql",
}


def good_multilingual_task(t):
    # Only allow t2t
    if t.metadata.category != "t2t":
        return False
    if t.is_aggregate == True:
        return False
    langs = t.metadata.languages
    # remove programming languages
    filtered = [l for l in langs if l not in CODE_LANGS]
    unique = set(filtered)
    if len(unique) > 1:
        print(filtered)
        return True
    if len(unique) == 1 and "eng" not in unique:
        print(filtered)
        return True
    return False


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


def model_short_name(model_name: str) -> str:
    return model_name.split("__")[-1]


def get_scores_dataframe(target_domains=TARGET_DOMAINS):
    all_tasks = [
        t
        for t in mteb.get_tasks(task_types=["Retrieval"], languages=["eng"])
        if good_task(t) and not dataset_too_large(t)
    ]
    results_dir = os.path.join(CACHE_DIR, "remote", "results")
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return None
    tasks = [t for t in all_tasks if task_has_target_domain(t, target_domains)]
    print(f"Total tasks with target domains: {len(tasks)}")
    print(f"Target domains: {target_domains}")
    task_to_domains = {task.metadata.name: get_task_domains(task) for task in tasks}
    task_names = set(task_to_domains.keys())
    print("Tasks by domain:")
    for domain in target_domains:
        count = sum(1 for d in task_to_domains.values() if domain in d)
        print(f"  {domain}: {count} tasks")
    print("Scanning models...")
    model_dirs = []
    for item in os.listdir(results_dir):
        model_path = os.path.join(results_dir, item)
        if os.path.isdir(model_path):
            model_dirs.append((item, model_path))
    model_task_counts = defaultdict(int)
    model_scores = defaultdict(dict)
    for model_name, model_path in model_dirs:
        for root, dirs, files in os.walk(model_path):
            for file in files:
                if file.endswith(".json") and file != "model_meta.json":
                    task_name = file[:-5]
                    if task_name in task_names:
                        json_path = os.path.join(root, file)
                        try:
                            with open(json_path, "r") as f:
                                data = json.load(f)
                            if (
                                "scores" in data
                                and "test" in data["scores"]
                                and len(data["scores"]["test"]) > 0
                            ):
                                test_scores = data["scores"]["test"][0]
                                if "ndcg_at_10" in test_scores:
                                    model_task_counts[model_name] += 1
                                    model_scores[model_name][task_name] = test_scores[
                                        "ndcg_at_10"
                                    ]
                        except (json.JSONDecodeError, KeyError, IOError):
                            continue
    qualified_models = get_models()
    print(f"Total models found: {len(model_dirs)}")
    print("Qualified models:")
    for model in sorted(qualified_models):
        print(f"  {model}: {model_task_counts[model]} tasks")
    rows = []
    for task_name in sorted(task_names):
        row = {"task_name": task_name, "domains": task_to_domains[task_name]}
        for model in sorted(qualified_models):
            row[model] = model_scores[model].get(task_name, np.nan)
        rows.append(row)
    df = pd.DataFrame(rows)
    columns = ["task_name", "domains"] + sorted(qualified_models)
    return df[columns]


def get_benchmark_dataframe(
    target_domains=TARGET_DOMAINS, gm=get_models, multilingual=False
):
    cache = mteb.ResultCache(CACHE_DIR)
    # cache.download_from_remote()
    task_checker = good_multilingual_task if multilingual else good_task
    task_languges = None if multilingual else ["eng"]
    all_tasks = [
        t
        for t in mteb.get_tasks(task_types=["Retrieval"], languages=task_languges)
        if task_checker(t)
    ]
    tasks = [t for t in all_tasks if task_has_target_domain(t, target_domains)]
    task_to_domains = {task.metadata.name: get_task_domains(task) for task in tasks}
    task_names = set(task_to_domains.keys())
    results_dir = os.path.join(CACHE_DIR, "remote", "results")
    model_task_counts = defaultdict(int)
    model_scores = defaultdict(dict)
    print(os.path.realpath(results_dir))
    for model in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model)
        if not os.path.isdir(model_path):
            continue
        for root, dirs, files in os.walk(model_path):
            for file in files:
                if file.endswith(".json") and file != "model_meta.json":
                    task = file[:-5]
                    if task not in task_names:
                        continue
                    path = os.path.join(root, file)
                    try:
                        with open(path) as f:
                            data = json.load(f)
                        if "scores" in data:
                            model_task_counts[model] += 1
                            model_scores[model][task] = True
                    except:
                        pass
    qualified_models = gm()
    rows = []
    for task in sorted(task_names):
        row = {"task_name": task, "domains": task_to_domains[task]}
        non_nan_found = False
        for model in qualified_models:
            val = model_scores[model].get(task, np.nan)
            if not np.isnan(val):
                non_nan_found = True
            row[model] = val
        if non_nan_found:
            rows.append(row)
    df = pd.DataFrame(rows)
    columns = ["task_name", "domains"] + qualified_models
    return df[columns]
