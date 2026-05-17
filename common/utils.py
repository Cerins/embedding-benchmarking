import os
import sys
import json
from collections import defaultdict
from enum import Enum
import numpy as np
import pandas as pd
import mteb
from datasets import load_dataset_builder


class ScoreSource(Enum):
    CACHE = "cache"
    COMPUTED = "computed"


# The cache dir for mteb
CACHE_DIR = os.environ.get("RESULT_CACHE_DIR")


# Full filepath for a thesis file
def thesis_file(filename):
    thesis_path = os.environ.get("THESIS_PATH")
    if not thesis_path:
        sys.exit("THESIS_PATH not set")
    return os.path.join(thesis_path, filename)


# Max domain dataset
MAX_SIZE = 500_000_000  # 500 MB


# Models used for the domains analysis
def get_models():
    models = [
        "sentence-transformers__all-MiniLM-L6-v2",  # 0.023B
        "BAAI__bge-small-en-v1.5",  # 0.033B
        "ibm-granite__granite-embedding-small-english-r2",  # 0.048B
        "sentence-transformers__all-mpnet-base-v2",  # 0.109B
        "BAAI__bge-base-en-v1.5",  # 0.109B
        "intfloat__multilingual-e5-small",  # 0.118B
        "Alibaba-NLP__gte-base-en-v1.5",  # 0.137B
        "ibm-granite__granite-embedding-english-r2",  # 0.149B
        "jinaai__jina-embeddings-v5-text-nano",  # 0.212B
        "sentence-transformers__paraphrase-multilingual-mpnet-base-v2",  # 0.278B
        "Alibaba-NLP__gte-multilingual-base",  # 0.305B
        "mixedbread-ai__mxbai-embed-large-v1",  # 0.335B
        "BAAI__bge-large-en-v1.5",  # 0.335B
        "WhereIsAI__UAE-Large-V1",  # 0.335B
        "intfloat__multilingual-e5-large-instruct",  # 0.560B
        "Snowflake__snowflake-arctic-embed-l-v2.0",  # 0.568B
        "jinaai__jina-embeddings-v5-text-small",  # 0.596B
    ]
    return models


# Some models for debugging
def get_debug_models():
    models = [
        "sentence-transformers__all-MiniLM-L6-v2",  # 0.023B
        "sentence-transformers__all-mpnet-base-v2",  # 0.109B
    ]
    return models


# Models used for lv task analysis
def get_latvian_models():
    return get_models() + ["lv-mbert-embed-base"]


# AILab models
def get_ailab_models():
    models = [
        "AiLab-IMCS-UL__lv-mbert-mini",
        "AiLab-IMCS-UL__lv-deberta-base",
        "AiLab-IMCS-UL__lv-mbert-base",
        "AiLab-IMCS-UL__lv-mbert-large",
    ]
    return models


# Models used for multilingual analysis
def get_multilingual_models():
    models = [
        "sentence-transformers__all-MiniLM-L6-v2",  # 0.023B
        "BAAI__bge-small-en-v1.5",  # 0.033B
        "sentence-transformers__all-mpnet-base-v2",  # 0.109B
        "BAAI__bge-base-en-v1.5",  # 0.109B
        "intfloat__multilingual-e5-small",  # 0.118B
        "sentence-transformers__paraphrase-multilingual-mpnet-base-v2",  # 0.278B
        "mixedbread-ai__mxbai-embed-large-v1",  # 0.335B
        "BAAI__bge-large-en-v1.5",  # 0.335B
        "WhereIsAI__UAE-Large-V1",  # 0.335B
        "intfloat__multilingual-e5-large-instruct",  # 0.560B
        "Snowflake__snowflake-arctic-embed-l-v2.0",  # 0.568B
    ]
    return models


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


# The good task for domain analydsis
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


def latex_escape(s: str) -> str:
    return (
        s.replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )


def model_short_name(model_name: str) -> str:
    if "/" in model_name:
        return model_name.split("/")[-1]
    return model_name.split("__")[-1]


_TASK_ENG_NAMES_FILE = os.path.join(os.path.dirname(__file__), "task_eng_names.json")
_TASK_MULTI_NAMES_FILE = os.path.join(
    os.path.dirname(__file__), "task_multi_names.json"
)


def get_eng_tasks(domain_filter=False) -> list:
    with open(_TASK_ENG_NAMES_FILE) as f:
        names = json.load(f)
    tasks = mteb.get_tasks(tasks=names)
    if domain_filter:
        tasks = [
            t for t in tasks if any(d in get_task_domains(t) for d in TARGET_DOMAINS)
        ]
    return tasks


def get_multilingual_tasks() -> list:
    with open(_TASK_MULTI_NAMES_FILE) as f:
        names = json.load(f)
    return mteb.get_tasks(tasks=names)


# A method to extract the score, currently only assumes that score is under test
def _extract_ndcg_at_10(data, source):
    test = data.get("scores", {}).get("test", [])
    if test and "ndcg_at_10" in test[0]:
        return test[0]["ndcg_at_10"]
    return None


# Extract the way that score was gotten
def _extract_source(data, source):
    if _extract_ndcg_at_10(data, source) is None:
        return None
    return source


# Walks a results dir and populates model_scores
# It overwrites, so you can go from lowest priorty
def _scan_scores_dir(directory, task_names, score_extractor, model_scores, source):
    if not os.path.isdir(directory):
        return 0
    found_models = 0
    for model_name in os.listdir(directory):
        model_path = os.path.join(directory, model_name)
        if not os.path.isdir(model_path):
            continue
        found_models += 1
        for root, _, files in os.walk(model_path):
            for file in files:
                if not file.endswith(".json") or file == "model_meta.json":
                    continue
                task_name = file[:-5]
                if task_name not in task_names:
                    continue
                try:
                    with open(os.path.join(root, file)) as f:
                        data = json.load(f)
                    score = score_extractor(data, source)
                except (json.JSONDecodeError, KeyError, IOError):
                    continue
                if score is None:
                    continue
                model_scores[model_name][task_name] = score
    return found_models


def _build_scores_df(
    tasks,
    qualified_models,
    score_extractor,
    extra_columns=None,
    drop_empty_rows=False,
    sort_models=False,
    verbose=False,
):
    # Where the mteb remote cache lives (e.g. CQADupstackWordpressRetrieval)
    results_dir = os.path.join(CACHE_DIR, "remote", "results")
    # Where locally-computed results are stored (priority over remote)
    computed_dir = os.path.join(os.path.dirname(__file__), "..", "computed")
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
    if not os.path.exists(computed_dir):
        print(f"Computed directory not found: {computed_dir}")
    if not os.path.exists(results_dir) and not os.path.exists(computed_dir):
        return None
    # Task to domains map
    task_to_domains = {task.metadata.name: get_task_domains(task) for task in tasks}
    # All the task names
    task_names = set(task_to_domains.keys())
    if verbose:
        print(f"Total tasks: {len(tasks)}")
        print("Scanning models...")
    # Store scores
    model_scores = defaultdict(dict)
    # Scan remote results first, then computed so computed overrides on conflict.
    _scan_scores_dir(
        results_dir, task_names, score_extractor, model_scores, ScoreSource.CACHE
    )
    _scan_scores_dir(
        computed_dir, task_names, score_extractor, model_scores, ScoreSource.COMPUTED
    )
    # Calcualte how many tasks per model were done
    model_task_counts = {model: len(scores) for model, scores in model_scores.items()}
    # Sort models based on given sorting algo
    models_in_order = (
        sorted(qualified_models) if sort_models else list(qualified_models)
    )
    if verbose:
        print("Qualified models:")
        for model in sorted(qualified_models):
            print(f"  {model}: {model_task_counts.get(model, 0)} tasks")
    # Additional data to calc
    extra_columns = extra_columns or {}
    # The rows
    rows = []
    for task_name in sorted(task_names):
        # Populate main 2
        row = {"task_name": task_name, "domains": task_to_domains[task_name]}
        # Additional ones
        for col_name, col_data in extra_columns.items():
            row[col_name] = col_data[task_name]
        # Try to find non nan score
        non_nan_found = False
        for model in models_in_order:
            val = model_scores[model].get(task_name, np.nan)
            if not pd.isna(val):
                non_nan_found = True
            row[model] = val
        # Drop empty rows if so
        if drop_empty_rows and not non_nan_found:
            continue
        # Append the rows
        rows.append(row)
    # Compute the dataframe
    df = pd.DataFrame(rows)
    columns = ["task_name", "domains"] + list(extra_columns.keys()) + models_in_order
    # Return dataframe
    # Temporary dropna to ensure no bugs
    return df[columns].dropna()


def get_scores_dataframe(gm=get_models):
    # English tasks
    # With correct domains
    # Extract ndcg
    # Sort them
    # And debug output
    return _build_scores_df(
        tasks=get_eng_tasks(domain_filter=True),
        qualified_models=gm(),
        score_extractor=_extract_ndcg_at_10,
        sort_models=True,
        verbose=True,
    )


def get_multilingual_scores_dataframe(gm=get_multilingual_models):
    # multilingual models
    # Extract ndcg
    # And also store languages
    tasks = get_multilingual_tasks()
    task_to_langs = {
        task.metadata.name: list(task.metadata.languages) for task in tasks
    }
    return _build_scores_df(
        tasks=tasks,
        qualified_models=gm(),
        score_extractor=_extract_ndcg_at_10,
        extra_columns={"languages": task_to_langs},
    )


def get_benchmark_dataframe(gm=get_models, multilingual=False):
    # Get the tasks
    tasks = (
        get_multilingual_tasks() if multilingual else get_eng_tasks(domain_filter=True)
    )
    # Extract the source instad of ndcg
    # And do not keep empty rows
    return _build_scores_df(
        tasks=tasks,
        qualified_models=gm(),
        score_extractor=_extract_source,
        drop_empty_rows=True,
    )


# List of models by size
def sort_models_by_size(models: list[str]) -> list[str]:
    def sort_key(m):
        size = MODEL_TO_PARAM_SIZE.get(m)
        return (size is None, size or 0, m)

    return sorted(models, key=sort_key)


# Model sizes - read from hugging face
MODEL_TO_PARAM_SIZE = {
    "sentence-transformers__all-MiniLM-L6-v2": 23_000_000,
    "BAAI__bge-small-en-v1.5": 33_000_000,
    "ibm-granite__granite-embedding-small-english-r2": 48_000_000,
    "sentence-transformers__all-mpnet-base-v2": 109_000_000,
    "BAAI__bge-base-en-v1.5": 109_000_000,
    "intfloat__multilingual-e5-small": 118_000_000,
    "Alibaba-NLP__gte-base-en-v1.5": 137_000_000,
    "ibm-granite__granite-embedding-english-r2": 149_000_000,
    "jinaai__jina-embeddings-v5-text-nano": 212_000_000,
    "sentence-transformers__paraphrase-multilingual-mpnet-base-v2": 278_000_000,
    "Alibaba-NLP__gte-multilingual-base": 305_000_000,
    "mixedbread-ai__mxbai-embed-large-v1": 335_000_000,
    "BAAI__bge-large-en-v1.5": 335_000_000,
    "WhereIsAI__UAE-Large-V1": 335_000_000,
    "intfloat__multilingual-e5-large-instruct": 560_000_000,
    "Snowflake__snowflake-arctic-embed-l-v2.0": 568_000_000,
    "jinaai__jina-embeddings-v5-text-small": 596_000_000,
}

# Some debug stuff
if __name__ == "__main__":
    print(get_benchmark_dataframe(get_models).dropna())
    print(get_scores_dataframe(get_models).dropna())
    print(get_multilingual_scores_dataframe(get_multilingual_models).dropna())
    print(get_benchmark_dataframe(get_multilingual_models, multilingual=True).dropna())
