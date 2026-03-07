from pathlib import Path
from datetime import datetime
import traceback
import multiprocessing as mp
import time
import mteb
from common.utils import *

POST_PROCESS_SLEEP_SECONDS = 2

MODE = "cuda"

RESULTS_DIR = os.environ.get('RESULTS_DIR')

def run_single_task(model_hf_repo: str, task_name: str, output_folder: str):
    try:
        task = mteb.get_tasks(tasks=[task_name])[0]
        evaluation = mteb.MTEB(tasks=[task])
        evaluation.run(
            model=mteb.get_model(model_hf_repo, device=MODE),
            show_progress=True,
            output_folder=output_folder,
        )
    except Exception:
        log_path = Path(output_folder) / "error.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"[{datetime.now().isoformat()}] Model: {model_hf_repo} | Task: {task_name}\n"
            )
            f.write(traceback.format_exc())
            f.write("\n" + "-" * 80 + "\n")
        raise


def already_processed(model_name, task):
    base = Path(RESULTS_DIR) / model_name
    if not base.exists() or not base.is_dir():
        return False
    subdirs = [p for p in base.iterdir() if p.is_dir()]
    if len(subdirs) != 1:
        return False
    return (subdirs[0] / f"{task.metadata.name}.json").exists()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    all_tasks = [
        t
        for t in mteb.get_tasks(task_types=["Retrieval"], languages=["eng"])
        if good_task(t)
    ]
    tasks = [t for t in all_tasks if task_has_target_domain(t, TARGET_DOMAINS)]
    cache = mteb.ResultCache(CACHE_DIR)
    models = get_models()

    for m in models:
        model_hf_repo = mteb_to_hf_repo(m)
        existing = cache.load_results(
            models=[model_hf_repo], tasks=tasks, include_remote=True
        )
        existing_task_names = set(existing.task_names)
        missing_tasks = [t for t in tasks if t.metadata.name not in existing_task_names]

        for i, task in enumerate(missing_tasks):
            if dataset_too_large(task):
                print("Skipping too large:", task.metadata.name)
                continue
            if already_processed(m, task):
                print("Skipping already done:", m, task.metadata.name)
                continue
            print("Processing", i, "out of ", len(missing_tasks), task.metadata.name, m)
            p = mp.Process(
                target=run_single_task,
                args=(model_hf_repo, task.metadata.name, RESULTS_DIR),
            )
            p.start()
            p.join()
            time.sleep(POST_PROCESS_SLEEP_SECONDS)

            if p.exitcode != 0:
                print("FAILED:", m, task.metadata.name)
            else:
                print("DONE:", m, task.metadata.name)
