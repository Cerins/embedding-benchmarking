import multiprocessing as mp
import time
import mteb
from common.utils import *
from evaluation.eval_common import run_single_task, already_processed, RESULTS_DIR

POST_PROCESS_SLEEP_SECONDS = 2

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    all_tasks = [
        t
        for t in mteb.get_tasks(task_types=["Retrieval"])
        if good_multilingual_task(t)
    ]
    tasks = [t for t in all_tasks if task_has_target_domain(t, TARGET_DOMAINS)]
    cache = mteb.ResultCache(CACHE_DIR)
    models = get_multilingual_models()

    for m in models:
        model_hf_repo = mteb_to_hf_repo(m)
        existing = cache.load_results(
            models=[model_hf_repo], tasks=tasks, include_remote=True
        )
        existing_task_names = set(existing.task_names)
        missing_tasks = [t for t in tasks if t.metadata.name not in existing_task_names]

        for i, task in enumerate(missing_tasks):
            if dataset_too_large_multi(task):
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
