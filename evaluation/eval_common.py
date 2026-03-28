from pathlib import Path
from datetime import datetime
import traceback
import mteb
from common.utils import *

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
