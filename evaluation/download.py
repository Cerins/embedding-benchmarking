from sentence_transformers import SentenceTransformer
from common.utils import *
import mteb

models = get_models()

all_tasks = [
    t
    for t in mteb.get_tasks(task_types=["Retrieval"], languages=["eng"])
    if good_task(t)
]

# Filter to only tasks with TARGET_DOMAINS
tasks = [t for t in all_tasks if task_has_target_domain(t, TARGET_DOMAINS)]

for i, task in enumerate(tasks):
    print("Processing", i, "out of ", len(tasks), task)
    if dataset_too_large(task):
        print("Skipping too large:", task.metadata.name)
        continue
    task.load_data()


for m in models:
    model = mteb_to_hf_repo(m)
    print("Working on", model)
    m = mteb.get_model(model, device="cpu")
