from sentence_transformers import SentenceTransformer
from common.utils import *
import mteb
import os

all_tasks = [
    t for t in mteb.get_tasks(task_types=["Retrieval"]) if good_multilingual_task(t)
]

# print(len(mteb.get_tasks(task_types=["Retrieval"])))

# print(len(all_tasks))

tasks = all_tasks

total = 0

for i, task in enumerate(all_tasks):
    print("Processing", i, "out of ", len(tasks), task)
    if dataset_too_large_multi(task):
        print("Skipping too large:", task.metadata.name)
        continue
    task.load_data()
    total += 1

print("Downloaded", total)
