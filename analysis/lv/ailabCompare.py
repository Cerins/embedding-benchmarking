from common.utils import thesis_file, get_models, get_ailab_models
from analysis.lv.compare import load_scores, plot

TASKS = ["WikiLV", "MultiEupV2LV"]


def main():
    all_models = set(get_models() + get_ailab_models())
    model_scores = {
        m: s for m, s in load_scores(TASKS).items() if m in all_models
    }
    print(f"Models with full coverage: {len(model_scores)}")
    plot(model_scores, TASKS, thesis_file("lv_compare_ailab.png"))


if __name__ == "__main__":
    main()
