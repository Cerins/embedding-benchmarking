# Helper script which is made to load mteb cache, if not already exists
import mteb
from common.utils import *

if __name__ == "__main__":
    cache = mteb.ResultCache(cache_path=CACHE_DIR)
    cache.download_from_remote()
    print("Cache loaded")
