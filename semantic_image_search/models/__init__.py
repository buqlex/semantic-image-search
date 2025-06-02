import os

from huggingface_hub.constants import default_cache_path

HUGGINGFACE_CACHE = os.getenv("MODEL_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
if "MODEL_CACHE_DIR" in os.environ:
    if not os.path.isdir(HUGGINGFACE_CACHE):
        os.makedirs(HUGGINGFACE_CACHE, exist_ok=True)

os.environ["HF_HOME"] = HUGGINGFACE_CACHE
os.environ["TRANSFORMERS_CACHE"] = HUGGINGFACE_CACHE
