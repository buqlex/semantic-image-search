import os
os.environ["HF_HOME"] = os.getenv("MODEL_CACHE_DIR", os.path.expanduser("~/.cache/huggingface"))

from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
from functools import cache
import warnings
import argparse
import time
import socket

from PIL import Image, ExifTags
import gradio as gr
from transformers import pipeline

from semantic_image_search.models.documents import ImageVectorStore
from semantic_image_search.models.rerank import Reranker
from semantic_image_search.config import CHROMA_PATH

translator = pipeline("translation_ru_to_en", model="Helsinki-NLP/opus-mt-ru-en")

try:
    import clip
except ImportError:
    warnings.warn("CLIP is not installed, reranking will be disabled")
    clip = None

rerankers = {
    "standard": None,
    "fast": Reranker("ViT-B/32") if clip else None,
    "accurate": Reranker("ViT-L/14") if clip else None
}

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    raise ImportError("Для поддержки HEIC установите 'pillow-heif': pip install pillow-heif")

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--chroma_path", type=str, default=CHROMA_PATH, help="Override the path to ChromaDB")
args = parser.parse_args()

photo_store = ImageVectorStore(chroma_persist_path=args.chroma_path)
print(f"Количество изображений в базе: {len(photo_store)}")
OUTPUT_TYPE = "pil"
MAX_WORKERS = 4

@cache
def _get_rotation_key() -> int:
    return max(ExifTags.TAGS.items(), key=lambda x: x[1] == 'Orientation', default=(-1, None))[0]

def search(query: str, score_threshold: float, rerank_mode: str, lang: str) -> List[Tuple[Image.Image, str]]:
    if lang == "ru":
        try:
            query = translator(query)[0]['translation_text']
        except Exception as e:
            print(f"Ошибка перевода: {e}")

    hits = photo_store.query(query, n_results=200)
    if not hits:
        return [(Image.new("RGB", (300, 100), color=(255, 255, 255)), "No images found.")]

    paths = [hit.metadata["path"] for hit, score in hits]
    if rerank_mode != "standard" and rerankers[rerank_mode]:
        reranked = rerankers[rerank_mode].rerank_images(paths, query)
        hits = [(next(h for h in hits if h[0].metadata["path"] == path)[0], 1 - sim) for path, sim in reranked]

    filtered_hits = [(hit, score) for hit, score in hits if score <= score_threshold]
    if not filtered_hits:
        raise gr.Error("Изображений не найдено.")

    def _load(hit) -> Tuple[Image.Image, str]:
        scale = 0.3
        hit, score = hit if isinstance(hit, (tuple, list)) else (hit, None)
        try:
            with Image.open(hit.metadata["path"]) as img:
                img = img.copy()
                if hasattr(img, '_getexif') and (exif := img._getexif()):
                    orientation_key = _get_rotation_key()
                    if orientation_key in exif:
                        orientation = exif[orientation_key]
                        if orientation == 3:
                            img = img.transpose(Image.ROTATE_180)
                        elif orientation == 6:
                            img = img.transpose(Image.ROTATE_270)
                        elif orientation == 8:
                            img = img.transpose(Image.ROTATE_90)
                img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))
                caption = hit.metadata.get("caption", "No caption available")
                return img, f"Score: {round(score, 2)}\nCaption: {caption}"
        except Exception as e:
            print(f"Ошибка загрузки {hit.metadata['path']}: {e}")
            return Image.new("RGB", (300, 100), color=(255, 255, 255)), "Ошибка загрузки"

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            return list(executor.map(_load, filtered_hits))
    except Exception as e:
        print(f"Ошибка многопоточной загрузки: {e}")
        return [_load(hit) for hit in filtered_hits]

def build_app() -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Soft(), title="Семантический поиск фото") as demo:
        gr.Markdown("# Семантический поиск фотографий\nВведите запрос, чтобы найти релевантные изображения.")
        with gr.Column():
            query_bar = gr.Textbox(lines=1, label="Поисковый запрос", interactive=True)
            search_button = gr.Button("Поиск")
            with gr.Row():
                lang_toggle = gr.Radio(
                    choices=[("🇷🇺 Русский", "ru"), ("🇬🇧 English", "en")],
                    value="ru",
                    label="Язык запроса"
                )
                rerank_mode = gr.Radio(
                    choices=["standard", "fast", "accurate"],
                    value="standard",
                    label="Режим ранжирования"
                )
            score_slider = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=0.5,
                step=0.01,
                label="Порог релевантности"
            )
            gallery = gr.Gallery(
                label="Найденные изображения",
                columns=4,
                object_fit="contain",
                height="75vh",
                interactive=False,
                type=OUTPUT_TYPE
            )

        inputs = [query_bar, score_slider, rerank_mode, lang_toggle]
        search_button.click(fn=search, inputs=inputs, outputs=gallery)
        query_bar.submit(fn=search, inputs=inputs, outputs=gallery)
    return demo


if __name__ == '__main__':
    app = build_app()
    max_retries = 3
    for attempt in range(max_retries):
        try:
            app.queue(max_size=10).launch(server_name="0.0.0.0", server_port=7860)
            break
        except socket.error as e:
            if "too many open files" in str(e):
                print(f"Ошибка: слишком много файлов. Попытка {attempt + 1}/{max_retries}...")
                time.sleep(2)
            else:
                raise e
