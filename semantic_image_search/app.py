# Приложение для семантического поиска фотографий
# Позволяет пользователям искать фотографии по текстовым запросам, используя векторную базу данных изображений.
# Поддерживает перевод запросов с русского на английский и улучшение результатов поиска с помощью reranker.

from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
from functools import cache
import warnings
import argparse
import os
import time
import socket

from PIL import Image, ExifTags
import gradio as gr

from semantic_image_search.models.documents import ImageVectorStore
from semantic_image_search.models.rerank import Reranker

try:
    from googletrans import Translator
    translator = Translator()
except ImportError:
    warnings.warn("googletrans is not installed, translation will be skipped")
    translator = None

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
    raise ImportError("Для поддержки HEIC изображений установите 'pillow-heif': pip install pillow-heif")

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--chroma_path", type=str, help="Override the path to the ChromaDB database", required=False)
args = parser.parse_args()

chroma_path = args.chroma_path if args.chroma_path is not None else os.getenv("MODEL_CACHE_DIR")
photo_store = ImageVectorStore(chroma_persist_path=chroma_path)
print(f"Количество изображений в базе: {len(photo_store)}")
OUTPUT_TYPE = "pil"

@cache
def _get_rotation_key() -> int:
    return max(ExifTags.TAGS.items(), key=lambda x: x[1] == 'Orientation', default=(-1, None))[0]

def search(query: str, score_threshold: float, rerank_mode: str, lang: str) -> List[Tuple[Image.Image, str]]:
    if lang == "ru" and translator is not None:
        try:
            query = translator.translate(query, src="ru", dest="en").text
        except Exception as e:
            print(f"Ошибка перевода: {e}")
            query = query  # Используем исходный запрос

    hits = photo_store.query(query, n_results=200)  # количество результатов

    paths = [hit.metadata["path"] for hit, score in hits]
    scores = [score for _, score in hits]

    if rerank_mode != "standard" and rerankers[rerank_mode] is not None:
        reranker = rerankers[rerank_mode]
        reranked = reranker.rerank_images(paths, query)
        hits = [(next(h for h in hits if h[0].metadata["path"] == path)[0], 1 - sim) for path, sim in reranked]

    filtered_hits = [(hit, score) for hit, score in hits if score <= score_threshold]
    if not filtered_hits:
        raise gr.Error("Изображений не найдено. Попробуйте изменить поисковый запрос или понизить точность.")

    def _load(hit):
        scale = 0.3
        score = None
        if isinstance(hit, (tuple, list)):
            hit, score = hit
        try:
            with Image.open(os.path.join(hit.metadata["path"])) as img:
                img = img.copy()  # Create a copy to ensure the file is closed
                if hasattr(img, '_getexif'):
                    try:
                        orientation_key = _get_rotation_key()
                        e = img._getexif()
                        if e is not None and orientation_key in e:
                            if e[orientation_key] == 3:
                                img = img.transpose(Image.ROTATE_180)
                            elif e[orientation_key] == 6:
                                img = img.transpose(Image.ROTATE_270)
                            elif e[orientation_key] == 8:
                                img = img.transpose(Image.ROTATE_90)
                    except Exception as e:
                        print(f"Ошибка обработки EXIF: {e}")

                img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))
        except Exception as e:
            print(f"Ошибка загрузки изображения {hit.metadata['path']}: {e}")
            return Image.new("RGB", (300, 100), color=(255, 255, 255)), "Ошибка загрузки изображения"

        if isinstance(score, (float, int)):
            return img, f"Score: {round(score, 2)}"
        return img, None

    if not hits:
        return [(Image.new("RGB", (300, 100), color=(255, 255, 255)), "No images found. Try changing the query or reducing accuracy.")]

    if OUTPUT_TYPE == "filepath":
        return [(hit.metadata["path"], f"Score: {round(score, 2)}") for hit, score in hits]
    else:
        try:
            with ThreadPoolExecutor(max_workers=4) as executor:  # Limit to 4 workers
                return list(executor.map(_load, hits))
        except Exception as e:
            print(f"Ошибка многопоточной загрузки: {e}")
            return [_load(hit) for hit in hits]

LANG_TEXT = {
    "en": {
        "title": "Semantic photos search",
        "instruction": "Run a query to see relevant photos with the relevance score (lower scores are better).",
        "search": "Search",
        "language": "Language",
        "filter": "Relevance Score Filter",
        "mode": "Search Mode"
    },
    "ru": {
        "title": "Семантический поиск фото",
        "instruction": "Введите запрос, чтобы увидеть подходящие изображения (чем ниже рейтинг, тем лучше)",
        "search": "Поиск",
        "language": "Язык",
        "filter": "Фильтр по точности",
        "mode": "Режим поиска"
    }
}

def build_app() -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Soft(), title="Semantic photo search") as demo:
        gr.Markdown("""
        <h1>Semantic photos search</h1>
        Run a query to see relevant photos with the relevance score (lower scores are better).
        """)

        with gr.Column():
            with gr.Row():
                query_bar = gr.Textbox(lines=1, label="Search", interactive=True)

            search_button = gr.Button("Search")

            with gr.Row():
                lang_toggle = gr.Radio(["🇷🇺 Русский", "🇬🇧 English"], value="🇷🇺 Русский", label="Language")
                rerank_mode = gr.Radio(
                    choices=["standard", "fast", "accurate"],
                    value="standard",
                    label="Search mode (accuracy vs speed)"
                )

            score_slider = gr.Slider(minimum=0.0, maximum=2.0, value=1.8, step=0.01, label="Relevance Score Filter")

            gallery = gr.Gallery(
                label="Photo hits",
                show_label=True,
                columns=[4],
                object_fit="contain",
                height="75vh",
                interactive=False,
                type=OUTPUT_TYPE
            )

        # inputs = [query_bar, score_slider, rerank_mode]
        # query_bar.submit(fn=search, inputs=inputs, outputs=gallery)
        # search_button.click(fn=search, inputs=inputs, outputs=gallery)
        search_button.click(fn=search, inputs=[query_bar, score_slider, rerank_mode, lang_toggle], outputs=[gallery])
        query_bar.submit(fn=search, inputs=[query_bar, score_slider, rerank_mode, lang_toggle], outputs=[gallery])

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
                print(f"Ошибка: слишком много открытых файлов. Повторная попытка {attempt + 1}/{max_retries}...")
                time.sleep(2)
            else:
                raise e
