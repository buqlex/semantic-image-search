import os
os.environ["HF_HOME"] = os.getenv("MODEL_CACHE_DIR", os.path.expanduser("~/.cache/huggingface"))

from typing import List, Tuple, Iterator
import argparse
import warnings

from tqdm import tqdm
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC = True
except ImportError:
    HEIC = False
    warnings.warn("pillow-heif not installed, HEIC images skipped. Install: pip install pillow-heif")

from semantic_image_search.galleries.database import DigikamReader, MacPhotosReader, Media
from semantic_image_search.geographies import GeonamesReverseGeocoder
from semantic_image_search.models.caption import ImageCaption
from semantic_image_search.models.documents import ImageVectorStore
from semantic_image_search.models.schema import ImageData
from semantic_image_search.models.utils import get_accelerator
from semantic_image_search.utils import describe_people_in_scene
from semantic_image_search.constants import Supported
from semantic_image_search.galleries.windows_reader import WindowsPhotosReader
from semantic_image_search.config import CHROMA_PATH, BATCH_SIZE, MODEL_NAME, COLLECTION_NAME, LIBRARY_TYPE, PHOTO_LIB_PATH, GEONAMES_USER, ALBUMS

def batch_caption(images: List[ImageData], captioner: ImageCaption) -> List[ImageData]:
    try:
        captions = captioner.caption([img.path for img in images])
        for img, caption in zip(images, captions):
            img.caption = caption
        print(f"Сгенерировано описаний: {len(captions)}")
    except Exception as e:
        print(f"Ошибка генерации описаний: {e}")
    return images

def generate_people_in_scene_descriptions(image: ImageData, metadata: Media) -> ImageData:
    if metadata.people_names:
        image.people_description = describe_people_in_scene(metadata.people_names.split(','))
    return image

def stream_digikam_albums(photo_library_dir: str, albums: List[str]) -> Iterator[Tuple[ImageData, Media]]:
    with DigikamReader(photolibrary_path=photo_library_dir) as db:
        album_map = db.albums
        for album in albums:
            for record in tqdm(db.stream_media_from_album(album_id=album_map[album]["album_id"]), total=album_map[album]["count"], desc=f"Loading {album}"):
                if record.image_file_name.lower().endswith('.heic') and not HEIC:
                    continue
                meta = album_map[record.relative_path]
                yield ImageData(path=os.path.join(meta["path"], record.image_file_name), album_name=meta["name"], file_name=record.image_file_name, created=record.creation_date), record

def stream_macos_albums(photo_library_dir: str, albums: List[str]) -> Iterator[Tuple[ImageData, Media]]:
    with MacPhotosReader(photolibrary_path=photo_library_dir) as db:
        album_map = db.albums
        for album in albums:
            for record in tqdm(db.stream_media_from_album(album_id=album_map[album]["album_id"]), total=album_map[album]["count"], desc=f"Loading {album}"):
                if record.image_file_name.lower().endswith('.heic') and not HEIC:
                    continue
                yield ImageData(path=os.path.join(record.relative_path, record.image_file_name), album_name=album, file_name=record.image_file_name, created=record.creation_date), record

def stream_windows_albums(photo_library_dir: str, albums: List[str]) -> Iterator[Tuple[ImageData, Media]]:
    reader = WindowsPhotosReader(photolibrary_path=photo_library_dir)
    for album in albums:
        for record in reader.stream_media_from_album(album_name=str(album)):
            if record.image_file_name.lower().endswith('.heic') and not HEIC:
                continue
            yield ImageData(path=os.path.join(record.relative_path, record.image_file_name), album_name=album, file_name=record.image_file_name, created=record.creation_date), record

def validate_albums(library_type: Supported, library_dir: str):
    albums = None
    if library_type == Supported.DIGIKAM_PHOTO_LIBRARY:
        with DigikamReader(photolibrary_path=library_dir) as db:
            albums = db.albums
    elif library_type == Supported.MACOS_PHOTO_LIBRARY:
        with MacPhotosReader(photolibrary_path=library_dir) as db:
            albums = db.albums
    elif library_type == Supported.WINDOWS_PHOTO_LIBRARY:
        with WindowsPhotosReader(photolibrary_path=library_dir) as db:
            albums = db.albums
    return albums

def build(library_type: Supported, library_dir: str, chroma_path: str, albums: List[str], geonames_user: str) -> int:
    streamers = {
        Supported.DIGIKAM_PHOTO_LIBRARY: stream_digikam_albums,
        Supported.MACOS_PHOTO_LIBRARY: stream_macos_albums,
        Supported.WINDOWS_PHOTO_LIBRARY: stream_windows_albums
    }
    if library_type not in streamers:
        raise TypeError(f"{library_type.value} не поддерживается")

    device = get_accelerator()
    captioner = ImageCaption(model="Salesforce/blip-image-captioning-large", device=device, batch_size=16)
    vector_store = ImageVectorStore(chroma_persist_path=chroma_path, collection_name=COLLECTION_NAME, model_name=MODEL_NAME, model_kwargs={"device": device})

    image_batch = []
    for image, metadata in streamers[library_type](photo_library_dir=library_dir, albums=albums):
        print(f"Обрабатывается: {image.path}")
        image = generate_people_in_scene_descriptions(image, metadata)
        image_batch.append(image)
        if len(image_batch) >= BATCH_SIZE:
            image_batch = batch_caption(image_batch, captioner)
            vector_store.add_images(image_batch)
            image_batch.clear()

    if image_batch:
        print(f"Обработка пакета из {len(image_batch)} изображений")
        image_batch = batch_caption(image_batch, captioner)
        vector_store.add_images(image_batch)
        print(f"Добавлено: {len(vector_store)}")
    print(f"Завершено. Всего: {len(vector_store)}")
    return len(vector_store)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--geonames_user", type=str, default=GEONAMES_USER, help="Username for Geonames API")
    parser.add_argument("--type", type=Supported.argparse, choices=list(Supported), default=LIBRARY_TYPE, help="Photo library type")
    parser.add_argument("--photo_lib_path", type=str, default=PHOTO_LIB_PATH, help="Path to photo library")
    parser.add_argument("--chroma_path", type=str, default=CHROMA_PATH, help="Path to ChromaDB")
    parser.add_argument("--album", action="append", default=ALBUMS, help="Album name")
    args = parser.parse_args()

    if not args.album:
        available_albums = validate_albums(args.type, args.photo_lib_path)
        if not available_albums:
            raise TypeError(f"`{args.type}` не поддерживается")
        available = '\n ** '.join(k for k, v in available_albums.items() if v["count"] > 0)
        raise AttributeError(f"Альбомы не указаны. Доступные: \n ** {available}")

    print("Начало сборки...")
    build(
        library_type=args.type,
        library_dir=args.photo_lib_path,
        chroma_path=args.chroma_path,
        albums=args.album,
        geonames_user=args.geonames_user
    )
