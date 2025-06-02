import os
from typing import Dict, Iterator, Any
from datetime import datetime
from semantic_image_search.galleries.database import Media, SqliteReaderBase

class WindowsPhotosReader(SqliteReaderBase):
    def __init__(self, photolibrary_path: str):
        self.photo_dir = os.path.expanduser(photolibrary_path)
        if not os.path.exists(self.photo_dir):
            raise FileNotFoundError(f"Directory not found: {self.photo_dir}")

    @property
    def albums(self) -> Dict[str, Dict[str, Any]]:
        output = {}
        try:
            for root, dirs, files in os.walk(self.photo_dir):
                count = sum(1 for f in files if f.split('.')[-1].lower() in self.ALLOWED_TYPES)
                if count == 0:
                    continue
                rel = os.path.relpath(root, self.photo_dir)
                output[rel] = {
                    "album_id": hash(rel),
                    "name": rel.replace(os.sep, " :: "),
                    "path": root,
                    "count": count
                }
        except Exception as e:
            print(f"Ошибка при сканировании директории {self.photo_dir}: {e}")
            return {}
        return output

    def stream_media_from_album(self, album_name: str) -> Iterator[Media]:
        album_path = os.path.join(self.photo_dir, album_name)
        if not os.path.exists(album_path):
            print(f"Альбом '{album_name}' не найден в {self.photo_dir}")
            return

        try:
            for root, dirs, files in os.walk(album_path):
                for f in files:
                    if f.split('.')[-1].lower() in self.ALLOWED_TYPES:
                        full_path = os.path.join(root, f)
                        yield Media(
                            album_id=hash(album_name),
                            image_id=hash(full_path),
                            image_file_name=f,
                            relative_path=root,
                            creation_date=datetime.fromtimestamp(os.path.getctime(full_path)),
                            lat=None,
                            lon=None,
                            people_names=None
                        )
        except Exception as e:
            print(f"Ошибка при обработке альбома '{album_name}': {e}")
            return
