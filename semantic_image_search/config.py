# Конфигурация для семантического поиска изображений
from semantic_image_search.constants import Supported

# Путь к базе данных ChromaDB
CHROMA_PATH = "F:\\Education\\Diploma\\semantic-image-search\\chroma_db"

# Размер пакета для обработки изображений
BATCH_SIZE = 256

# Модель для векторизации текста
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Название коллекции в ChromaDB
COLLECTION_NAME = "semantic-image-search"

# Тип библиотеки по умолчанию
LIBRARY_TYPE = Supported.WINDOWS_PHOTO_LIBRARY

# Путь к библиотеке фотографий по умолчанию
PHOTO_LIB_PATH = "C:\\Users\\buqlex\\Pictures"

# Имя пользователя для Geonames API
GEONAMES_USER = "buqlex"

# Альбомы по умолчанию
ALBUMS = ["Screenshots"]