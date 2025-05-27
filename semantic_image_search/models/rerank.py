from typing import List, Tuple
import torch
import clip
from PIL import Image


class Reranker:
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def rerank_images(self, images: List[str], query: str, top_k: int = None) -> List[Tuple[str, float]]:
        # Загрузка и предобработка изображений
        image_tensors = [self.preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(self.device) for path in images]
        images_tensor = torch.cat(image_tensors)

        # Текстовая обработка
        text_tokens = clip.tokenize([query]).to(self.device)

        # Вычисление эмбеддингов
        with torch.no_grad():
            image_features = self.model.encode_image(images_tensor)
            text_features = self.model.encode_text(text_tokens)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarities = (image_features @ text_features.T).squeeze().tolist()

        results = list(zip(images, similarities))
        results.sort(key=lambda x: -x[1])  # Чем выше, тем лучше (cosine sim)

        return results[:top_k] if top_k else results
