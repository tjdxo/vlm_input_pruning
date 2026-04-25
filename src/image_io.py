from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps


def load_image(path: str | Path) -> Image.Image:
    image = Image.open(Path(path))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def save_image(image: Image.Image, path: str | Path, quality: int = 95) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(out_path, quality=quality, optimize=True)
    else:
        image.save(out_path)
    return out_path


def safe_stem(path: str | Path) -> str:
    return Path(path).stem.replace(" ", "_")
