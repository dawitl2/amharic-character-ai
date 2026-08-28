"""Shared image preparation for CNN training, evaluation, and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class PreprocessingSpec:
    width: int = 64
    height: int = 64
    channels: int = 1
    normalization_mean: tuple[float, ...] = (0.0,)
    normalization_std: tuple[float, ...] = (1.0,)

    def to_metadata(self) -> dict:
        metadata = asdict(self)
        metadata["normalization_mean"] = list(self.normalization_mean)
        metadata["normalization_std"] = list(self.normalization_std)
        metadata["mode"] = "grayscale"
        metadata["value_range"] = [0.0, 1.0]
        return metadata

    @classmethod
    def from_metadata(cls, metadata: dict) -> "PreprocessingSpec":
        preprocessing = metadata.get("preprocessing", metadata)
        return cls(
            width=int(preprocessing.get("width", metadata.get("image_width", 64))),
            height=int(preprocessing.get("height", metadata.get("image_height", 64))),
            channels=int(preprocessing.get("channels", metadata.get("channels", 1))),
            normalization_mean=tuple(preprocessing.get("normalization_mean", [0.0])),
            normalization_std=tuple(preprocessing.get("normalization_std", [1.0])),
        )


def fit_character_to_canvas(
    image: Image.Image,
    *,
    canvas_size: tuple[int, int],
    foreground_fraction: float = 0.55,
) -> Image.Image:
    """Crop and center content at the scale learned from training canvases."""
    grayscale = image.convert("L")
    pixels = np.asarray(grayscale, dtype=np.uint8)
    border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
    background = int(np.median(border))
    foreground = np.abs(pixels.astype(np.int16) - background) >= 12

    if not foreground.any():
        return grayscale.resize(canvas_size, Image.Resampling.LANCZOS)

    ys, xs = np.nonzero(foreground)
    crop = grayscale.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    target_width = max(1, round(canvas_size[0] * foreground_fraction))
    target_height = max(1, round(canvas_size[1] * foreground_fraction))
    scale = min(target_width / crop.width, target_height / crop.height)
    resized = crop.resize(
        (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("L", canvas_size, color=background)
    offset = ((canvas.width - resized.width) // 2, (canvas.height - resized.height) // 2)
    canvas.paste(resized, offset)
    return canvas


def preprocess_image(
    image: Image.Image,
    spec: PreprocessingSpec,
    *,
    prepare_external: bool = False,
) -> torch.Tensor:
    """Return one normalized ``[1, height, width]`` CNN input tensor."""
    if spec.channels != 1:
        raise ValueError(f"CharacterCNN requires one grayscale channel, got {spec.channels}.")

    grayscale = image.convert("L")
    size = (spec.width, spec.height)
    prepared = (
        fit_character_to_canvas(grayscale, canvas_size=size)
        if prepare_external
        else grayscale.resize(size, Image.Resampling.LANCZOS)
    )
    array = np.asarray(prepared, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0)
    mean = torch.tensor(spec.normalization_mean, dtype=tensor.dtype).view(-1, 1, 1)
    std = torch.tensor(spec.normalization_std, dtype=tensor.dtype).view(-1, 1, 1)
    if torch.any(std == 0):
        raise ValueError("Normalization standard deviation cannot be zero.")
    return (tensor - mean) / std


class CharacterTransform:
    """Pickle-safe dataset transform backed by the shared preprocessing function."""

    def __init__(self, spec: PreprocessingSpec):
        self.spec = spec

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return preprocess_image(image, self.spec, prepare_external=False)
