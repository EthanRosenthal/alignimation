import datetime as dt
import shlex
import subprocess
from pathlib import Path
from typing import List, Union

import kornia as K
import torch
import torchvision
from PIL import Image
from PIL.ImageOps import exif_transpose


def _get_image_date(path: Path) -> dt.datetime:
    """
    Get the image date. This assumes a particular schema to the image filenames.
    """
    _, date, _ = path.stem.split("_")
    return dt.datetime.strptime(date, "%Y%m%d")


def _load_image(path: Path) -> torch.Tensor:
    """
    Load an image as a tensor. This function makes sure to load the image in the
    correct orientation.
    """
    to_tensor = torchvision.transforms.ToTensor()
    return to_tensor(exif_transpose(Image.open(path))).unsqueeze_(0)


def load_images(path: Path) -> List[torch.Tensor]:
    """
    Load all images in a directory and sort them by their dates.

    Parameters
    ----------
    path: Directory that contains the images

    Returns
    -------
    images: List of images as tensors, sorted by date.
    """
    images = []
    patterns = ("*.jpg", "*.jpeg")
    for pattern in patterns:
        for img_path in path.rglob(pattern):
            img = _load_image(img_path)
            images.append((img, _get_image_date(img_path)))
    sorted_images, _ = zip(*sorted(images, key=lambda x: x[1]))
    return list(sorted_images)


def save_images(images: Union[torch.Tensor, List[torch.Tensor]], path: Path):
    """
    Save a tensor of images to the path directory. Images are saved with the filename
    pattern img$IDX.jpg where $IDX is the batch index in the images tensor.
    """
    path.mkdir(exist_ok=True)
    for ctr, img in enumerate(images, start=1):
        im = Image.fromarray((K.tensor_to_image(img) * 255.0).astype("uint8"))
        im.save((path / f"img{ctr}.jpg").as_posix())


def images_to_movie(images_path: Path, output: str, framerate: int):
    """
    Convert a series of images in images_path directory into a movie. Image names should
    follow the convention from save_images().

    Parameters
    ----------
    images_path: Directory containing images to turn into a movie
    output: Name of movie to create
    framerate: Number of frames per second that the movie should have.
    """
    command = (
        f"ffmpeg -f image2 -framerate {framerate} -y "
        f"-i {images_path.as_posix()}/img%d.jpg -c:v libx264 -c:a aac "
        f"-movflags +faststart  {output}"
    )
    subprocess.check_call(
        shlex.split(command), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )
