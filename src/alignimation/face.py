"""
This module is just a mess. It contains all the hacky shit I did to pick out my face
and facial landmarks. Lots of the code is taken and modified from this facenet_pytorch
example
https://github.com/timesler/facenet-pytorch/blob/master/examples/lfw_evaluate.ipynb
"""

import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import kornia as K
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from IPython import display
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm


class ImageDataset(torch.utils.data.Dataset):
    """
    Dataset corresponding to all the pictures of me. This class looks through
    `root_path` for JPEG images. Each image is expected to be in its own folder, so
    like:
    root_path
    ├── IMG_1
    │   └── pic.jpg
    ├── IMG_2
    │   └── pic.jpg
    └── IMG_3
        └── pic.jpg
    This is because later we'll place a small detections.json file inside each image
    folder where that file contains the bounding boxes and landmarks for that image.
    This class loads that detections.json file if it's there.
    """

    def __init__(self, root_path: Path, transform=None):

        self.root_path = root_path
        self.transform = transform
        self.idx_to_path = {}
        idx = 0
        for path in self.root_path.rglob("*"):
            if path.suffix in (".jpg", ".JPG"):
                self.idx_to_path[idx] = path
                idx += 1

    def __getitem__(self, idx):
        path = self.idx_to_path[idx]
        img = Image.open(self.idx_to_path[idx])
        try:
            # Apparently this is a way to get the timestamp for when the image was
            # created
            date_created = dt.datetime.strptime(
                img._getexif()[36867], "%Y:%m:%d %H:%M:%S"
            )
        except:
            # Sometimes it doesn't work though.
            # Some of my images have a filename that contains the date, so we attempt
            # to do that here
            try:
                # root_path/IMG_YYYYMMDD_HHMMSS/pic.jpg
                date_created = dt.datetime.strptime(
                    path.parent.stem.split("_")[-2], "%Y%m%d"
                )
            except:
                print(f"Cannot find date created for {path.as_posix()}")
                date_created = None

        # The exif_transpose is necessary to correctly rotate some images
        img = exif_transpose(img).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        detection_path = path.parent / "detections.json"
        if detection_path.exists():
            with detection_path.open("r") as f:
                detections = json.load(f)
        else:
            detections = {}
        return (img, path, detections, date_created)

    def __len__(self):
        return len(self.idx_to_path)


def collate_dataset(batch):
    return list(zip(*batch))


def make_detections(root_path: Path):
    """
    Loop through all images in root_path, detect faces with MTCNN, and write the facial
    bounding boxes and landmarks to a detections.json file within each image's
    directory.
    """
    dataset = ImageDataset(root_path)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(device=device, keep_all=True)
    loader = DataLoader(
        dataset,
        num_workers=0,
        # batch_size has to be 1 because the image have different sizes and I don't
        # feel like rescaling them here.
        batch_size=1,
        collate_fn=collate_dataset,
    )
    # Write all of the detections to disk
    # (bounding boxes and landmarks)

    for imgs, paths, batch_detections, _ in tqdm(loader):
        for img, path, detections in zip(imgs, paths, batch_detections):
            detections = {"detections": []}
            _detection = mtcnn.detect(img, landmarks=True)
            if _detection[0] is None:
                print(f"Cannot find faces in {path.as_posix()}")
                continue
            for box, prob, landmark in zip(*_detection):
                detections["detections"].append(
                    {"box": box.tolist(), "landmark": landmark.tolist()}
                )
            with (path.parent / "detections.json").open("w") as f:
                json.dump(detections, f)


def crop_box(
    img: torch.Tensor,
    box: list,
    landmarks: np.ndarray,
    min_side: Optional[int] = None,
    margin: float = 0.0,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Somehow this clusterfuck of a function works.

    In this function, we crop `img` to the rectangle specified by `box`. The box should
    be the detection output from the MTCNN model, and I think its contents are
    something like:
    [height_start, width_start, height_end, width_end]

    We also modify the landmarks so that they are accurate relative to the new cropped
    image.

    We also resize the img so that it has a minimum side length of `min_side`. We also
    resize the landmarks accordingly.

    This function also does a bunch of other things with the other parameters:

    Parameters
    ----------
    img: The image to crop. A tensor of shape (1, C, H, W)
    box: The facial bounding box to use for cropping the img.
    margin: If you want to crop bigger than the box, then the margin is the "extra"
        percentage of the box to crop. So, margin = 0.3 means make the box 30% bigger
        and then crop. Sometimes this causes the box to fall off the edge of the image.
        In that case, we limit the box to the edge of the image. The landmarks are kept
        in sync with the img when the margin is passed.
    min_side: If passed, then ensure that each image has a minimum side length equal to
        min_side. If the image needs to be resized, then the landmarks also get resized.

    """
    # margin is relative to box size.
    # margin = 0.3 means make the box 30% bigger.
    image_size = img.shape[-2:]
    dx = box[3] - box[1]
    dy = box[2] - box[0]

    # Make landmarks relative to box
    # Don't move the landmarks if the box is off the edge.
    if 0 < box[0] < image_size[0]:
        landmarks[:, 0] -= box[0]
    if 0 < box[1] < image_size[1]:
        landmarks[:, 1] -= box[1]

    # Need to handle scenario where box extends outside of the image.
    # In that case, the landmarks no longer are relative to the box edge.

    new_x_start = int(max(box[1] - dx * margin / 2, 0))
    new_x_end = int(min(box[3] + dx * margin / 2, image_size[0]))
    new_y_start = int(max(box[0] - dy * margin / 2, 0))
    new_y_end = int(min(box[2] + dy * margin / 2, image_size[1]))

    img = img[:, new_x_start:new_x_end, new_y_start:new_y_end]

    # Scale landmarks by the margin, but don't scale off the edge of the image.
    # The with 0 handles the case where the box value is negative.
    landmarks[:, 0] += max(min(dy * margin / 2, box[0]), 0)
    landmarks[:, 1] += max(min(dx * margin / 2, box[1]), 0)

    height, width = img.shape[-2:]
    current_min_side = min(height, width)

    if min_side is not None and current_min_side < min_side:
        scale_factor = min_side / current_min_side
        new_height = int(scale_factor * height)
        new_width = int(scale_factor * width)
        img = K.resize(img, (new_height, new_width))
        landmarks *= scale_factor
    return img, landmarks


def make_embeddings(
    root_path: Path, min_side: int = 160, margin: float = 0.2
) -> Dict[Path, List[np.ndarray]]:
    """
    For each image in root_path, generate embeddings for all faces detected in the
    image. The returned object is a dictionary where the key is the path to the image
    and the value is a list of the detected face embeddings.
    """
    trans = torchvision.transforms.Compose(
        [np.float32, torchvision.transforms.ToTensor(), fixed_image_standardization]
    )
    dataset = ImageDataset(root_path, transform=trans)
    loader = DataLoader(
        dataset, num_workers=0, batch_size=1, collate_fn=collate_dataset
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    resnet = InceptionResnetV1(classify=False, pretrained="vggface2").to(device)

    # Construct full embedding matrix
    path_to_detection_embeddings = defaultdict(list)
    resnet.eval()
    with torch.inference_mode():
        for imgs, paths, batch_detections, _ in tqdm(loader):
            for img, path, detections in zip(imgs, paths, batch_detections):
                if detections == {}:
                    continue
                for detection in detections["detections"]:
                    cropped, landmark = crop_box(
                        img,
                        detection["box"],
                        np.array(detection["landmark"]),
                        min_side=min_side,
                        margin=margin,
                    )
                    embeddings = (
                        resnet(cropped.unsqueeze(0).to(device)).to("cpu").numpy()
                    )
                    path_to_detection_embeddings[path].append(embeddings)
    return path_to_detection_embeddings


def distance(embeddings1, embeddings2, distance_metric=0):
    """
    This is straight copied from
    https://github.com/timesler/facenet-pytorch/blob/master/examples/lfw_evaluate.ipynb

    And from there, it's copied from David Sandberg's FaceNet implementation

    It's just Euclidean and Cosine distance.
    """
    if distance_metric == 0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)
    elif distance_metric == 1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / np.pi
    else:
        raise "Undefined distance metric %d" % distance_metric

    return dist


def facing_forward(landmark: np.ndarray, threshold: float = 0.7) -> bool:
    """
    Determine whether or not a person is facing forward (in terms of yaw, not pitch).
    This function relies on the facial landmarks and measures both the ratio of either
    eye to the nose and either lip corner to the nose. Presumably, if one eye is much
    closer to the nose than the other in the image, then you're looking to the side.
    """
    nose = landmark[[2]]
    distances = distance(nose, landmark, 0)
    eyes_ratio = min(distances[:2]) / max(distances[:2])
    lips_ratio = min(distances[3:]) / max(distances[3:])
    is_facing_forward = eyes_ratio > threshold and lips_ratio > threshold
    return is_facing_forward


def make_my_embeddings(
    root_path: Path,
    path_to_detection_embeddings: Dict[Path, List[np.ndarray]],
    sample_pos_examples: bool = True,
    num_pos_samples: int = 10,
) -> np.ndarray:
    """
    This is a weird function to generate 'my' face embedding. The default embedding for
    me is the average of all face embeddings in the dataset. This is a fair default
    because I'm in every picture. For pictures with multiple faces, each face's
    embedding is downweighted by the number of faces in the image.

    If sample_pos_examples = True, then an interactive dialogue pops up (this should be
    done within a jupyter notebook). Picture are randomly selected, and then the face
    in the image with the closest embedding to my average embedding is shown. If the
    shown face is me and a good one to use for generating my embedding, then the user
    indicates that. This random sampling continues until `num_pos_samples` examples of
    my face have been generated. Then, "my embedding" becomes the average embedding
    from these selected faces.

    """
    # Start by making a weighted embedding out of myself
    weighted_embeddings = []
    for detection_embeddings in path_to_detection_embeddings.values():
        weight = 1 / len(detection_embeddings)
        weighted_embeddings += [
            weight * embedding for embedding in detection_embeddings
        ]

    my_embeddings = np.vstack(weighted_embeddings).mean(axis=0)
    if not sample_pos_examples:
        return my_embeddings

    trans = torchvision.transforms.Compose(
        [np.float32, torchvision.transforms.ToTensor()]
    )
    dataset = ImageDataset(root_path, transform=trans)

    # Now, let's sample a bunch of faces and see which ones are me.
    samples = np.arange(len(dataset))
    np.random.shuffle(samples)
    sampled_me_embeddings = []
    num_found = 0
    ctr = 0
    while num_found < num_pos_samples:
        img, path, detections, date_created = dataset[samples[ctr]]
        # Closest face
        detection_embeddings = path_to_detection_embeddings[path]
        dist = distance(np.vstack(detection_embeddings), my_embeddings)
        closest = detections["detections"][dist.argmin()]
        cropped_img, cropped_landmarks = crop_box(
            img, closest["box"], np.array(closest["landmark"]), margin=0.5, min_side=160
        )
        plt.imshow(K.tensor_to_image(cropped_img) / 255.0)
        display.display(plt.gcf())
        display.clear_output(wait=True)
        is_me = input(
            f"({num_found}/{num_pos_samples})Is this a good photo to use of me? y/n"
        )
        if is_me == "y":
            sampled_me_embeddings.append(detection_embeddings[dist.argmin()])
            num_found += 1
        ctr += 1

    print(f"Found {len(sampled_me_embeddings)} examples of me.")
    return np.vstack(sampled_me_embeddings).mean(axis=0)


def pickout_correct_faces(
    root_path: Path,
    path_to_detection_embeddings: Dict[Path, List[np.ndarray]],
    my_embeddings: np.ndarray,
    min_side: int,
    margin: float,
) -> List[Dict[str, Any]]:
    """
    Given a dataset of images in `root_path`, pre-calculated face embeddings for each
    image in `path_to_detection_embeddings` and a vector to use as `my_embedding`, this
    function picks out and crops my face in each image. In the event that I'm not
    facing forward in the image, that image is skipped.

    Noteably, there is no threshold for how close the face embedding in the image should
    be to my embedding. Thus, there are a lot of false negatives (aka other people that
    end up being assumed to be me.)

    The return object is a list with a dict for each kept image. The dict contains the
    image tensor, the date the image was created, my face landmarks, and the path to the
    image. The returned list is sorted in ascending order of when the images were taken.
    """
    trans = torchvision.transforms.Compose(
        [np.float32, torchvision.transforms.ToTensor()]
    )
    dataset = ImageDataset(root_path, transform=trans)

    faces = []
    skips = 0
    for i in tqdm(range(len(dataset))):
        img, path, detections, date_created = dataset[i]
        if detections == {}:
            continue
        dist = distance(np.vstack(path_to_detection_embeddings[path]), my_embeddings)
        my_detection = detections["detections"][dist.argmin()]

        if not facing_forward(np.array(my_detection["landmark"]), threshold=0.7):
            # print(f"Skipping {path.as_posix()}")
            skips += 1
            continue
        cropped, landmark = crop_box(
            img,
            my_detection["box"],
            np.array(my_detection["landmark"]),
            margin=margin,
            min_side=min_side,
        )
        faces.append(
            {
                "img": cropped,
                "date_created": date_created,
                "landmark": landmark,
                "path": path,
            }
        )
    print(f"Skipped {skips}/{len(dataset)} images")
    faces = list(
        sorted(
            (x for x in faces if x["date_created"] is not None),
            key=lambda x: x["date_created"],
        )
    )
    return faces


def resize_and_crop(
    img: torch.Tensor, max_side: int, landmarks: np.ndarray
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Odd function that I don't remember writing that resizes the images so that they are
    squares with sides equal to max_side. The images don't change aspect ratio, though.
    They are just left with black borders for their shorter sides.
    """
    height, width = img.shape[-2:]
    current_max_side = max(height, width)
    height_is_larger_side = height > width

    scaling_factor = max_side / current_max_side

    landmarks[:, 0] *= scaling_factor
    landmarks[:, 1] *= scaling_factor

    new_height = int(height * scaling_factor)
    new_width = int(width * scaling_factor)

    resized = K.resize(img, (new_height, new_width))
    new_img = torch.zeros((3, max_side, max_side), dtype=img.dtype)
    center = max_side / 2
    new_img[
        :,
        int(center - new_height / 2) : int(center + new_height / 2),
        int(center - new_width / 2) : int(center + new_width / 2),
    ] = resized

    if height_is_larger_side:
        # We need to add on sizing to the width
        to_add = (max_side - new_width) / 2
        landmarks[:, 0] += to_add
    else:
        to_add = (max_side - new_height) / 2
        landmarks[:, 1] += to_add

    return new_img, landmarks


def main(
    root_path: Path,
    margin: float = 0.40,
    max_side: int = 1_000,
    sample_pos_examples: bool = False,
    num_pos_examples: int = 10,
) -> List[Tuple[torch.Tensor, np.ndarray]]:
    """
    Entry point to this terrible module. Pick out "my" face in all images contained
    within `root_path`. It's assumed that all images in `root_path` contain pictures of
    me. See the `ImageDataset` documentation for what `root_path` should look like.
    """
    make_detections(root_path)
    # I think this has to be >=160 or something for this function to work.
    min_side = 160
    path_to_detection_embeddings = make_embeddings(
        root_path, min_side=min_side, margin=0.0
    )

    my_embeddings = make_my_embeddings(
        root_path,
        path_to_detection_embeddings,
        sample_pos_examples=sample_pos_examples,
        num_pos_samples=num_pos_examples,
    )
    faces = pickout_correct_faces(
        root_path, path_to_detection_embeddings, my_embeddings, min_side, margin
    )

    fixed_faces = []
    for f in faces:
        img, landmarks = resize_and_crop(
            # We clone because these get modified in place and it was messing me up
            # while debugging things.
            f["img"].clone(),
            max_side,
            f["landmark"].copy(),
        )
        fixed_faces.append((img, landmarks))
    return fixed_faces
