from typing import Any, Dict, List, Optional, Tuple

import kornia as K
import numpy as np
import torch
from kornia.geometry import ImageRegistrator
from torch.nn import functional as F
from torchvision.models.detection import (
    keypointrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn,
)
from tqdm.autonotebook import tqdm

from alignimation.constants import BODY_KEYPOINT_NAMES, COCO_INSTANCE_CATEGORY_NAMES


def _get_top_segmentation_masks(
    batch_segmentation: List[Dict[str, torch.Tensor]], class_index: int
) -> List[torch.Tensor]:
    """
    For a segmentation output by a segmentation model, find the top-scoring mask for
    each image associated with the class_index class.

    Parameters
    ----------
    batch_segmentation: The output from the segmentation model.
    class_index: The class to pick out masks from.

    Returns
    -------
    top_masks: Tensor of segmentation masks of shape (B, C, H, W) where C is a single
        dimension (I think!).
    """
    top_masks = []
    for segmentation in batch_segmentation:
        labels = segmentation["labels"].to("cpu")
        masks = segmentation["masks"].to("cpu")
        scores = segmentation["scores"].to("cpu")
        best_score = -1
        best_mask = None
        for label, mask, score in zip(labels, masks, scores):
            if label == class_index:
                if score > best_score:
                    best_mask = mask
                    best_score = score
        if best_mask is None:
            raise ValueError(f"No mask found for class index {class_index}")
        # Unsqueeze 0 dim because we lost that dimension while iterating through the for
        # loop.
        top_masks.append(best_mask.unsqueeze(0))

    return top_masks


def get_segmentation_masks(
    imgs: torch.Tensor,
    model: torch.nn.Module,
    device: str,
    class_index: int,
    batch_size: int = 8,
) -> torch.Tensor:
    """
    Run the images in batches through a forward pass of the model. Pick out the
    top-scoring mask in each image corresponding to the class_index.

    Parameters
    ----------
    imgs: A tensor of images of size (B, C, H, W)
    model: The model to use to extract segmentation masks
    device: The PyTorch device on which to run the model.
    class_index: The class index to pick out segmentation masks.
    batch_size: Number of images to pass through the model at once.

    Returns
    -------
    imgs_masks: Tensor of segmentation masks of size (B, C, H, W) where there is only
        a single Channel dimension (I think!).
    """
    with torch.inference_mode():
        imgs_masks = []
        model = model.to(device)
        total = int(np.ceil(imgs.shape[0] / batch_size))
        pbar = tqdm(
            enumerate(range(0, imgs.shape[0], batch_size), start=1),
            total=total,
            desc="Segmenting Images",
        )
        for batch_idx, i in pbar:
            pbar.set_description(f"Batch {batch_idx}", refresh=True)
            start = i
            end = min(i + batch_size, imgs.shape[0])
            batch_imgs_segments = model(imgs[start:end].to(device))
            imgs_masks += _get_top_segmentation_masks(batch_imgs_segments, class_index)

    return torch.cat(imgs_masks)


def make_gaussian(
    size: Tuple[int, int], center: Tuple[int, int] = (0, 0), std: float = 1.0
) -> torch.Tensor:
    """
    Make a 2D Gaussian matrix.

    Parameters
    ----------
    size: The (height, width) of the Gaussian matrix
    center: The location of the center of the Gaussian
    std: The standard deviation of the Gaussian

    Returns
    -------
    gaussian: The 2D Gaussian matrix
    """
    grid_x, grid_y = torch.meshgrid(torch.arange(size[0]), torch.arange(size[1]))
    return (
        1
        / (2 * np.pi)
        * torch.exp(
            -1
            / (2 * std ** 2)
            * ((grid_x - center[0]) ** 2 + (grid_y - center[1]) ** 2)
        )
    )


def get_keypoints(
    model: torch.nn.Module,
    imgs: torch.Tensor,
    device: str,
    batch_size: int,
    keypoint_indices: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Get the keypoints associated with a tensor of images.

    Parameters
    ----------
    model: The model to use to extract keypoints
    imgs: A tensor of images of shape (B, C, H, W)
    device: The PyTorch device on which to run the forward passes of the model.
    batch_size: Number of images to process at once with the model.
    keypoint_indices: Indices of the keypoints to extract from the model.

    Returns
    -------
    keypoints: A tensor containing the locations of the keypoints found in each image.
        The tensor has shape (B, K, 2) where K is the keypoint indices.
    """
    with torch.inference_mode():
        keypoints: List[torch.Tensor] = []
        model = model.to(device)
        total = int(np.ceil(imgs.shape[0] / batch_size))
        pbar = tqdm(
            enumerate(range(0, imgs.shape[0], batch_size), start=1),
            total=total,
            desc="Finding Keypoints",
        )
        for batch_idx, i in pbar:
            pbar.set_description(f"Batch {batch_idx}", refresh=True)
            start = i
            end = min(i + batch_size, imgs.shape[0])
            batch_keypoints = model(imgs[start:end].to(device))

            keypoints += [
                keypoint["keypoints"][[0], :, :2].to("cpu")
                for keypoint in batch_keypoints
            ]

        keypoints_tensor = torch.cat(keypoints)
        keypoints_tensor = keypoints_tensor[:, keypoint_indices, :]

    return keypoints_tensor


def make_keypoint_gaussians(
    keypoints: torch.Tensor, size: Tuple[int, int], std: Optional[float] = None
) -> torch.Tensor:
    """
    Create a 2D Gaussian mask centered at each keypoint.

    Parameters
    ----------
    keypoints: A Tensor containing the locations of each keypoint. This is the output
        from get_keypoints().
    size: The height and width of the overall image mask that will be created.
    std: The value for the standard deviation of the Gaussians. If not passed, then a
        heuristic value of 0.04 * the maximum side will be used.

    """
    std = std or max(size) * 0.04
    gaussians = []
    for img_keypoints in keypoints:
        these_gaussians = []
        for keypoint in img_keypoints:
            # We have to reverse the keypoints because I think the gaussians expect
            # transposed images or something.
            xc, yc = keypoint.squeeze().tolist()[::-1]
            gaussian = make_gaussian(size, center=(xc, yc), std=std).unsqueeze(0)
            these_gaussians.append(gaussian)
        gaussians.append(torch.cat(these_gaussians).unsqueeze(0))
    return torch.cat(gaussians)


def _scale_to_template(
    template: torch.Tensor, imgs: List[torch.Tensor], max_side: Optional[int] = None
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Make sure all imgs have the same size as the template and optionally rescale
    everything.

    Parameters
    ----------
    template: The template image. All imgs will be resized to have the same size as
        the template.
    imgs: A list of images to resize.
    max_side: If passed, then rescale all images such that the larger side of
        the image has size=max_side.

    Returns
    -------
    template: The original template. It will be rescaled if max_side was passed.
    imgs: The original list of images, resized and optionally rescaled.
    """
    # Make sure everything has the same size as the template.
    height, width = template.shape[-2:]
    imgs = [K.resize(img, (height, width)) for img in imgs]

    # Batch up the imgs into a single Tensor
    imgs_tensor = torch.cat(imgs, dim=0)
    # Now, scale everything down to make it easier to work with
    scaling_factor = None
    if max_side:
        current_max_side = max(height, width)
        scaling_factor = max_side / current_max_side
        template = K.rescale(template, scaling_factor)
        imgs_tensor = K.rescale(imgs_tensor, scaling_factor)

    return template, [img.unsqueeze(0) for img in list(imgs_tensor)]


def alignimate(
    template: torch.Tensor,
    imgs: List[torch.Tensor],
    max_side: Optional[int] = None,
    batch_size: int = 8,
    add_segmentation_mask: bool = True,
    add_keypoint_mask: bool = False,
    template_keypoints: Optional[torch.Tensor] = None,
    imgs_keypoints: Optional[torch.Tensor] = None,
    keypoint_std: Optional[float] = None,
    registrator_kwargs: Dict[str, Any] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    This is the main entry point to the code. Fair warning: this function is kind of a
    mess! This function takes in a template image tensor and a list of image tensors
    and aligns all images with the template. The alignment process is as follows:

    1. Resize all images to match the template size.
    2. Optionally rescale everything.
    3. Run a segmentation model on the rescaled images to pick out the person in the
      image and create a segmentation mask.
    4. Optionally pick out keypoints from the rescaled images, create a gaussian mask
      around each keypoint and multiply this by the segmentation mask.
    5. Fit an image registration model to register each image mask with the template
      mask.
    6. Use each image registration model to warp the originally-sized input images so
      that they're aligned with the template.

    Parameters
    ----------
    template: The template image tensor to align all images with
    imgs: A list of image tensors to be aligned with the template
    max_side: If passed, then rescale all images such that the larger side of
        the image has size=max_side.
    batch_size: Number of images to process at once for segmentation and keypoint
        models.
    add_segmentation_mask: Whether or not to use a segmentation model to create a mask
        corresponding to the body in the image.
    add_keypoint_mask: Whether or not to add a mask of gaussians around keypoints to
        the segmentation mask for use in registration. The keypoints are hardcoded
        right now to just be the left ear, left shoulder, and left hip.
    template_keypoints: Optional keypoints to use for the template image rather than
        detecting the keypoints in this function.
    imgs_keypoints: Optional keypoints to use for the images to register rather than
        detecting the keypoints in this function.
    keypoint_std: The value for the standard deviation of the keypoint Gaussians. If
        not passed, then a heuristic value of 0.04 * the maximum side will be used.
    registrator_kwargs: Any keyword arguments you want to pass into the kornia
        ImageRegistrator. If nothing is passed here, then the following kwargs that
        seem to result in good performance are passed:
        {"loss_fn": F.mse_loss, "tolerance": 1e-8}

    Returns
    -------
    aligned_imgs: List of all images aligned with the template image. The template
        image is the first image in the list. Each image is a tensor of size
        (1, C, H, W).
    aligned_masks: List of all masks aligned with the template. The template mask is
        the first mask in the tensor. If max_side was passed, then the masks
        will be a different size than the aligned_imgs. Each mask is a tensor of size
        (1, K, H, W) where K is the number of keypoints (and 1 if no keypoints were
        used).
    """
    registrator_kwargs = registrator_kwargs or {
        "loss_fn": F.mse_loss,
        "tolerance": 1e-8,
    }
    if template_keypoints is not None and imgs_keypoints is not None:
        keypoints_provided = True
        if max_side is not None:
            raise ValueError("Cannot scale to `max_side` when keypoints are passed in.")
    elif template_keypoints is None and imgs_keypoints is None:
        keypoints_provided = False
    else:
        raise ValueError(
            "template_keypoints and imgs_keypoints must both be None or both not be "
            "None."
        )
    # First, scale all imgs so that they have the same size as the template. We
    # purposefully do not pass max_side in.
    template, imgs = _scale_to_template(template, imgs, max_side=None)
    # Copy these so that we can use them later.
    original_template = template.clone()
    original_imgs = [img.clone() for img in imgs]

    # Now, rescale imgs and template down to max_side
    template, imgs = _scale_to_template(template, imgs, max_side=max_side)
    height, width = template.shape[-2:]

    # Stack together so that later operations happen equivalently on both the template
    # and the imgs.
    imgs_tensor = torch.cat([template] + imgs)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    masks = torch.ones_like(imgs_tensor)

    if add_segmentation_mask:
        # Run image segmentation mask to pick out body masks.
        # TODO: Allow user to pass in a segmentation model and the segmentation class_index.
        segmentation_model = maskrcnn_resnet50_fpn(pretrained=True, progress=False)
        segmentation_model = segmentation_model.eval()
        person_index = COCO_INSTANCE_CATEGORY_NAMES.index("person")
        masks = get_segmentation_masks(
            imgs_tensor, segmentation_model, device, person_index, batch_size=batch_size
        )

    if add_keypoint_mask or keypoints_provided:
        if keypoints_provided:
            # We know these are not None, but we have to do this to satisfy the mypy
            # gods
            assert template_keypoints is not None
            assert imgs_keypoints is not None
            keypoints = torch.cat([template_keypoints, imgs_keypoints])
        elif not keypoints_provided:
            # TODO: Allow user to pass in a keypoint model and specify the keypoint_names.

            # Get keypoints
            keypoint_model = keypointrcnn_resnet50_fpn(pretrained=True, progress=False)
            keypoint_model = keypoint_model.eval()
            keypoint_names = ["left_ear", "left_shoulder", "left_hip"]
            keypoint_indices = [BODY_KEYPOINT_NAMES.index(k) for k in keypoint_names]
            keypoints = get_keypoints(
                keypoint_model, imgs_tensor, device, batch_size, keypoint_indices
            )

        gaussians = make_keypoint_gaussians(
            keypoints, (height, width), std=keypoint_std
        )
        # Multiply the segmentation mask with the gaussian masks. Each Gaussian gets
        # its own channel dimension. This way, when we run image registration, we
        # explicitly match the same gaussians for two different images (e.g. the left
        # shoulder of one image gets registered to the left shoulder of the other
        # image.)
        masks = masks * gaussians

    template_mask = masks[[0]]

    aligned_masks = [template_mask]
    aligned_imgs = [original_template]

    template_mask = template_mask.to(device)

    registrator = ImageRegistrator("similarity", **registrator_kwargs).to(device)

    # For each mask and image, fit a registrator on the template and image masks and
    # then use the registrator to warp the original-sized image.
    for mask, original_img in tqdm(zip(masks[1:], original_imgs), total=len(imgs) - 1):
        # We lose the batch dimension when we iterate through the masks,
        # so need to unsqueeze
        mask = mask.to(device).unsqueeze(0)

        registrator.register(mask, template_mask)
        with torch.inference_mode():
            original_img = original_img.to(device)

            alignment = registrator.warp_src_into_dst(original_img)
            aligned_imgs.append(alignment.to("cpu"))

            aligned_mask = registrator.warp_src_into_dst(mask)
            aligned_masks.append(aligned_mask.to("cpu"))

    return aligned_imgs, aligned_masks


def subtract_background(imgs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    Subtract the background from a tensor of images by using the masks to just pick out
    the parts of the images to keep.
    """
    background_sub = torch.ones_like(imgs)
    # masks might be smaller than imgs, so rescale it to match. Then, broadcast it along
    # the channel dimension.
    thresholded_mask = (K.resize(masks, imgs.shape[-2:]) > 0.5).expand(-1, 3, -1, -1)

    background_sub[thresholded_mask] = imgs[thresholded_mask]
    return background_sub


def get_start_end(imgs: torch.Tensor, dim: int) -> Tuple[int, int]:
    vals, _ = (
        (
            #            batch_dim, channel_dim
            ((imgs == 0).any(dim=0).any(dim=0).float().diff(dim=dim) < 0.0).any(dim=dim)
        )
        .diff()
        .nonzero()
        .squeeze()
        .sort()
    )
    if len(vals) == 1:
        # Is it a start or an end
        val = vals[0]
        size = imgs.shape[-2:][dim]
        if val >= int(size / 2):
            # It's the end.
            end = val
            start = 0
        elif val < int(size / 2):
            # It's the start
            start = val
            end = size
    elif len(vals) == 2:
        start, end = vals
    else:
        raise ValueError(f"Something's wrong with vals: {vals}")
    start, end = start.item(), end.item()
    # Make sure we have an even cropping
    if start % 2 == 1:
        start += 1
    if end % 2 == 1:
        end -= 1
    return start, end


def crop_borders(imgs: torch.Tensor) -> torch.Tensor:
    x_start, x_end = get_start_end(imgs, 0)
    y_start, y_end = get_start_end(imgs, 1)
    return imgs[:, :, y_start:y_end, x_start:x_end]
