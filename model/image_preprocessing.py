"""HE/H&E image preprocessing adapters following COSIE.

This module supports both COSIE HE paths:
1. Existing UNI features in an h5ad ``.obsm`` field.
2. Raw HE image + mask -> UNI feature extraction -> HE AnnData.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Tuple

import anndata as ad
import numpy as np
from PIL import Image
from skimage.transform import rescale

from .configure import UNI_DIR
from .utils import ensure_dir, load_h5ad_if_needed

try:
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - only used when torch is unavailable.
    Dataset = object

Image.MAX_IMAGE_PIXELS = None


def save_pickle(x, filename):
    """Save a pickle object to an explicit path."""

    ensure_dir(filename)
    with open(filename, "wb") as file:
        pickle.dump(x, file)
    print(filename)


def load_pickle(filename, verbose=True):
    """Load a pickle object from disk."""

    with open(filename, "rb") as file:
        x = pickle.load(file)
    if verbose:
        print(f"Pickle loaded from {filename}")
    return x


# Adapted from /home/hujinlan/cosie/COSIE/image_preprocessing.py::load_image
def load_image(filename, verbose=True):
    """Load an image file into a NumPy array, dropping alpha if present."""

    print("loading image...")
    Image.MAX_IMAGE_PIXELS = 2**40
    img = Image.open(filename)
    img = np.array(img)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    if verbose:
        print(f"Image loaded from {filename}")
    return img


# Adapted from /home/hujinlan/cosie/COSIE/image_preprocessing.py::rescale_image
def rescale_image(img, scale):
    """Rescale an image using COSIE's preserve-range behavior."""

    if img.ndim == 2:
        scale = [scale, scale]
    elif img.ndim == 3:
        scale = [scale, scale, 1]
    else:
        raise ValueError("Unrecognized image ndim")
    img = rescale(img, scale, preserve_range=True)
    img = np.round(img)
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    return img


# Adapted from /home/hujinlan/cosie/COSIE/image_preprocessing.py::get_white_superpixel_centers
def get_white_superpixel_centers(image_path, superpixel_size=16):
    """Return centers of fully white square superpixels in a binary mask."""

    img = Image.open(image_path).convert("L")
    mask = np.array(img)

    height, width = mask.shape
    centers = []

    for i in range(0, height, superpixel_size):
        for j in range(0, width, superpixel_size):
            patch = mask[i : i + superpixel_size, j : j + superpixel_size]

            if patch.shape == (superpixel_size, superpixel_size) and np.all(patch == 255):
                center_y = i + superpixel_size // 2
                center_x = j + superpixel_size // 2
                centers.append((center_x, center_y))

    return centers


# Adapted from /home/hujinlan/cosie/COSIE/image_preprocessing.py::PatchDataset
class PatchDataset(Dataset):
    """Extract 224x224 image patches centered at COSIE pixel coordinates."""

    def __init__(self, image, location):
        from torchvision import transforms

        self.image = image
        self.location = location
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

        self.shape_ori = np.array(image.shape[:2])
        print("shape_ori:", self.shape_ori)
        self.total_patches = self.location.shape[0]

    def __len__(self):
        return self.total_patches

    def __getitem__(self, idx):
        center_i = self.location[idx, 0]
        center_j = self.location[idx, 1]

        start_i, start_j = max(0, center_i - 112), max(0, center_j - 112)
        end_i, end_j = min(self.shape_ori[0], center_i + 112), min(self.shape_ori[1], center_j + 112)

        patch = self.image[start_i:end_i, start_j:end_j]

        if patch.shape[0] < 224 or patch.shape[1] < 224:
            padded_patch = np.zeros((224, 224, 3), dtype=patch.dtype)
            padded_patch[
                (224 - patch.shape[0]) // 2 : (224 - patch.shape[0]) // 2 + patch.shape[0],
                (224 - patch.shape[1]) // 2 : (224 - patch.shape[1]) // 2 + patch.shape[1],
            ] = patch
            patch = padded_patch

        patch = Image.fromarray(patch.astype("uint8")).convert("RGB")
        return self.transform(patch), (center_i, center_j)


# Adapted from /home/hujinlan/cosie/COSIE/image_preprocessing.py::create_model
def create_model(local_dir):
    """Create the UNI ViT-L/16 model and load ``pytorch_model.bin``."""

    import timm
    import torch

    checkpoint_path = os.path.join(local_dir, "pytorch_model.bin")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"UNI checkpoint not found: {checkpoint_path}")

    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        global_pool="",
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)
    return model


# Adapted from /home/hujinlan/cosie/COSIE/image_preprocessing.py::extract_features
def extract_features(model, batch) -> Tuple[object, object]:
    """Extract COSIE global and local UNI features from an image batch."""

    import torch

    with torch.inference_mode():
        feature_emb = model(batch)
        final_output, _ = model.forward_intermediates(batch, return_prefix_tokens=False)
        local_emb = final_output[:, 1:]
        patch_emb = local_emb.permute(0, 2, 1).reshape(batch.shape[0], 1024, 14, 14)
    return feature_emb, patch_emb


# Adapted from /home/hujinlan/cosie/COSIE/image_preprocessing.py::image_feature_extraction
def image_feature_extraction(
    he_image,
    uni_local_dir,
    cell_location,
    device=None,
    batch_size=128,
    num_workers=4,
    output_cache_path=None,
):
    """
    Extract COSIE UNI features and optionally save them to an explicit cache path.

    Unlike COSIE's original function, this does not write ``uni_embeddings.pickle``
    into the current working directory unless that exact path is explicitly passed.
    """

    import torch
    import tqdm
    from torch.utils.data import DataLoader

    print("cell num:", cell_location.shape[0])

    model = create_model(uni_local_dir)
    print("Finish loading model")

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    dataset = PatchDataset(he_image, cell_location)
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    patch_embeddings = []
    for batch_idx, (patches, positions) in enumerate(tqdm.tqdm(dataloader, total=len(dataloader))):
        patches = patches.to(device, non_blocking=True)
        if batch_idx == 0:
            print(f"Batch {batch_idx}:")
            print(f"Shape of patches: {patches.shape}")
            print(f"Shape of positions[0]: {positions[0].shape}")
            print(f"Content of positions[0][:10]: {positions[0][:10]}")
            print(f"Content of positions[1][:10]: {positions[1][:10]}")

        feature_emb, patch_emb = extract_features(model, patches)

        if batch_idx == 0:
            print(f"Shape of feature_emb: {feature_emb.shape}")
            print(f"Shape of patch_emb: {patch_emb.shape}")

        for idx in range(len(positions[0])):
            center_feature = feature_emb[idx, 0]
            patch_feature = patch_emb[idx, :, 7, 7]
            combined_feature = torch.cat([center_feature, patch_feature])
            patch_embeddings.append(combined_feature.cpu().numpy())

    feature_array = np.asarray(patch_embeddings)
    if output_cache_path:
        if str(output_cache_path).endswith(".npy"):
            ensure_dir(output_cache_path)
            np.save(output_cache_path, feature_array)
        else:
            save_pickle(patch_embeddings, output_cache_path)
    return feature_array


def build_he_adata_from_uni_feature(
    reference_adata,
    uni_feature_key="UNI_feature",
    spatial_key="spatial",
):
    """Build HE AnnData from an existing UNI feature stored in ``reference_adata.obsm``."""

    reference_adata = load_h5ad_if_needed(reference_adata)
    if reference_adata is None:
        return None
    if uni_feature_key not in reference_adata.obsm:
        raise KeyError(f"Reference AnnData is missing obsm['{uni_feature_key}'].")
    if spatial_key not in reference_adata.obsm:
        raise KeyError(f"Reference AnnData is missing obsm['{spatial_key}'].")

    adata_he = ad.AnnData(X=np.asarray(reference_adata.obsm[uni_feature_key]))
    adata_he.obsm["spatial"] = np.asarray(reference_adata.obsm[spatial_key]).copy()
    return adata_he


def build_he_adata_from_feature_file(
    feature_file,
    spatial=None,
    spatial_key="spatial",
):
    """Build HE AnnData from a saved feature file plus explicit spatial coordinates."""

    path = Path(feature_file)
    if path.suffix == ".npy":
        features = np.load(path)
    elif path.suffix in {".pkl", ".pickle"}:
        features = np.asarray(load_pickle(path))
    else:
        raise ValueError("HE feature file must be .npy, .pkl, or .pickle.")

    if spatial is None:
        raise ValueError("spatial coordinates are required for HE feature files.")
    if isinstance(spatial, (str, os.PathLike)):
        spatial = np.load(spatial)

    adata_he = ad.AnnData(X=np.asarray(features))
    adata_he.obsm["spatial"] = np.asarray(spatial).copy()
    if spatial_key != "spatial":
        adata_he.obsm[spatial_key] = adata_he.obsm["spatial"].copy()
    return adata_he


def build_he_adata_from_image_and_mask(
    he_image_path,
    mask_path,
    uni_dir=UNI_DIR,
    device=None,
    batch_size=128,
    num_workers=4,
    output_cache_path=None,
    spatial_key="spatial",
):
    """Build HE AnnData from raw HE image and mask using COSIE's UNI path."""

    he_image = load_image(he_image_path)
    centers = np.asarray(get_white_superpixel_centers(mask_path))
    if centers.size == 0:
        raise ValueError("No fully white superpixels found in mask.")
    centers = centers[:, [1, 0]]
    spatial_location = (centers - 8) // 16
    print(f"Find {len(centers)} superpixels")

    features = image_feature_extraction(
        he_image,
        uni_dir,
        centers,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        output_cache_path=output_cache_path,
    )

    adata_he = ad.AnnData(X=np.asarray(features))
    adata_he.obsm["spatial"] = spatial_location
    if spatial_key != "spatial":
        adata_he.obsm[spatial_key] = spatial_location.copy()
    return adata_he
