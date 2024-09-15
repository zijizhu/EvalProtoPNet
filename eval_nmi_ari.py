import argparse
import logging
import os
import sys
from math import sqrt
from pathlib import Path

import albumentations as A
from einops import rearrange
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from albumentations.augmentations.crops.functional import crop_keypoint_by_coords
from PIL import Image
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

import model

N_KEYPOINTS = 15


class CUBEvalDataset(ImageFolder):
    def __init__(self,
                 images_root: str,
                 annotations_root: str,
                 normalization: bool = True,
                 input_size: int = 224):
        transforms = [A.Resize(width=input_size, height=input_size)]
        transforms += [A.Normalize(mean=(0.485, 0.456, 0.406,), std=(0.229, 0.224, 0.225,))] if normalization else []

        super().__init__(
            root=images_root,
            transform=A.Compose(
                transforms,
                keypoint_params=A.KeypointParams(
                    format='xy',
                    label_fields=None,
                    remove_invisible=True,
                    angle_in_degrees=True
                )
            ),
            target_transform=None
        )
        self.input_size = input_size
        annotations_root = Path("datasets") / "CUB_200_2011"

        path_df = pd.read_csv(annotations_root / "images.txt", header=None, names=["image_id", "image_path"], sep=" ")
        bbox_df = pd.read_csv(annotations_root / "bounding_boxes.txt", header=None, names=["image_id", "x", "y", "w", "h"], sep=" ")
        self.bbox_df = path_df.merge(bbox_df, on="image_id")
        self.part_loc_df = pd.read_csv(annotations_root / "parts" / "part_locs.txt", header=None, names=["image_id", "part_id", "kp_x", "kp_y", "visible"], sep=" ")
        
        attributes_np = np.loadtxt(annotations_root / "attributes" / "class_attribute_labels_continuous.txt")
        self.attributes = F.normalize(torch.tensor(attributes_np, dtype=torch.float32), p=2, dim=-1)

    def __getitem__(self, index: int):
        im_path, label = self.samples[index]
        im = np.array(Image.open(im_path).convert("RGB"))

        row = self.bbox_df[self.bbox_df["image_path"] == "/".join(Path(im_path).parts[-2:])].iloc[0]
        image_id = row["image_id"]
        bbox_coords = row[["x", "y", "w", "h"]].values.flatten()
        
        mask = self.part_loc_df["image_id"] == image_id
        keypoints = self.part_loc_df[mask][["kp_x", "kp_y"]].values

        keypoints_cropped = [crop_keypoint_by_coords(keypoint=tuple(kp) + (None, None,), crop_coords=bbox_coords[:2]) for kp in keypoints]
        keypoints_cropped = [(np.clip(x, 0, self.input_size), np.clip(y, 0, self.input_size),) for x, y, _, _ in keypoints_cropped]
        
        transformed = self.transform(image=im, keypoints=keypoints_cropped)
        transformed_im, transformed_keypoints = transformed["image"], transformed["keypoints"]
        
        return to_tensor(transformed_im), torch.tensor(transformed_keypoints, dtype=torch.float32), label, self.attributes[label, :], index


@torch.no_grad()
def get_attn_maps(net: nn.Module, images: torch.Tensor, labels: torch.Tensor, C: int = 200, K: int = 10):
    logits, (cosine_min_distances, project_activations, shallow_feas, deep_feas) = net(images)
    batch_size, CK, n_patches = project_activations.shape
    H = W = int(sqrt(n_patches))
    project_activations = rearrange(project_activations, "B (C K) (H W) -> B C K H W", C=C, K=K, H=H, W=W)
    return project_activations[torch.arange(batch_size), labels, ...]


def eval_nmi_ari(net: nn.Module, dataloader: DataLoader, C: int = 200, K: int = 10, device: str = "cpu"):
    """
    Get Normalized Mutual Information, Adjusted Rand Index for given method

    Parameters
    ----------
    net: nn.Module
        The trained net to evaluate
    data_loader: DataLoader
        The dataset to evaluate
    device: str

    Returns
    ----------
    nmi: float
        Normalized Mutual Information between predicted parts and gt parts as %
    ari: float
        Adjusted Rand Index between predicted parts and gt parts as %
    """
    device = torch.device(device)
    net.eval()

    all_class_ids = []
    all_keypoint_part_assignments = []
    all_ground_truths = []

    for batch in tqdm(dataloader):
        batch = tuple(item.to(device) for item in batch)  # type: tuple[torch.Tensor, ...]
        images, keypoints, labels, attributes, sample_indices = batch
        batch_size, _, input_h, input_w = images.shape

        attn_maps = get_attn_maps(net=net, images=images, labels=labels, C=C, K=K)
        attn_maps_resized = F.interpolate(attn_maps, size=(input_h, input_w,), mode='bilinear', align_corners=False)

        kp_visibilities = (keypoints.sum(dim=-1) > 0).to(dtype=torch.bool)
        keypoints = keypoints.clone()[..., None, :]  # B, N_KEYPOINTS, 1, 2

        keypoints /= torch.tensor([input_w, input_h]).to(dtype=torch.float32, device=device)
        keypoints = keypoints * 2 - 1  # map keypoints from range [0, 1] to [-1, 1]

        keypoint_part_logits = F.grid_sample(attn_maps_resized, keypoints, mode='nearest', align_corners=False)  # B K N_KEYPOINTS, 1
        keypoint_part_assignments = torch.argmax(keypoint_part_logits, dim=1).squeeze()  # B N_KEYPOINTS

        for assignments, is_visible, class_id in zip(keypoint_part_assignments.unbind(dim=0), kp_visibilities.unbind(dim=0), labels):
            all_keypoint_part_assignments.append(assignments[is_visible])
            all_class_ids.append(torch.stack([class_id] * is_visible.sum()))
            all_ground_truths.append(torch.arange(N_KEYPOINTS)[is_visible])
    
    
    all_class_ids = torch.cat(all_class_ids, axis=0)
    all_keypoint_part_assignments_np = torch.cat(all_keypoint_part_assignments, axis=0)
    all_ground_truths = torch.cat(all_ground_truths, axis=0)

    all_classes_nmi, all_classes_ari = [], []

    for c in torch.unique(all_class_ids):
        mask = all_class_ids == c
        kp_part_assignment_c = all_keypoint_part_assignments_np[mask].flatten()
        ground_truths_c = all_ground_truths[mask].flatten()

        nmi = normalized_mutual_info_score(kp_part_assignment_c, ground_truths_c)
        ari = adjusted_rand_score(kp_part_assignment_c, ground_truths_c)

        all_classes_nmi.append(nmi)
        all_classes_ari.append(ari)

    return sum(all_classes_nmi) / len(all_classes_nmi), sum(all_classes_ari) / len(all_classes_nmi)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', type=str, default='0')
    parser.add_argument('--data_set', default='CUB2011', type=str)
    parser.add_argument('--data_path', type=str, default='datasets/cub200_cropped/')
    parser.add_argument('--nb_classes', type=int, default=200)

    # Model
    parser.add_argument('--base_architecture', type=str, default='vgg16')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--prototype_shape', nargs='+', type=int, default=[2000, 64, 1, 1])
    parser.add_argument('--prototype_activation_function', type=str, default='log')
    parser.add_argument('--add_on_layers_type', type=str, default='regular')

    parser.add_argument('--resume', required=True, type=str)
    args = parser.parse_args()

    log_dir = Path(args.resume)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_dir.parent.parent / "eval_nmi_ari.log"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    logger = logging.getLogger(__name__)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
    img_size = args.input_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    C = args.nb_classes
    CK, DIM, _, _ = args.prototype_shape
    K = CK // C

    dataset_eval = CUBEvalDataset(images_root=Path("datasets") / "cub200_cropped" / "test_cropped",
                                annotations_root=Path("datasets") / "CUB_200_2011",
                                input_size=args.input_size)
    dataloader_eval = DataLoader(dataset=dataset_eval, batch_size=128, num_workers=8, shuffle=True)

    # Load the model
    ppnet = model.construct_OursNet(base_architecture=args.base_architecture,
                                pretrained=True, img_size=img_size,
                                prototype_shape=args.prototype_shape,
                                num_classes=args.nb_classes,
                                prototype_activation_function=args.prototype_activation_function,
                                add_on_layers_type=args.add_on_layers_type)
    ppnet = ppnet.to(device=device)

    checkpoint = torch.load(args.resume, map_location='cpu')
    ppnet.load_state_dict(checkpoint['model'])

    mean_nmi, mean_ari = eval_nmi_ari(ppnet, dataloader=dataloader_eval, device=device, C=C, K=K)
    logger.info(f"Mean class-wise NMI: {float(mean_nmi)}")
    logger.info(f"Mean class-wise ARI: {float(mean_ari)}")
