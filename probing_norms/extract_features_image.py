import pdb
import requests

from functools import partial
from pathlib import Path
from tqdm import tqdm

import click
import h5py
import torch
import numpy as np
import timm
import torch.nn as nn

from transformers import (
    AutoModel,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
    PaliGemmaForConditionalGeneration,
    ViTMAEModel,
)
from transformers.models.siglip import SiglipVisionModel

from torch.utils.data import DataLoader

import torchvision.transforms as T

from open_clip import create_model_from_pretrained

from probing_norms.data import DATASETS


def count_params(model):
    return sum(p.numel() for p in model.parameters())


class ImageBackboneDINO(nn.Module):
    def __init__(self, type_):
        assert type_ == "resnet50"
        super(ImageBackboneDINO, self).__init__()

        self.model = torch.hub.load(
            "facebookresearch/dino:main",
            "dino_" + type_,
            pretrained=True,
        )
        self.model = nn.Sequential(*list(self.model.children())[:-2])

        IMAGE_SIZE = 224
        transform_image_norm = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.transform = T.Compose(
            [
                T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                T.ToTensor(),
                transform_image_norm,
            ]
        )

    def forward(self, x):
        features = self.model(x)
        features = features.mean([2, 3])
        return features


class CLIP(nn.Module):
    def __init__(self):
        super(CLIP, self).__init__()
        model_id = "openai/clip-vit-large-patch14"
        self.model = CLIPModel.from_pretrained(model_id).eval()
        self.feature_dim = self.model.config.projection_dim
        self.processor = CLIPProcessor.from_pretrained(model_id)

    def transform(self, x):
        output = self.processor(images=x, return_tensors="pt")
        output = output["pixel_values"]
        output = output.squeeze(0)
        return output

    def forward(self, x):
        return self.model.get_image_features(x)


class OpenCLIP(nn.Module):
    def __init__(self, name):
        super(OpenCLIP, self).__init__()
        assert name == "hf-hub:apple/DFN2B-CLIP-ViT-L-14"
        model, processor = create_model_from_pretrained(name)
        self.model = model.eval()
        self.transform = processor

        name_short = name.split(":")[1]
        URL = "https://huggingface.co/{}/resolve/main/config.json".format(name_short)
        config = requests.get(URL)
        config = config.json()
        self.feature_dim = config["projection_dim"]

    def forward(self, x):
        return self.model.encode_image(x)


class SigLIP(nn.Module):
    def __init__(self, use_random_weights=False):
        super(SigLIP, self).__init__()
        model_id = "google/siglip-so400m-patch14-224"
        model = AutoModel.from_pretrained(model_id).eval()
        self.feature_dim = model.config.vision_config.hidden_size

        if use_random_weights:
            model = SiglipVisionModel(model.config.vision_config)
            model = model.eval()

        self.model = model.vision_model
        self.model.head = nn.Identity()
        self.processor = AutoProcessor.from_pretrained(model_id)

    def transform(self, x):
        output = self.processor(images=x, return_tensors="pt")
        output = output["pixel_values"]
        output = output.squeeze(0)
        return output

    def forward(self, x):
        features = self.model(x)
        features = features.last_hidden_state.mean(1)
        return features


class PaliGemma(nn.Module):
    def __init__(self):
        super(PaliGemma, self).__init__()
        model_id = "google/paligemma-3b-mix-224"
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.feature_dim = self.model.config.vision_config.hidden_size

    def transform(self, x):
        output = self.processor.image_processor(images=x, return_tensors="pt")
        output = output["pixel_values"]
        output = output.squeeze(0)
        return output

    def forward(self, x):
        features = self.model.vision_tower(x)
        features = features.last_hidden_state.mean(1)
        return features


class VITMAE(nn.Module):
    def __init__(self):
        super(VITMAE, self).__init__()
        model_id = "facebook/vit-mae-large"
        self.model = ViTMAEModel.from_pretrained(model_id).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.feature_dim = self.model.config.hidden_size

    def transform(self, x):
        output = self.processor(images=x, return_tensors="pt")
        output = output["pixel_values"]
        output = output.squeeze(0)
        return output

    def forward(self, x):
        features = self.model(x)
        features = features.last_hidden_state.mean(1)
        return features


class TimmModel(nn.Module):
    def __init__(self, model_id):
        super(TimmModel, self).__init__()
        self.model = timm.create_model(model_id, pretrained=True, num_classes=0)
        self.model = self.model.eval()
        self.data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(
            **self.data_config,
            is_training=False,
        )
        self.feature_dim = self.model.num_features

    def forward(self, x):
        return self.model(x)


class DINOV2(nn.Module):
    def __init__(self):
        super(DINOV2, self).__init__()
        model_id = "facebook/dinov2-large"
        self.model = AutoModel.from_pretrained(model_id).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.feature_dim = self.model.config.hidden_size

    def transform(self, x):
        output = self.processor(images=x, return_tensors="pt")
        output = output["pixel_values"]
        output = output.squeeze(0)
        return output

    def forward(self, x):
        features = self.model(x)
        features = features.last_hidden_state.mean(1)
        return features



FEATURE_EXTRACTORS = {
    # fmt: off
    # Self supervised models
    "dino-resnet50": partial(ImageBackboneDINO, type_="resnet50"),
    "dino-v2": DINOV2,
    "vit-mae-large": VITMAE,
    # Image-text models
    "clip": CLIP,
    "clip-dfn2b": partial(OpenCLIP, name="hf-hub:apple/DFN2B-CLIP-ViT-L-14"),
    "siglip-224": SigLIP,
    "pali-gemma-224": PaliGemma,
    # Supervised models
    "swin-v2-ssl": partial(TimmModel, model_id="swinv2_large_window12_192.ms_in22k"),
    "swin-v2": partial(TimmModel, model_id="swinv2_large_window12to24_192to384.ms_in22k_ft_in1k"),
    "max-vit-large": partial(TimmModel, model_id="maxvit_large_tf_384.in1k"),
    "max-vit-large-in21k": partial(TimmModel, model_id="maxvit_large_tf_224.in21k"),
    # Random models
    "random-siglip": partial(SigLIP, use_random_weights=True),
    # fmt: on
}


@click.command()
@click.option("-d", "--dataset", "dataset_name", type=str, required=True)
@click.option("-f", "--feature-type", "feature_type", type=str, required=True)
def main(dataset_name, feature_type):
    DEVICE = "cuda"

    feature_extractor = FEATURE_EXTRACTORS[feature_type]()
    feature_extractor.eval()
    feature_extractor.to(DEVICE)

    dataset = DATASETS[dataset_name](transform=feature_extractor.transform)
    dataloader = DataLoader(dataset, batch_size=16, num_workers=4)

    def extract1(image):
        with torch.no_grad():
            image = image.to(DEVICE)
            feature = feature_extractor(image)
            feature = feature.cpu().numpy()
            return feature

    num_samples = len(dataset)
    feature_dim = feature_extractor.feature_dim

    X = np.zeros((num_samples, feature_dim))
    y = np.zeros(num_samples)

    i = 0
    for batch in tqdm(dataloader):
        features = extract1(batch["image"])
        for j, feature in enumerate(features):
            X[i] = feature
            y[i] = batch["label"][j].item()
            i += 1

    path_np = f"output/features-image/{dataset_name}-{feature_type}.npz"
    np.savez(path_np, X=X, y=y)


if __name__ == "__main__":
    main()
