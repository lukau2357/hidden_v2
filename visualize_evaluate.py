import matplotlib.pyplot as plt
import torch
import json
import os
import numpy as np
import tqdm
import yaml

from model import Embedder, Extractor
from augmentations.geometric import *
from augmentations.valuemetric import *
from augmentations.splicing import *
from utils import normalize
from dataset import ImageDataset
from sklearn.model_selection import train_test_split

def training_plot(checkpoint_path: str):
    data_path = os.path.join(checkpoint_path, "metadata.json")
    with open(data_path, "r", encoding = "utf-8") as f:
        data = json.load(f)

    plt.style.use("ggplot")
    epochs = data["epochs"]
    
    x = np.arange(start = 0, stop = epochs, step = 1)
    fig, ax = plt.subplots(ncols = 2, figsize = (12, 6), sharex = True)

    # TODO: Once final model is trained, add lines on plots for some models that might get gapped by the trained model
    ax[0].set_title("Bit recovery accuracy on validation data")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Ratio of recovered bits")

    ax[1].set_title("Imperceptibility during training on validation data")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("PSNR [dB]")

    ax[0].plot(x, data["acc_values"])
    ax[1].plot(x, data["psnr_values"])

    plt.show()

def eval_aug(loader: torch.utils.data.DataLoader, embedder: Embedder, extractor: Extractor, device: str, augmentation):
    acc = np.array([])

    with torch.no_grad():
        loop = tqdm.tqdm(loader, total = len(loader))

        for X in loop:
            X = X.to(device)
            B = X.shape[0]
            messages = (torch.rand(B, embedder.capacity, device = device) >= 0.5).to(torch.int32) # [B, capacity]
            X_w = embedder(X, messages)

            X = normalize(X)
            X_w = augmentation(X, X_w) if isinstance(augmentation, PixelSplicing) or isinstance(augmentation, BoxSplicing) else augmentation(X_w)

            messages_logits = extractor(X_w)
            messages_logits = (messages_logits >= 0).to(torch.int32)

            current_acc = (messages == messages_logits).to(torch.float32).mean(dim = -1).cpu().numpy()
            acc = np.concatenate((acc, current_acc))

            if "cuda" in device:
                free, total = torch.cuda.mem_get_info(device)
                mem_used_MB = (total - free) / 1024 ** 2
                loop.set_postfix({"GPU memory": f"{mem_used_MB :.2f} MB"})
        
    return float(acc.mean())

def robustness_eval(checkpoint_path: str, conf_path: str, data_path: str, model_id: str = "best.pt", save_file: str = "robustness_eval.json"):
    str2class = {
        "DiffJPEG": DiffJPEG,
        "Brightness": Brightness,
        "Contrast": Contrast,
        "GaussianBlur": GaussianBlur,
        "Hue": Hue,
        "Saturation": Saturation,
        "HorizontalFlip": HorizontalFlip,
        "Perspective": Perspective,
        "Resize": Resize,
        "Crop": Crop,
        "Rotate": Rotate,
        "Combine": Combine,
        "PixelSplicing": PixelSplicing,
        "BoxSplicing": BoxSplicing,
        "Identity": nn.Identity
    }

    aug_groups = {
        "Valuemetric": ["DiffJPEG", "Brightness", "Contrast", "GaussianBlur", "Hue", "Saturation"],
        "Geometric": ["HorizontalFlip", "Perspective", "Resize", "Crop", "Rotate"],
        "Splicing": ["PixelSplicing", "BoxSplicing"],
        "Combine": ["Combine"]
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.path.join(checkpoint_path, model_id)
    model = torch.load(model_path, weights_only = True)
    embedder = Embedder.load(model["embedder"])
    extractor = Extractor.load(model["extractor"])

    embedder = embedder.to(device)
    extractor = extractor.to(device)
    embedder.eval()
    extractor.eval()

    with open(conf_path, "r", encoding = "utf-8") as f:
        aug_conf = yaml.safe_load(f)["augmentations_eval"]["aug_strength"]
    
    metadata_path = os.path.join(checkpoint_path, "metadata.json")
    with open(metadata_path, "r", encoding = "utf-8") as f:
        training_metadata = json.load(f)
    
    image_files = os.listdir(data_path)
    _, eval_files = train_test_split(image_files, train_size = training_metadata["train_ratio"], random_state = training_metadata["data_generation_seed"])
    eval_dataset = ImageDataset(data_path, eval_files, embedder.true_resolution)
    # eval_dataset.images = eval_dataset.images[:100]

    eval_dl = torch.utils.data.DataLoader(eval_dataset, batch_size = training_metadata["batch_size"], 
                                          shuffle = True, 
                                          num_workers = 2,
                                          pin_memory = True)
    res = {}

    # Compute robustness for specific augmentation types
    for aug_type, cls in str2class.items():
        aug = cls(**aug_conf[aug_type]) if aug_conf[aug_type] is not None else cls()
        aug = aug.to(device)
        print(f"Evaluating robustness under augmentation: {aug_type}")
        aug_robustness = eval_aug(eval_dl, embedder, extractor, device, aug)
        res[aug_type] = aug_robustness
        print(f"Inferred robustness: {aug_robustness}")
    
    # Aggregate over augmentation groups
    for ag in aug_groups.keys():
        values = [res[x] for x in aug_groups[ag]]
        agg = sum(values) / len(values)
        res[ag] = agg

    save_path = os.path.join(checkpoint_path, save_file)
    with open(save_path, "w+", encoding = "utf-8") as f:
        json.dump(res, f, indent = 4)

if __name__ == "__main__":
    # training_plot("./third_checkpoint")
    robustness_eval("./third_checkpoint", "./model_configurations/base.yaml", "./val2014/val2014")