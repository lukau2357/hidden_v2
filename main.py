import torch
import matplotlib.pyplot as plt
import numpy as np
import yaml
import cv2
import time

from augmentations.valuemetric import DiffJPEG, Hue, GaussianBlur, Contrast, Brightness, Saturation
from augmentations.geometric import Combine, Rotate, Crop
from augmentations.geometric import Resize as ResizeAug, Perspective, HorizontalFlip
from augmentations.splicing import PixelSplicing, BoxSplicing
from model import Embedder, Extractor
from modules import JND
from PIL import Image
from torchvision import transforms
from utils import normalize, unnormalize, psnr
from augmentations.augmenter import Augmenter
from dataset import ImageDataset
from train import eval

"""
Ad-hoc script for debuggin mostly, does not contain anything related to model definition, training etc.
"""

if __name__ == "__main__":
    device = "cuda"
    # image_orig = Image.open("./val2014/val2014/COCO_val2014_000000000395.jpg")
    image_orig = Image.open("./test_images/lena.jpg")
    # image_orig = cv2.imread("./val2014/val2014/COCO_val2014_000000000074.jpg")
    # image_orig = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
    with open("./model_configurations/base.yaml", encoding = "utf-8") as f:
        conf = yaml.safe_load(f)

    # X = torch.tensor(image_orig.transpose((2, 0, 1)), device = device, dtype = torch.float32).unsqueeze(0)
    tr = conf["embedder"]["true_resolution"]

    transform = transforms.Compose([
            # transforms.RandomResizedCrop(size = (128, 128), scale = (0.5, 1)),
            # transforms.Resize((128, 128)),
            transforms.ToTensor()
    ])

    X = transform(image_orig).unsqueeze(0).to(device) * 255.0
    checkpoint = torch.load("./second_checkpoint/best.pt", weights_only = True)
    embedder = Embedder.load(checkpoint["embedder"]).to(device)
    embedder.jnd_alpha = 0.25
    embedder.jnd.contrast_scale = 0.2

    extractor = Extractor.load(checkpoint["extractor"]).to(device)
    # augmenter = Augmenter.load(checkpoint["augmenter"]).to(device)
    # augmenter = DiffJPEG(min_quality = 40, max_quality = 80).to(device)
    embedder.eval()
    extractor.eval()
    
    message = (torch.rand((1, embedder.capacity), device = device) >= 0.5).to(torch.int32)
    X_wm = embedder(X, message)
    
    print(f"Imperceptibility: {psnr(unnormalize(X_wm), X, max_value = 255.0).item()} dB.")

    preds = extractor(X_wm)
    l = torch.nn.functional.binary_cross_entropy_with_logits(preds, message.to(torch.float32))
    l.backward()

    non_zero_grads = 0
    num_params = 0

    preds = (preds >= 0).to(torch.int32)

    # X = unnormalize(X)
    X_wm = unnormalize(X_wm)
    print(X_wm.min(), X_wm.max())
    print(X.shape)
    print(X_wm.shape)

    pred_acc = (message == preds).to(torch.float32).mean().item()
    print(f"Decoding acc: {pred_acc:.2f}")

    # embedder.jnd.gamma = 0.6
    X_jnd = embedder.jnd(X) * 255
    X = X.cpu().numpy()[0].transpose((1, 2, 0))
    X_wm = X_wm.detach().cpu().numpy()[0].transpose((1, 2, 0))

    diff = 25 * np.abs(X - X_wm).astype(np.uint8)
    X = X.astype(np.uint8)
    X_wm = X_wm.astype(np.uint8)
    X_jnd = X_jnd.detach().cpu().numpy()[0].transpose((1, 2, 0)).astype(np.uint8)

    fig, ax = plt.subplots(ncols = 4)
    ax[0].imshow(X)
    ax[1].imshow(X_wm)
    ax[3].imshow(X_jnd)

    ax[0].set_axis_off()
    ax[1].set_axis_off()
    ax[2].set_axis_off()
    ax[3].set_axis_off()

    ax[0].set_title("Original image")
    ax[1].set_title("Watermarked image")
    ax[2].set_title("Global watermark")
    ax[3].set_title("JND heatmap")
    plt.show()
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # image_orig = Image.open("./val2014/val2014/COCO_val2014_000000000074.jpg")
    # image_orig = np.asarray(image_orig)
    # X = torch.tensor(image_orig.transpose((2, 0, 1)), device = device, dtype = torch.float32).unsqueeze(0)
    # model = Embedder(32, num_layers = 3).to(device)
    # params = 0

    # for param in model.parameters():
    #     if param.requires_grad:
    #         params += param.numel()
    
    # print(f"Number of parameters: {params / 1e6}M.")

    # messages = (torch.rand((1, 32), device = device) >= 0.5).to(torch.int32)
    # X_wm = model(X, messages)
    # jpeg_layer = JPEG(min_quality = 40, max_quality = 80)
    # X_wm = jpeg_layer(X_wm)

    # optim = torch.optim.Adam(model.parameters())
    # optim.zero_grad()
    # l = ((X_wm - normalize(X)) ** 2).mean()
    # print(l.item())
    # l.backward()

    # for param in model.parameters():
    #     if param.requires_grad:
    #         print(param.grad)

    # print(f"Allocated GPU memory: {torch.cuda.memory_allocated() / 1024 ** 2}MB.")

    # wm = y.detach().cpu().numpy()[0].transpose((1, 2, 0))
    # fig, ax = plt.subplots(ncols = 2)
    # ax[0].imshow(image_orig)
    # ax[1].imshow(wm)
    
    # ax[0].set_axis_off()
    # ax[1].set_axis_off()
    # plt.show()

    # image_orig = Image.open("./val2014/val2014/COCO_val2014_000000000074.jpg")
    # image_orig = np.asarray(image_orig)
    # image = torch.tensor(image_orig.transpose((2, 0, 1)), device = device, dtype = torch.float32).unsqueeze(0)

    # model = JND(gamma = 3).to(device)
    # start = time.time()
    # h = model(image)
    # end = time.time()

    # print(f"JND forward prop in gradient mode time in seconds: {end - start}s.")

    # h_heatmap = h[0].cpu().numpy().transpose((1, 2, 0))
    # fig, ax = plt.subplots(ncols = 2, figsize = (12, 6))

    # ax[0].imshow(image_orig)
    # ax[0].set_title("Original image")

    # ax[1].imshow(h_heatmap)
    # ax[1].set_title(f"JND heatmap.\n$\gamma = 0.3$")

    # ax[0].set_axis_off()
    # ax[1].set_axis_off()

    # plt.show()
