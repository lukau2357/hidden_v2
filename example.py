import torch
import numpy as np
import yaml

from model import Embedder, Extractor
from PIL import Image
from torchvision import transforms
from utils import unnormalize, psnr

if __name__ == "__main__":
    # Not recommended to run on CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Add paths to images you want to watermark
    images = [# "./val2014/val2014/COCO_val2014_000000013333.jpg",
               # "./test_images/lena.jpg",
               # "./val2014/val2014/COCO_val2014_000000002149.jpg",
               # "./val2014/val2014/COCO_val2014_000000020371.jpg",
               "./val2014/val2014/COCO_val2014_000000000285.jpg"
               ]
    
    image_orig = [Image.open(x) for x in images]
    with open("./model_configurations/base.yaml", encoding = "utf-8") as f:
        conf = yaml.safe_load(f)

    checkpoint = torch.load("./checkpoint_v1/best.pt", weights_only = True)
    true_resolution = checkpoint["embedder"]["args"]["true_resolution"]

    # If this is set to False, all images in the batch need to have same resolution!
    # Alternatively, pass images through the model one by one.
    resize_to_tr = False

    transform = transforms.Compose([transforms.Resize((true_resolution, true_resolution)), transforms.ToTensor()] if resize_to_tr else [transforms.ToTensor()])

    X = torch.stack([transform(x) for x in image_orig]) * 255.0
    X = X.to(device)

    print(f"Checkpoint accuracy without augmentations: {checkpoint['eval_acc'] :.4f}")
    print(f"Checkpoint imperceptibility: {checkpoint['eval_psnr'] :.4f} dB")

    embedder = Embedder.load(checkpoint["embedder"]).to(device)
    extractor = Extractor.load(checkpoint["extractor"]).to(device)
    embedder.eval()
    extractor.eval()
    
    em_params, ex_params = 0, 0
    for param in embedder.parameters():
        if param.requires_grad:
            em_params += param.numel()

    for param in extractor.parameters():
        if param.requires_grad:
            ex_params += param.numel()
    
    print(f"Embedder number of parameters: {em_params / 1e6 :.2f} M")
    print(f"Extractor number of parameters: {ex_params / 1e6 :.2f} M")

    B = X.shape[0]
    message = (torch.rand((B, embedder.capacity), device = device) >= 0.5).to(torch.int32)

    # Watermark images
    with torch.no_grad():
        X_wm = embedder(X, message)
    
    # Report achieved imperceptibility
    print(f"Mean imperceptibility: {psnr(unnormalize(X_wm), X, max_value = 255.0).mean().item() :.4f} dB")

    # Run watermarked images through extractor
    with torch.no_grad():
        preds = extractor(X_wm)

    preds = (preds >= 0).to(torch.int32)

    # Transform back to [0, 255]
    X_wm = unnormalize(X_wm)
    pred_acc = (message == preds).to(torch.float32).mean().item()
    # Compute and report message decoding accuracy
    print(f"Decoding accuracy: {pred_acc :.4f}")

    X = X.cpu().numpy()
    X_wm = X_wm.cpu().numpy().astype(np.uint8)

    # Compute difference image for visualization
    diff = (10 * np.abs(X - X_wm)).astype(np.uint8)
    X = X.astype(np.uint8)

    # [B, C, H, W] => [B, H, W, C]
    X_t = X.transpose((0, 2, 3, 1))
    X_wm_t = X_wm.transpose((0, 2, 3, 1))
    diff_t = diff.transpose((0, 2, 3, 1))

    # Concatenate horizontally for each row
    rows = []
    for i in range(B):
        row = np.concatenate([X_t[i], X_wm_t[i], diff_t[i]], axis = 1)
        rows.append(row)

    create_grid = True
    
    if not create_grid:
        for row in rows:
            img = Image.fromarray(row)
            img.show()

    else:
        # Concatenate all rows vertically
        grid = np.concatenate(rows, axis = 0)

        # Ensure values are in [0, 255] range and convert to uint8
        grid = np.clip(grid, 0, 255).astype(np.uint8)

        # Create PIL Image
        img = Image.fromarray(grid)
        img.show()