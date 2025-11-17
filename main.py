import torch
import matplotlib.pyplot as plt
import numpy as np

from augmentations.valuemetric import DiffJPEG
from PIL import Image
from torchvision.transforms import Resize
if __name__ == "__main__":
    device = "cuda"
    image_orig = Image.open("./val2014/val2014/COCO_val2014_000000000074.jpg")
    image_orig = np.asarray(image_orig)
    X = torch.tensor(image_orig.transpose((2, 0, 1)), device = device, dtype = torch.float32).unsqueeze(0)
    r = Resize((512, 512))
    X = r(X)
    
    diff_jpeg = DiffJPEG(min_quality = 100, max_quality = 100).to(device)
    X_ed = diff_jpeg(X).clamp(0, 255)

    X = X.cpu().numpy()[0].transpose((1, 2, 0)).astype(np.uint8)
    X_ed = X_ed.cpu().numpy()[0].transpose((1, 2, 0)).astype(np.uint8)

    fig, ax = plt.subplots(ncols = 2)
    ax[0].imshow(X)
    ax[1].imshow(X_ed)
    ax[0].set_axis_off()
    ax[1].set_axis_off()
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