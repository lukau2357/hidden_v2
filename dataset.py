import torch
import os
import cv2
import tqdm
import time

from functools import partial

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, images):
        self.root_path = root_path
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        res = cv2.imread(os.path.join(self.root_path, self.images[idx]))
        res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
        return torch.tensor(res, dtype = torch.float32)

def collate_fn(batch, target_resolution = 256):
    # DataLoader collation
    batch = [torch.nn.functional.interpolate(x.unsqueeze(0), (target_resolution, target_resolution), mode = "bilinear") for x in batch]
    return torch.concatenate(batch, dim = 0)

if __name__ == "__main__":
    files = os.listdir("val2014/val2014/")

    min_res = (4096, 4096)
    max_res = (1, 1)

    test = ImageDataset("./val2014/val2014", ["COCO_val2014_000000058651.jpg"])
    dl = torch.utils.data.DataLoader(test, batch_size = 1, shuffle=  True, collate_fn = partial(collate_fn), num_workers = 2)
    start = time.time()

    for file in tqdm.tqdm(files):
        # image = Image.open(os.path.join("val2014", "val2014", file))
        image = cv2.imread(os.path.join("val2014", "val2014", file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(image.min(), image.max())
        # image = np.asarray(image)
        current_res = image.shape

        if current_res[0] * current_res[1] < min_res[0] * min_res[1]:
            min_res = current_res
        
        if current_res[0] * current_res[1] > max_res[0] * max_res[1]:
            max_res = current_res
    
    end = time.time()
    print(f"Iteration time: {end - start} s.")
    print(f"Min resolution: {min_res}")
    print(f"Max resolution: {max_res}")