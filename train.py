import torch
import math
import importlib
import os
import tqdm
import torch.nn.functional as F
import json
import yaml

from itertools import chain
from torch.optim.lr_scheduler import _LRScheduler
from utils import normalize, unnormalize
from model import Embedder, Extractor
from augmentations.augmenter import Augmenter
from typing import Union
from dataset import ImageDataset, collate_fn
from sklearn.model_selection import train_test_split
from functools import partial

class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """
    Scheduler popularized by Attention is all you need paper. Learning rate is warmed up linearly for the given number of steps first,
    then decayed following a cosine schedule.
    """
    def __init__(self, optimizer, warmup_steps, max_steps, eta_min = 0.0, last_epoch = -1):
        assert max_steps > warmup_steps, "max_steps must be greater than warmup_steps"

        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eta_min = eta_min # minimum achievable learning rate, will practically always be 0

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Warmup phase
        if self.last_epoch < self.warmup_steps:
            warmup_factor = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        # Cosine annealing phase
        progress = (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        cos_term = 0.5 * (1 + math.cos(math.pi * progress))

        return [
            self.eta_min + (base_lr - self.eta_min) * cos_term
            for base_lr in self.base_lrs
        ]

def load_class(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def serialize_optimizer(optimizer):
    return {
        "class": optimizer.__class__.__name__,
        "module": optimizer.__class__.__module__,
        "init_kwargs": optimizer.defaults,   # includes LR, betas, weight_decay, etc.
        "state": optimizer.state_dict(),
    }

def deserialize_optimizer(opt_info, model_params):
    cls = load_class(opt_info["module"], opt_info["class"])
    optimizer = cls(model_params, **opt_info["init_kwargs"])
    optimizer.load_state_dict(opt_info["state"])
    return optimizer

def serialize_scheduler(scheduler: LinearWarmupCosineAnnealingLR):
        return {
        "class": scheduler.__class__.__name__,
        "module": scheduler.__class__.__module__,
        "init_kwargs": {
            "warmup_steps": scheduler.warmup_steps,
            "max_steps": scheduler.max_steps,
            "eta_min": scheduler.eta_min,
            "last_epoch": scheduler.last_epoch
        },
        "state": scheduler.state_dict(),
    }

def deserialize_scheduler(sched_info, optimizer):
    if sched_info is None:
        return None
    
    cls = load_class(sched_info["module"], sched_info["class"])
    scheduler = cls(optimizer, **sched_info["init_kwargs"])
    scheduler.load_state_dict(sched_info["state"])
    return scheduler

def eval(loader: torch.utils.data.DataLoader, embedder: Embedder, extractor: Extractor, device: str):
    acc = []

    is_embedder_train = embedder.training
    is_extractor_train = extractor.training

    embedder.eval()
    extractor.eval()

    with torch.no_grad():
        loop = tqdm.tqdm(loader, total = len(loader))

        for X in loop:
            X = X.to(device)
            B = X.shape[0]
            messages = (torch.rand(B, embedder.capacity, device = device) >= 0.5).to(torch.int32) # [B, capacity]
            X_w = embedder(X, messages)
            messages_logits = extractor(X_w)
            messages_logits = (messages_logits >= 0).to(torch.int32)

            current_acc = (messages == messages_logits).sum().item()
            acc.append(current_acc)
            if "cuda" in device:
                free, total = torch.cuda.mem_get_info(device)
                mem_used_MB = (total - free) / 1024 ** 2
                loop.set_postfix({"GPU memory": f"{mem_used_MB :.2f} MB"})
    
    if is_embedder_train:
        embedder.train()
    
    if is_extractor_train:
        extractor.train()
    
    return sum(acc) / len(acc)

def state_checkpoint(epoch: int, 
                     eval_acc: float, 
                     embedder: Embedder, 
                     extractor: Extractor, 
                     optimizer: torch.optim.Optimizer, 
                     lr_scheduler: LinearWarmupCosineAnnealingLR,
                     checkpoint_dir: str,
                     checkpoint_name: str):
    d = {
        "epoch": epoch,
        "eval_acc": eval_acc,
        "embedder": embedder.to_dict(),
        "extractor": extractor.to_dict(),
        "optimizer": serialize_optimizer(optimizer)
    }

    if lr_scheduler is not None:
        d["lr_scheduler"] = serialize_scheduler(lr_scheduler)
    
    torch.save(d, os.path.join(checkpoint_dir, checkpoint_name))

def train(embedder: Embedder, 
          extractor: Extractor, 
          augmenter: Augmenter,
          optimizer: torch.optim.Optimizer, 
          scheduler: Union[None, LinearWarmupCosineAnnealingLR],
          chck_path: str,
          train_dl: torch.utils.data.DataLoader,
          eval_dl: torch.utils.data.DataLoader,
          device: str,
          train_metadata = {}
          ):
    
    """
        - Inside chck_path save best.pt, last.pt, and training metadata, such as loss history, accuracy on validation and training, best achieved accuracy
    """
    
    current_epoch = train_metadata.get("current_epoch")
    lambda_1 = train_metadata.get("lambda_1")
    lambda_2 = train_metadata.get("lambda_2")
    best_acc = train_metadata.get("best_acc")
    loss_ema_beta = train_metadata.get("loss_ema_beta")
    epochs = train_metadata.get("epochs")

    acc_values = train_metadata.get("acc_values", [])
    prev_loss = train_metadata.get("prev_loss", 0)

    embedder.train()
    extractor.train()

    for i in range(current_epoch, epochs):
        loop = tqdm.tqdm(train_dl, total = len(train_dl), leave = True)
        loop.set_description(f"Epoch [{i + 1} / {epochs}]")

        for j, X in enumerate(loop):
            # X [B, C, H, W], [0, 255] quantization
            X = X.to(device)
            optimizer.zero_grad()
            B = X.shape[0]
            messages = (torch.rand(B, embedder.capacity, device = device) >= 0.5).to(torch.int32) # [B, capacity]
            X_w = embedder(X, messages) # [B, C, H, W], [-1, 1] quantization
            X_wa = augmenter(normalize(X), X_w)
            messages_logits = extractor(X_wa) # [B, capacity]

            reconstruction_loss = lambda_1 * ((normalize(X) - X_w) ** 2).mean()
            decoding_loss = lambda_2 * (F.binary_cross_entropy_with_logits(messages_logits, messages.to(torch.float32)))
            current_loss = reconstruction_loss + decoding_loss
            current_loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            current_loss = current_loss * (1 - loss_ema_beta) + loss_ema_beta * prev_loss
            logger_dict = {"Loss": f"{current_loss:.4f}"}
            if "cuda" in device:
                free, total = torch.cuda.mem_get_info(device)
                mem_used_MB = (total - free) / 1024 ** 2
                logger_dict["GPU memory"] = f"{mem_used_MB:.2f} MB"

            loop.set_postfix(logger_dict)
            prev_loss = current_loss.item()

        # Eval only on validation dataset. Will not be able to see overfitting...
        print("Evaluating...")
        current_acc = eval(eval_dl, embedder, extractor, device)
        acc_values.append(current_acc)

        print(f"Epoch {i + 1} validation message recovery accuracy: {current_acc}")
        if current_acc > best_acc:
            best_acc = current_acc
            print("Creating the checkpoint for new best accuracy.")
            state_checkpoint(i + 1, best_acc, embedder, extractor, optimizer, scheduler, chck_path, "best.pt")
        
        print("Creating the checkpoint for the last epoch.")
        state_checkpoint(i + 1, best_acc, embedder, extractor, optimizer, scheduler, chck_path, "last.pt")

        print("Saving training metadata.")

        # Update only the changing parameters
        train_metadata["current_epoch"] = i + 1
        train_metadata["best_acc"] = best_acc
        train_metadata["prev_loss"] = prev_loss
        train_metadata["acc_values"] = acc_values

        with open(os.path.join(chck_path, "metadata.json"), "w+", encoding = "utf-8") as f:
            json.dump(train_metadata, f, indent = 4)

def load_and_train(model_conf_path: str, 
                   data_root_path: str,
                   chck_path: str,
                   batch_size: int = 8,
                   train_ratio: float = 0.8,
                   data_generation_seed: int = 41,
                   epochs: int = 10,
                   learning_rate: float = 3e-4,
                   adam_betas = (0.9, 0.999),
                   create_scheduler = True,
                   warmup_ratio = 0.1,
                   lambda_1: float = 0.2, # MSE scaling factor
                   lambda_2: float = 1.0, # BCE scaling factor
                   loss_ema_beta: float = 0.1
                   ):
    
    """
    MSE for inputs in [-1, 1] and BCE are not on the same scale. MSE is in range [0, 4] for inputs in [-1, 1], BCE is practically unbounoded,
    but for single class classification in the case of highest entropy we would have -ln(1 / 2) ~ 0.7. If we keep lambda_2 = 1, an attempt
    to re-scale MSE to BCE range would be to set lambda_1 = 0.7 / 4 ~ 0.2, which is the default value.
    """

    # Load model configuration
    with open(model_conf_path, "r", encoding = "utf-8") as f:
        conf = yaml.safe_load(f)
    
    # Load training metadata, or create new data appropriately
    metadata_path = os.path.join(chck_path, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding = "utf-8") as f:
            training_metadata = json.load(f)
    
    else:
        training_metadata = {
            "loss_ema_beta": loss_ema_beta,
            "lambda_1": lambda_1,
            "lambda_2": lambda_2,
            "learning_rate": learning_rate,
            "adam_betas": adam_betas,
            "warmup_ratio": warmup_ratio,
            "epochs": epochs,
            "current_epoch": 0,
            "best_acc": 0,
            "batch_size": batch_size,
            "train_ratio": train_ratio,
            "data_generation_seed": data_generation_seed
        }

    image_files = os.listdir(data_root_path)
    train_files, eval_files = train_test_split(image_files, train_size = training_metadata["train_ratio"], random_state = training_metadata["data_generation_seed"])

    train_dataset = ImageDataset(data_root_path, train_files)
    eval_dataset = ImageDataset(data_root_path, eval_files)

    train_dataset.images = train_dataset.images[:100]
    eval_dataset.images = eval_dataset.images[:100]

    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size = training_metadata["batch_size"], 
                                           shuffle = True, 
                                           num_workers = 2,
                                           collate_fn = partial(collate_fn, target_resolution = conf["embedder"]["true_resolution"]),
                                           pin_memory = True,
                                           )
    
    eval_dl = torch.utils.data.DataLoader(eval_dataset, batch_size = training_metadata["batch_size"], 
                                          shuffle = True, 
                                          num_workers = 2,
                                          collate_fn = partial(collate_fn, target_resolution = conf["embedder"]["true_resolution"]),
                                          pin_memory = True)

    augmenter = Augmenter(conf["augmentations"]["train"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    augmenter = augmenter.to(device)

    if not os.path.exists(chck_path):
        os.mkdir(chck_path)
    
    last_path = os.path.join(chck_path, "last.pt")
    if os.path.exists(last_path):
        d = torch.load(last_path, weights_only = True)
        embedder = Embedder.load(d["embedder"])
        extractor = Extractor.load(d["extractor"])
        embedder = embedder.to(device)
        extractor = extractor.to(device)

        optimizer = deserialize_optimizer(d["optimizer"], chain(embedder.parameters(), extractor.parameters()))
        lr_scheduler = None

        if "lr_scheduler" in d:
            lr_scheduler = deserialize_scheduler(d["lr_scheduler"], optimizer)
    
    else:
        embedder = Embedder(**conf["embedder"])
        extractor = Extractor(**conf["extractor"])
        embedder = embedder.to(device)
        extractor = extractor.to(device)

        optimizer = torch.optim.Adam(chain(embedder.parameters(), extractor.parameters()), lr = learning_rate, betas = adam_betas)
        lr_scheduler = None

        if create_scheduler:
            num_steps = epochs * len(train_dl)
            lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_steps = int(warmup_ratio * num_steps), max_steps = num_steps)
    
    train(embedder, extractor, augmenter, optimizer, lr_scheduler, chck_path, train_dl, eval_dl, device, training_metadata)

if __name__ == "__main__":
    data_root_path = os.path.join("val2014", "val2014")
    batch_size = 4
    model_conf_path = os.path.join("model_configurations", "base.yaml")
    checkpoint_path = "./first_checkpoint"
    load_and_train(model_conf_path, data_root_path, checkpoint_path, batch_size = batch_size, create_scheduler = False)