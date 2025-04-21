"""
Model training script for Mineral Classifier

Author: Ivan Klevanski

"""
import ast
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import os
import scipy.stats as stats
import pickle
import random
import sklearn.model_selection as sk_ms
import sklearn.metrics as skl_m
import torch.nn as nn
import torch.cuda
import torch.utils
import torch.utils.data
import torchvision
import traceback
import warnings

from model import *
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

warnings.simplefilter("ignore", category=(pd.errors.SettingWithCopyWarning))

# Static Arguments
abs_path = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 42

# Argparser Arguments
argParser = argparse.ArgumentParser()

argParser.add_argument("-m", "--model", type=str, default="mineral_cnn", help="Model Specifier")
argParser.add_argument("-p", "--path", type=str, default="D:\\Users\\ibarn\\Documents\\Dataset Repository\\image\\mineralimage5k", help="Images path")
argParser.add_argument("-e", "--epochs", type=int, default=40, help="Number of epochs")
argParser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch Size")
argParser.add_argument("-o", "--output", type=str, default=os.path.join(abs_path, "output"), help="Output directory")
argParser.add_argument("-pm", "--pretrained_model", type=str, default="", help="Path to pretrained model")
args = argParser.parse_args()

model_name = args.model
dataset_path = args.path
epochs = args.epochs
batch_size = args.batch_size
use_pretrained = args.pretrained_model != ""
pretrained_path = args.pretrained_model

images_path = os.path.join(dataset_path, "mineral_images")

# Dynamic Arguments
num_classes = 10 # Will change depending on number of usable images from dataset


""" ------ Dataset ------ """

class ImageDataSet(Dataset):
    """
    PyTorch Dataset for image data
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.horiz_oversample_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0)
        ])

        self.vert_oversample_transforms = transforms.Compose([
            transforms.RandomVerticalFlip(p=1.0)
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        path = self.df["path"].iloc[item]
        oversample = self.df["oversamples"].iloc[item]
        image_path = self.df["path"].iloc[item]

        label = ast.literal_eval(str(self.df["OHE"].iloc[item]).replace(' ', ','))

        image = Image.open(os.path.join(images_path, path)).convert("RGB")

        bboxes = ast.literal_eval(self.df["mineral_boxes"].iloc[item])
        #bbox = [box for box in bboxes if box["confidence"] == max([float(box["confidence"]) for box in bboxes])][0]
        bbox = self.get_correct_bbox(bboxes)

        crop = (bbox["box"][0] * image.size[0], bbox["box"][1] * image.size[1], bbox["box"][2] * image.size[0], bbox["box"][3] * image.size[1])
        image = image.crop(crop)
        #image.show()

        if oversample == 1 or oversample == 3:
            image = self.horiz_oversample_transforms(image)
        elif oversample == 2:
            image = self.vert_oversample_transforms(image)
        
        image = self.transforms(image)

        return image, torch.from_numpy(np.array(label).astype(float)), image_path
    
    def get_correct_bbox(self, bboxes: dict):
        output = [box for box in bboxes if box["confidence"] == max([float(box["confidence"]) for box in bboxes])][0]

        if len(bboxes) == 1 and ((output["box"][0] > 0.5 and output["box"][2] > 0.5) or (output["box"][0] <= 0.5 and output["box"][2] <= 0.5)):
            output = {"box": [0.15, 0.15, 0.85, 0.85]}
        return output



""" ------ Utility methods ------ """

def setup():
    """
    Set up the application environment
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        cuda_info = "Cuda modules loaded."
    else:
        cuda_info = "Cuda modules not loaded."

    print("[Info]: " + cuda_info + '\n')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def oversample_df(df: pd.DataFrame):
    classes = df["en_name"].value_counts()
    threshold = max(classes.values)
    for idx, val in classes.items():
        to_add = max(threshold - val, 0)
        adding = to_add > 0

        while adding:
            df_copy = df.loc[df["en_name"] == idx].copy() if len(df.loc[df["en_name"] == idx]) < to_add else df.loc[df["en_name"] == idx].sample(n=to_add)
            to_add -= len(df_copy)
            df_copy.loc[df_copy["en_name"] == idx, "oversamples"] += 1

            if df_copy.loc[df_copy["en_name"] == idx, "oversamples"].max() >= 4:
                break

            df = pd.concat((df, df_copy))
            if to_add <= 0:
                adding = False
    df = df.sample(frac=1).reset_index()
    return df


def debug_samples(dl: DataLoader):
    image_tensor, _, paths = next(iter(dl))

    for i, path in enumerate(paths):
        print(f"{i+1}: {path}")
    visualize_tensor(image_tensor)


def visualize_tensor(t: torch.Tensor, plot_title: str = "", nrow: int = None):

    grid = torchvision.utils.make_grid(t, nrow=int(np.ceil(t.shape[0] ** 0.5)) if nrow == None else nrow)

    plt.imshow(grid.permute(1, 2, 0))
    if plot_title != "":
        plt.title(plot_title)
    plt.show()


def load_preprocess_data(save_pruned = True, max_samples_per_class = -1, override_classes = None):
    print("[Info]: Data Preprocessing")
    raw_df = pd.read_csv(os.path.join(dataset_path, "minerals_full.csv"))

    # Remove records with missing images
    to_remove = []

    print("[Info]: Image data integrity check")
    for idx, row in tqdm(raw_df.iterrows(), total=raw_df.shape[0]):
        if not os.path.exists(os.path.join(images_path, row["path"])):
            to_remove.append(idx)

    raw_df.drop(to_remove, errors="ignore", inplace=True)

    n_classes = raw_df["en_name"].value_counts()

    usable_classes = dict()
    for idx, val in n_classes.items():
        if val > 400: # Can change if needed # 600: 5 classes
            usable_classes[idx] = val
    
    pruned_df = raw_df[raw_df["en_name"].isin(usable_classes.keys() if override_classes is None else override_classes)]

    if max_samples_per_class > 0:
        pruned_df = pruned_df.groupby("en_name").sample(n=max_samples_per_class)
    else:
        pruned_df = raw_df[raw_df["en_name"].isin(usable_classes.keys())]

    # One Hot Encoding
    pruned_df.loc[:, "OHE"] = [x.astype(int) for x in [row.to_numpy() for _, row in pd.get_dummies(pruned_df["en_name"]).iterrows()]]
    pruned_df.reset_index(inplace=True)

    output = pruned_df[["OHE", "en_name", "path", "mineral_boxes"]]

    if save_pruned:
        output.to_csv(os.path.join(dataset_path, "minerals_full_pruned.csv"), index=False)

    n_classes = output["en_name"].nunique()

    # Split dataframe into training, validation, and testing datasets. Add oversampling
    output.loc[:, "oversamples"] = 0

    output = output.sample(frac=1).reset_index()

    train_df, test_df = sk_ms.train_test_split(output, train_size=0.8, test_size=0.2)
    train_df, validation_df = sk_ms.train_test_split(train_df, train_size=0.85, test_size=0.15)

    train_df = oversample_df(train_df)
    validation_df = oversample_df(validation_df)
    test_df = oversample_df(test_df)
  
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    validation_df = validation_df.sample(frac=1).reset_index(drop=True)  
    test_df = test_df.sample(frac=1).reset_index(drop=True)

    return train_df, validation_df, test_df, n_classes


def train_model(model: nn.Module,
                m_name: str,                 
                train_dl: DataLoader, 
                validate_dl: DataLoader,
                loss_fn = nn.CrossEntropyLoss(),
                optim_fn: torch.optim.Optimizer = torch.optim.Adam,
                lr: float = 1e-3,
                epochs: int = epochs):
    """
    Trains and evaluates input model
    """

    def train():

        model.train()

        run_loss = 0
        run_correct = 0

        for images, labels, _ in tqdm(train_dl):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(images)
                loss = loss_fn(outputs, labels)

                loss.backward()
                optimizer.step()

            run_loss += loss.item()
            run_correct += (outputs.argmax(1) == labels.argmax(1)).sum().item()

        return run_loss / len(train_dl.dataset), (run_correct / len(train_dl.dataset)) * 100

    def validate():
        
        model.eval()

        run_loss = 0
        run_correct = 0

        for images, labels, _ in tqdm(validate_dl):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                outputs = model(images)
                loss = loss_fn(outputs, labels)

            run_loss += loss.item()
            run_correct += (outputs.argmax(1) == labels.argmax(1)).sum().item()

        return run_loss / len(validate_dl.dataset), (run_correct / len(validate_dl.dataset)) * 100

    if model is not None:
        try:
            # Initialization
            model = model.to(device)

            optimizer = optim_fn(model.parameters(), lr=lr, weight_decay=0.001)

            best_weights = None
            best_loss = (np.inf, 0)
            train_losses = []
            validation_losses = []

            print("[Info]: Training {}.\n".format(m_name))

            for e in range(0, epochs):
                print("[Info]: Epoch {} out of {}.".format(e + 1, epochs))
                
                epoch_loss, epoch_acc = train()
                print("\n[Info]: Epoch Summary: Train Loss: {:.4f} | Train Accuracy: {:.4f}".format(epoch_loss, epoch_acc))
                train_losses.append(epoch_loss)

                epoch_loss, epoch_acc = validate()
                print("\n[Info]: Epoch Summary: Evaluation Loss: {:.4f} | Evaluation Accuracy: {:.4f}\n".format(epoch_loss, epoch_acc))
                validation_losses.append(epoch_loss)

                # Store best model weights based on highest accuracy
                if epoch_acc > best_loss[1]:
                    best_loss = (epoch_loss, epoch_acc)
                    best_weights = copy.deepcopy(model.state_dict())
            
            print("[Info]: Lowest validation loss: {:.4f} | Corresponding Accuracy: {:.4f}".format(best_loss[0], best_loss[1]))
            return best_weights, train_losses, validation_losses
        except:
            print("[Error]: {} training failed due to an exception, exiting...\n".format(m_name))
            print("[Error]: Exception occurred during training")
            traceback.print_exc()
            exit(1)


def test_model(model: nn.Module, m_name: str, test_dl: DataLoader):
    """
    Tests input model
    """

    if model is not None:
        try:
            model.eval()

            predictions = list()
            gt = list()

            correct = 0

            print("[Info]: Testing {}.\n".format(m_name))

            with torch.no_grad():
                for images, labels, _ in tqdm(test_dl):
                    images = images.to(device)                    

                    outputs = model(images)

                    correct += (outputs.cpu().argmax(1) == labels.argmax(1)).sum().item()

                    predictions.extend(outputs.cpu().argmax(1).numpy())
                    gt.extend(labels.argmax(1).numpy())
            
            predictions = np.array(predictions)
            gt = np.array(gt)

            cr = skl_m.classification_report(gt, predictions)

            print("[Info]: {} accuracy: {:.4f}\n".format(m_name, (correct / len(test_dl.dataset)) * 100))
            print("[Info]: {} Classification report:\n".format(m_name))
            print(cr)

        except:
            print("[Error]: Exception occurred during testing:\n")
            traceback.print_exc()


def prune_model(model: MineralCNNet, model_name: str, sparsity: float = 0.4):
    """
    Magnitude-based, fine-grained pruning
    """
    print(f"[Info]: Pruning: {model_name}")
    with torch.no_grad():
        masks = dict()
        for name, param in model.named_parameters():
            if param.dim() > 1: # conv and fc layers
                sparsity = min(max(0.0, sparsity), 1.0)

                if sparsity == 1.0:
                    param.zero_()
                    return torch.zeros_like(param)
                elif sparsity == 0.0:
                    return torch.ones_like(param)

                num_elements = param.numel()
                importance = torch.abs(param)
                threshold = torch.kthvalue(importance.flatten(), int(round(num_elements * sparsity)))
                mask = importance > threshold[0]
                param.mul_(mask)

                masks[name] = mask

        for name, param in model.named_parameters():
            if name in masks:
                param *= masks[name]
    
    return model


def main():
    setup()

    # Overridden classes
    mineral_classes = [
    'quartz',
    'topaz',
    'agate',
    'beryl',
    'silver',
    'gold',
    'amethyst',
    'diopside',
    'copper',
    'spinel',
    'opal',
    'sulfur'
    ]

    mineral_classes.sort()

    train_df, validation_df, test_df, n_classes = load_preprocess_data(save_pruned=True)

    num_classes = n_classes
    
    # Create datasets

    train_ds = ImageDataSet(train_df)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size)

    validation_ds = ImageDataSet(validation_df)
    validation_dl = DataLoader(validation_ds, shuffle=True, batch_size=batch_size)

    test_ds = ImageDataSet(test_df)
    test_dl = DataLoader(test_ds, shuffle=True, batch_size=batch_size)

    model = MineralCNNet(num_classes=num_classes)
    #model = CNNetWrapper(torchvision.models.resnet18(), base_num_classes=1000, num_classes=num_classes)

    model_name = "Mineral CNN"
    model_weights_fn = "mineralcnn_dsc_4_20_2025"

    do_train = True
    do_prune = False
    prune_loaded = False

    if do_train:
        weights, _, _ = train_model(model=model, m_name=model_name, train_dl=train_dl, validate_dl=validation_dl)
        torch.save(weights, os.path.join(abs_path, f"{model_weights_fn}.pt")) # backup
        model.load_state_dict(weights)

        if do_prune:

            # Prune model
            model = prune_model(model, model_name=model_name)

            # Fine-tune pruned model
            print(f"[Info]: Fine-tuning {model_name}")
            weights, _, _ = train_model(model=model, m_name=model_name, train_dl=train_dl, validate_dl=validation_dl, lr=1e-4)
            torch.save(weights, os.path.join(abs_path, f"{model_weights_fn}_pruned.pt"))
            model.load_state_dict(weights)

    else:
        weights = torch.load(os.path.join(abs_path, f"{model_weights_fn}.pt" if prune_loaded else f"{model_weights_fn}_pruned.pt"), map_location=device, weights_only=True)
        model = model.to(device)
        model.load_state_dict(weights)

        if prune_loaded and do_prune:
            # Prune model
            model = prune_model(model, model_name=model_name)

            # Fine-tune pruned model
            print(f"[Info]: Fine-tuning {model_name}")
            weights, _, _ = train_model(model=model, m_name=model_name, train_dl=train_dl, validate_dl=validation_dl, lr=1e-4)
            torch.save(weights, os.path.join(abs_path, f"{model_weights_fn}_pruned.pt"))
            model.load_state_dict(weights)


    test_model(model=model, m_name="Mineral CNN", test_dl=test_dl)

    pass


if __name__ == "__main__":

    main()
