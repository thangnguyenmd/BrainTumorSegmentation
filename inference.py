import os
import time
from datasets.dataset_synapse import MriDataset
import albumentations as A
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage.morphology import binary_dilation
from glob import glob
from data_frame_utils import get_file_row, iou_pytorch, dice_pytorch, BCE_dice, EarlyStopping
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tqdm import tqdm
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from tensorboardX import SummaryWriter

def test_run(model, test_loader):
    
    model.eval()
    with torch.no_grad():
        running_IoU = 0
        running_dice = 0
        running_test_loss = 0
        for i, data in enumerate(test_loader):
            img, mask = data
            img, mask = img.to(device), mask.to(device)
            predictions = model(img)
            predictions = predictions.squeeze(1)
            running_dice += dice_pytorch(predictions, mask).sum().item()
            running_IoU += iou_pytorch(predictions, mask).sum().item()
            loss = loss_fn(predictions, mask)
            running_test_loss += loss.item() * img.size(0)
    test_loss = running_test_loss / len(test_loader.dataset)
    test_dice = running_dice / len(test_loader.dataset)
    test_IoU = running_IoU / len(test_loader.dataset)
    
    print(f'| Test loss: {test_loss} | Test Mean IoU: {test_IoU} '
        f'| Test Dice coefficient: {test_dice}')
        
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = 'kaggle_3m/data.csv'
    files_dir = 'kaggle_3m/'
    file_paths = glob(f'{files_dir}/*/*[0-9].tif')
    df = pd.read_csv(csv_path)
    imputer = SimpleImputer(strategy="most_frequent")
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    filenames_df = pd.DataFrame((get_file_row(filename) for filename in file_paths), columns=['Patient', 'image_filename', 'mask_filename'])
    df = pd.merge(df, filenames_df, on="Patient")
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    test_df, valid_df = train_test_split(test_df, test_size=0.5, random_state=42)

    valid_dataset = MriDataset(valid_df)
    test_dataset = MriDataset(test_df)

    batch_size = 8
    img_size = 256

    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)
    config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
    config_vit.n_classes = 1
    config_vit.n_skip = 3

    model = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes).cuda()
    model.load_state_dict(torch.load("weights.pt"))

    loss_fn = BCE_dice
    test_run( model, valid_loader)
