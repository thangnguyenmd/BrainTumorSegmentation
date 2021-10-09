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

def training_loop(writer, epochs, model, train_loader, valid_loader, optimizer, loss_fn, lr_scheduler):
    history = {'train_loss': [], 'val_loss': [], 'val_IoU': [], 'val_dice': []}
    early_stopping = EarlyStopping(patience=7)
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        running_loss = 0
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            img, mask = data
            img, mask = img.to(device), mask.to(device)
            # print(img.shape, mask.shape)
            predictions = model(img)
            predictions = predictions.squeeze(1)
            # print(torch.max(predictions))
            loss = loss_fn(predictions, mask)
            running_loss += loss.item() * img.size(0)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        model.eval()
        with torch.no_grad():
            running_IoU = 0
            running_dice = 0
            running_valid_loss = 0
            for i, data in enumerate(valid_loader):
                img, mask = data
                img, mask = img.to(device), mask.to(device)
                predictions = model(img)
                predictions = predictions.squeeze(1)
                running_dice += dice_pytorch(predictions, mask).sum().item()
                running_IoU += iou_pytorch(predictions, mask).sum().item()
                loss = loss_fn(predictions, mask)
                running_valid_loss += loss.item() * img.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        val_loss = running_valid_loss / len(valid_loader.dataset)
        val_dice = running_dice / len(valid_loader.dataset)
        val_IoU = running_IoU / len(valid_loader.dataset)
        
        history['train_loss'].append(train_loss)
        writer.add_scalar("Training/Train loss", train_loss, epoch)
        writer.add_scalar("Training/Val loss", val_loss, epoch)
        writer.add_scalar("Metric/Val IoU", val_IoU, epoch)
        writer.add_scalar("Metric/Val Dice", val_dice, epoch)

        history['val_loss'].append(val_loss)
        history['val_IoU'].append(val_IoU)
        history['val_dice'].append(val_dice)
        print(f'Epoch: {epoch}/{epochs} | Training loss: {train_loss} | Validation loss: {val_loss} | Validation Mean IoU: {val_IoU} '
         f'| Validation Dice coefficient: {val_dice}')
        
        lr_scheduler.step(val_loss)
        if early_stopping(val_loss, model):
            early_stopping.load_weights(model)
            break
    model.eval()
    return history

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

    # Tensorboard
    writer = SummaryWriter("tensorboard_logs")

    transform = A.Compose([
    A.ChannelDropout(p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.ColorJitter(p=0.3),
    ])

    train_dataset = MriDataset(train_df, transform)
    valid_dataset = MriDataset(valid_df)
    test_dataset = MriDataset(test_df)

    batch_size = 8
    img_size = 256

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)
    config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
    config_vit.n_classes = 1
    config_vit.n_skip = 3

    model = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes).cuda()
    weight = np.load('../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz')
    model.load_from(weights=weight)

    optimizer = Adam(model.parameters(), lr=0.005)
    epochs = 60
    lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=2,factor=0.1)
    loss_fn = BCE_dice
    history = training_loop(writer, epochs, model, train_loader, valid_loader, optimizer, loss_fn, lr_scheduler)
