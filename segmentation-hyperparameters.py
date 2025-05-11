import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision.transforms.functional import to_tensor
from pathlib import Path
import random
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode
import optuna

SAVE_DIR = Path("segmentation/validation_results")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(r"segmentation/data")
MASK_DIR = Path(r"segmentation/generated_masks5")
THRESHOLD = 0.5


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2 with dropout"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    """U-Net architecture"""
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        self.encoder1 = DoubleConv(in_channels, 64)
        self.encoder2 = DoubleConv(64, 128)
        self.encoder3 = DoubleConv(128, 256)
        self.encoder4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(nn.MaxPool2d(2)(enc1))
        enc3 = self.encoder3(nn.MaxPool2d(2)(enc2))
        enc4 = self.encoder4(nn.MaxPool2d(2)(enc3))
        bottleneck = self.bottleneck(nn.MaxPool2d(2)(enc4))

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = self.decoder4(torch.cat((dec4, enc4), dim=1))
        dec3 = self.upconv3(dec4)
        dec3 = self.decoder3(torch.cat((dec3, enc3), dim=1))
        dec2 = self.upconv2(dec3)
        dec2 = self.decoder2(torch.cat((dec2, enc2), dim=1))
        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat((dec1, enc1), dim=1))

        return self.final_conv(dec1)

class UNetWithDropout(nn.Module):
    """Modified U-Net with dropout layers"""
    def __init__(self, in_channels, out_channels):
        super(UNetWithDropout, self).__init__()
        
        # Encoder
        self.encoder1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128, 64)
        
        # Final output
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = self.decoder4(torch.cat((dec4, enc4), dim=1))
        dec3 = self.upconv3(dec4)
        dec3 = self.decoder3(torch.cat((dec3, enc3), dim=1))
        dec2 = self.upconv2(dec3)
        dec2 = self.decoder2(torch.cat((dec2, enc2), dim=1))
        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat((dec1, enc1), dim=1))
        
        return self.final_conv(dec1)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def check_stop(self, val_loss):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)  # Sigmoid activation
        target = target.float()

        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_bce=0.5):
        super(CombinedLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.weight_dice * dice + self.weight_bce * bce


# Dataset Definition
class SegmentationDataset(Dataset):
    def __init__(self, DATA_DIR, MASK_DIR, joint_transform=None, tensor_transform=None):
        self.DATA_DIR = DATA_DIR
        self.MASK_DIR = MASK_DIR
        self.images = sorted(os.listdir(DATA_DIR))
        self.masks = sorted(os.listdir(MASK_DIR))
        self.joint_transform  = joint_transform
        self.tensor_transform = tensor_transform

    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.DATA_DIR / self.images[idx]
        mask_path = self.MASK_DIR / self.masks[idx]
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Convert to grayscale
        

        # paired geometric transforms 
        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)

        # channel / numeric transforms 
        if self.tensor_transform:
            image = self.tensor_transform(image)
            mask  = self.tensor_transform(mask)
        
        return image, mask

class JointCompose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img, msk):
        for t in self.transforms:
            img, msk = t(img, msk)
        return img, msk



class JointRandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img, msk):
        if random.random() < self.p:
            img = F.hflip(img)
            msk = F.hflip(msk)
        return img, msk


class JointRandomRotation:
    def __init__(self, degrees: float = 30):
        self.degrees = degrees

    def __call__(self, img, msk):
        angle = random.uniform(-self.degrees, self.degrees)
        img = F.rotate(img, angle,interpolation=InterpolationMode.BILINEAR, expand=False)
        msk = F.rotate(msk, angle, interpolation=InterpolationMode.NEAREST, expand=False)
        return img, msk


def dice_score(pred, target, threshold=THRESHOLD):

    pred = (torch.sigmoid(pred) > threshold).float()  # Binarize predictions
    target = target.float()
    
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection) / (union + 1e-7)  # Avoid division by zero
    
    return dice.mean().item()

def plot_segmentation(image, mask, prediction, epoch, idx):


    image = image.permute(1, 2, 0).cpu().numpy()  # Convert image to HWC format
    mask = mask.cpu().numpy()
    prediction = (torch.sigmoid(prediction) > THRESHOLD).float().cpu().numpy()
    out_path = SAVE_DIR / f"epoch{epoch}_batch{idx}.png"

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Ground Truth Mask")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(prediction, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis("off")
    
    plt.tight_layout()
    out_path = SAVE_DIR / f"validation_results_epoch{epoch}_batch{idx}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_one_epoch(model, loader, criterion, optimizer=None):
    """Trénuje alebo vyhodnotí model na jednom loaderi.
       Ak `optimizer` je None → eval mód, inak train."""
    train = optimizer is not None
    model.train() if train else model.eval()
    epoch_loss, dice_sum = 0.0, 0.0

    for imgs, msk in loader:
        imgs, msk = imgs.to(device), msk.to(device)

        if train: optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, msk)
        if train:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        dice_sum   += dice_score(preds, msk)

    n = len(loader)
    return epoch_loss / n, dice_sum / n          # avg_loss, avg_dice



def objective(trial):
    # ---------- 1. HYPER-PARAMETERS to be tunned ----------
    lr       = trial.suggest_loguniform("lr", 1e-5, 3e-3)
    bs       = trial.suggest_categorical("batch", [4, 8, 16, 32])
    drop_p   = trial.suggest_uniform("dropout", 0.0, 0.5)
    dice_w   = trial.suggest_uniform("dice_w", 0.3, 0.9)
    th       = trial.suggest_uniform("th", 0.3, 0.6)
    opt_name = trial.suggest_categorical("optim", ["Adam", "AdamW"])
    # -------------------------------------------------------------

    # 2. data
    joint_tf = JointCompose([
        JointRandomHorizontalFlip(p=0.5),
        JointRandomRotation(degrees=30)
    ])
    tensor_tf = T.Compose([T.Resize((256, 256)), T.ToTensor()])

    dataset = SegmentationDataset(DATA_DIR, MASK_DIR,
                                  joint_transform=joint_tf,
                                  tensor_transform=tensor_tf)
    train_len = int(0.8 * len(dataset))
    val_len   = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False)

    # 3. Model
    model = UNetWithDropout(3, 1)
    for m in model.modules():
        if isinstance(m, nn.Dropout2d):
            m.p = drop_p
    model.to(device)

    # 4. Loss & optimiser
    criterion = CombinedLoss(weight_dice=dice_w,
                             weight_bce=1 - dice_w)
    if opt_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:                           # AdamW s weight decay
        wd = trial.suggest_loguniform("weight_decay", 1e-7, 1e-4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                      weight_decay=wd)

    # 5. short training 5 epochs
    best_val_dice = 0
    for epoch in range(5):
        run_one_epoch(model, train_loader, criterion, optimizer)
        _, val_dice = run_one_epoch(model, val_loader, criterion)

        best_val_dice = max(best_val_dice, val_dice)
        trial.report(-best_val_dice, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # 6. Save the best model
    torch.save(model.state_dict(),
               f"segmentation/best_trial_{trial.number}.pth")

    # 7. return the best validation score
    return -best_val_dice





# Code:

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30, timeout=2*60*60)  

    print("\n===== NAJLEPŠÍ TRIAL =====")
    print("Val Dice:", -study.best_value)
    for k, v in study.best_trial.params.items():
        print(f"{k:>12}: {v}")

    # all results saved to CSV
    df = study.trials_dataframe()
    df.to_csv("segmentation/optuna_trials.csv", index=False)


best = study.best_trial
vals = {**best.params, "dice": -best.value}
tmpl = ("Learning-rate {lr:.2e}, batch {batch}, dropout {dropout:.2f}, "
        "dice_w {dice_w:.2f}, threshold {th:.2f}, optim {optim}, "
        "→ val Dice {dice:.3f}")
print(tmpl.format(**vals))
