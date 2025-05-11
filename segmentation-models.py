import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision.transforms.functional import to_tensor
from pathlib import Path
import random
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode
import segmentation_models_pytorch as smp
import timm



SAVE_DIR = Path("segmentation/validation_results")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(r"segmentation/data")
MASK_DIR = Path(r"segmentation/generated_masks5")
LR = 0.0023337072137759097
BATCH_SIZE=  16
DROPOUT_P = 0.43498360821913545
DICE_WEIGHT = 0.35695347056147286
THRESHOLD= 0.42610762019278425
OPTIM= "Adam"
EPOCHS = 5


def get_unetpp():
    return smp.UnetPlusPlus(
        encoder_name="resnet34",     # backbone
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None             # raw logits → BCE+Dice
    )




class TransUNet(nn.Module):
    def __init__(self, img_size=256, vit='vit_base_patch16_224', n_classes=1):
        super().__init__()
        # Vision-Transformer
        self.vit = timm.create_model(
            vit, pretrained=True, in_chans=3, img_size=img_size, num_classes=0
        )
        self.d4  = nn.Conv2d(768, 512, 1)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up1 = nn.ConvTranspose2d(128,  64, 2, 2)
        self.out = nn.Conv2d( 64, n_classes, 1)

    def forward(self, x):
        B, _, H, W = x.shape
        feat = self.vit.patch_embed(x)                      # (B, P, 768)
        feat = feat.transpose(1, 2).reshape(B, 768, H//16, W//16)

        x = self.d4(feat)            # 1/16
        x = self.up3(x)              # 1/8
        x = self.up2(x)              # 1/4
        x = self.up1(x)              # 1/2
        x = Fnn.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)  
        return self.out(x)






class _CBR(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

class EfficientNetUNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, backbone="efficientnet_b0"):
        super().__init__()

        # encoder
        self.backbone = timm.create_model(
            backbone,
            features_only=True,
            out_indices=(1, 2, 3, 4),   # stride 4,8,16,32
            in_chans=in_channels,
            pretrained=False            
        )
        ch = self.backbone.feature_info.channels()  # dynamic [C4,C8,C16,C32]
        c4, c8, c16, c32 = ch                     

        # decoder
        self.up4  = nn.ConvTranspose2d(c32, 256, 2, 2)          # 1/32→1/16
        self.dec4 = _CBR(256 + c16, 256)

        self.up3  = nn.ConvTranspose2d(256, 128, 2, 2)          # 1/16→1/8
        self.dec3 = _CBR(128 + c8, 128)

        self.up2  = nn.ConvTranspose2d(128,  64, 2, 2)          # 1/8→1/4
        self.dec2 = _CBR( 64 + c4,  64)

        self.up1  = nn.ConvTranspose2d( 64,  32, 2, 2)          # 1/4→1/2
        self.dec1 = _CBR(32, 32)

        self.out  = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        f4, f8, f16, f32 = self.backbone(x)  

        d4 = self.up4(f32)                    # 1/16
        d4 = self.dec4(torch.cat([d4, f16], 1))

        d3 = self.up3(d4)                     # 1/8
        d3 = self.dec3(torch.cat([d3, f8], 1))

        d2 = self.up2(d3)                     # 1/4
        d2 = self.dec2(torch.cat([d2, f4], 1))

        d1 = self.up1(d2)                     # 1/2
        d1 = self.dec1(d1)

        d0 = Fnn.interpolate(d1, scale_factor=2,
                           mode="bilinear", align_corners=True)  # full-res
        return self.out(d0)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2 with dropout"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=DROPOUT_P),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=DROPOUT_P)
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
        pred = torch.sigmoid(pred)  # Binarize predictions
        target = target.float()

        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, weight_dice=DICE_WEIGHT, weight_bce=1-DICE_WEIGHT):
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
        self.masks  = sorted(os.listdir(MASK_DIR))
        self.joint_transform  = joint_transform
        self.tensor_transform = tensor_transform

    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.DATA_DIR / self.images[idx]
        mask_path = self.MASK_DIR / self.masks[idx]
        

        img_name = self.images[idx]                   # "control (7).png"
        img_path = self.DATA_DIR / img_name
        image    = Image.open(img_path).convert("RGB")

        mask_name = f"mask_{img_name}"               # "mask_control (7).png"
        mask_path = self.MASK_DIR / mask_name
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask {mask_path} neexistuje")

        mask = Image.open(mask_path).convert("L")

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

def plot_segmentation(image, mask, prediction, epoch, idx,name):


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
    out_path = SAVE_DIR / f"model{name}_epoch_{epoch}_batch{idx}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()








# Code:

# Transforms
joint_tf = JointCompose([
        JointRandomHorizontalFlip(p=0.5),
        JointRandomRotation(degrees=30)
])


tensor_tf = T.Compose([
    T.Resize((256, 256)),  
    T.ToTensor(),           # Convert to PyTorch tensors
])

# Load Dataset
dataset = SegmentationDataset(DATA_DIR, MASK_DIR, joint_transform=joint_tf,
                               tensor_transform=tensor_tf)

# Split Dataset
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# DataLoaders
batch_size = BATCH_SIZE
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_zoo = {
    "UNet_baseline":     UNetWithDropout(in_channels=3, out_channels=1),
    "EffUNet":       EfficientNetUNet(in_channels=3, n_classes=1),
    "UNet_plus_plus":    get_unetpp(),
    "TransUNet":     TransUNet(img_size=256, n_classes=1)
}

results = {}

for name, net in model_zoo.items():
    print(f"\n=== Trénujem {name} ===")
    model = net.to(device)

    criterion = CombinedLoss(weight_dice=DICE_WEIGHT, weight_bce=1-DICE_WEIGHT)  
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    best_dice = 0.0
    epochs = EPOCHS
    for epoch in range(EPOCHS):
         # Training
        model.train()
        train_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_dice = 0
        with torch.no_grad():  # Disable gradient computation
            for idx, (images, masks) in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{epochs}")):
                images, masks = images.to(device), masks.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_dice += dice_score(outputs, masks)

                # Visualization for the first batch of each epoch
                if idx == 0:
                    for i in range(min(3, len(images))):  # Visualize up to 3 images per epoch
                        plot_segmentation(images[i], masks[i][0], outputs[i][0], epoch, idx,name)

        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        
        # Print Epoch Summary
        print(f"Epoch [{epoch+1}/{epochs}] - "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val Dice Score: {avg_val_dice:.4f}")
        

        best_dice = max(best_dice, avg_val_dice)
        if early_stopping.check_stop(avg_val_loss):
            break

    results[name] = {
        "val_dice": best_dice,
        "epochs": epoch + 1,
        "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
    }

print("\n--- Comparison ---")
print("{:<15} | {:>6} | {:>7} | {}".format("Model", "Dice", "ParamM", "Ep"))
for k, v in results.items():
    print("{:<15} | {:6.3f} | {:7.2f} | {}".format(
          k, v["val_dice"], v["params_M"], v["epochs"]))
