import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm 
BATCH      = 8
LR         = 0.00017460018677938576
DICE_W     = 0.5418555579004514         
DROPOUT    = 0.4798347392313353
WEIGHT_DEC = 1e-4                       



# Load images
data_dir = r'segmentation/data'
image_paths = glob(os.path.join(data_dir, '*.png')) 

# Split into train, validation, and test sets
train_paths, test_paths = train_test_split(image_paths, test_size=0.2, random_state=42)
train_paths, val_paths = train_test_split(train_paths, test_size=0.25, random_state=42) 

print(f"Training images: {len(train_paths)}")
print(f"Validation images: {len(val_paths)}")
print(f"Test images: {len(test_paths)}")



def residual_block(x, filters, p_drop=DROPOUT):
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), padding='same')(x)

    # convolutional layers
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    if p_drop > 0:                              # dropout len ak >0
        x = tf.keras.layers.SpatialDropout2D(p_drop)(x)
    x = Conv2D(filters, (3, 3), activation=None, padding='same')(x)

    # skip-connection
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    return x


def improved_unet(input_shape=(256, 256, 1)):
    inputs = Input(input_shape)
    
    # Encoder
    c1 = residual_block(inputs, 32)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = residual_block(p1, 64)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = residual_block(p2, 128)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = residual_block(p3, 256)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = residual_block(p4, 512)

    # Decoder
    u4 = UpSampling2D((2, 2))(c5)
    u4 = Concatenate()([u4, c4])
    c6 = residual_block(u4, 256)
    
    u3 = UpSampling2D((2, 2))(c6)
    u3 = Concatenate()([u3, c3])
    c7 = residual_block(u3, 128)
    
    u2 = UpSampling2D((2, 2))(c7)
    u2 = Concatenate()([u2, c2])
    c8 = residual_block(u2, 64)
    
    u1 = UpSampling2D((2, 2))(c8)
    u1 = Concatenate()([u1, c1])
    c9 = residual_block(u1, 32)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    return Model(inputs, outputs)


def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

@tf.keras.utils.register_keras_serializable()
def combined_loss(y_true, y_pred):
    dice = dice_loss(y_true, y_pred)
    bce  = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    return DICE_W * dice + (1.0 - DICE_W) * bce


def preprocess_image(image_path, target_size=(256, 256), alpha=1.2, beta=30):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)  # alpha controls contrast; beta adds brightness

    img_resized = cv2.resize(img, target_size) / 255.0  # Normalize
    img_resized = np.expand_dims(img_resized, axis=-1)
    
    # Generate pseudo-mask
    _, mask = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)  # High-intensity regions
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Combine with edges for better contours
    edges = cv2.Canny((blurred).astype(np.uint8), 50, 150)
    mask = cv2.bitwise_or(mask, edges)
    
    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    mask_resized = cv2.resize(mask, target_size) / 255.0
    mask_resized = np.expand_dims(mask_resized, axis=-1)
    return img_resized, mask_resized


model = improved_unet(input_shape=(256, 256, 1))
optim  = AdamW(learning_rate=LR, weight_decay=WEIGHT_DEC)
model.compile(optimizer=optim,
              loss=combined_loss,
              metrics=['accuracy'])


image_dir = r"segmentation/data"
image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])

images, pseudo_masks = zip(*[preprocess_image(img) for img in image_paths])
images = np.array(images)
pseudo_masks = np.array(pseudo_masks)

# Train-validation split
train_images, val_images, train_masks, val_masks = train_test_split(images, pseudo_masks, test_size=0.2, random_state=42)

early_stop = EarlyStopping(
        monitor="val_loss",       
        patience=10,              
        min_delta=1e-4,         
        restore_best_weights=True,  
        verbose=1)

# Training
history = model.fit(
    train_images, train_masks,
    validation_data=(val_images, val_masks),
    epochs=100,  
    batch_size=BATCH,
    callbacks=[early_stop]
)

# Predict
test_image = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
test_image_resized = cv2.resize(test_image, (256, 256)) / 255.0
test_image_resized = np.expand_dims(test_image_resized, axis=-1)
test_image_resized = np.expand_dims(test_image_resized, axis=0)

# Predict without thresholding
predicted_mask = model.predict(test_image_resized)[0, :, :, 0]

# Apply threshold to predicted mask
binary_mask = (predicted_mask > 0.5).astype(np.uint8)


output_mask_dir = r"segmentation\generated_masksNew"
os.makedirs(output_mask_dir, exist_ok=True)

# Process all images and generate masks
for image_path in tqdm(image_paths, desc="Generating Masks"):
    # Preprocess the input image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (256, 256)) / 255.0  # Resize and normalize
    img_resized = np.expand_dims(img_resized, axis=-1)  # Add channel dimension
    img_resized = np.expand_dims(img_resized, axis=0)   # Add batch dimension

    # Predict the mask
    predicted_mask = model.predict(img_resized)[0, :, :, 0]  # Remove batch and channel dims
    binary_mask = (predicted_mask > 0.5).astype(np.uint8) * 255  # Binarize and scale to 255

    # Save the mask
    image_name = os.path.basename(image_path)
    mask_output_path = os.path.join(output_mask_dir, f"mask_{image_name}")
    cv2.imwrite(mask_output_path, binary_mask)

print(f"Masks generated and saved in: {output_mask_dir}")
