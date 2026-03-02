import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.losses import BinaryFocalCrossentropy
import tensorflow.keras.backend as K


# -----------------------------
# Loss Functions
# -----------------------------

def dice_loss(y_true, y_pred):
    smooth = 1e-5
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    )


def combined_focal_dice_loss(y_true, y_pred):
    focal = BinaryFocalCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return focal + dice


# -----------------------------
# Hybrid ResNet50 Model
# -----------------------------

def build_resnet50_hybrid(
    input_shape=(128, 128, 12),
    dropout_rate=0.4,
    trainable_resnet=False
):

    inputs = Input(shape=input_shape)

    # 12 → 3 channel projection block
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(3, (1, 1), padding="same", activation="relu")(x)

    # ResNet50 backbone
    resnet = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(128, 128, 3)
    )

    resnet.trainable = trainable_resnet
    x = resnet(x)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)

    return model, resnet
