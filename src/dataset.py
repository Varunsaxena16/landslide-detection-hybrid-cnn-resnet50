import os
import numpy as np
import tensorflow as tf
import albumentations as A
import cv2

# Albumentations Pipeline
albumentations_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.GaussNoise(p=0.2),
    A.GaussianBlur(blur_limit=(1, 1), p=0.1),
])

# Load + Normalize + Resize
def load_npy_image(path, target_size=(128, 128), normalize=True):
    img = np.load(path).astype(np.float32)

    if normalize:
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)

    img_resized = np.stack([
        cv2.resize(img[:, :, i], target_size, interpolation=cv2.INTER_LINEAR)
        for i in range(img.shape[-1])
    ], axis=-1)

    return img_resized.astype(np.float32)


def load_and_preprocess(image_path, label, augment=False):

    def _load(path):
        path = path.numpy().decode("utf-8")
        img = load_npy_image(path)

        if augment:
            img = albumentations_transform(image=img)["image"]

        return img

    image = tf.py_function(func=_load, inp=[image_path], Tout=tf.float32)
    image.set_shape([128, 128, 12])

    return image, tf.cast(label, tf.float32)

# tf.data Builder
def create_tf_dataset(
    df,
    folder_path,
    batch_size=16,
    augment=False,
    shuffle=True
):

    image_paths = [
        os.path.join(folder_path, f"{img_id}.npy")
        for img_id in df["ID"]
    ]

    labels = df["label"].values

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)

    ds = ds.map(
        lambda path, label: load_and_preprocess(path, label, augment),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
