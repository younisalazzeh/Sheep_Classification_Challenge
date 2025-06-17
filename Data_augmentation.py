import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, save_img
import config
from tqdm import tqdm

def create_augmented_dataset(output_dir, augmentations_per_image=5):
    """
    Generates augmented images from the training dataset and saves them to disk.

    Args:
      output_dir (str): Directory to save augmented images.
      augmentations_per_image (int): Number of augmented versions per original image.
    """
    os.makedirs(output_dir, exist_ok=True)
    data_dir = config.DATA_DIR
    train_csv = config.TRAIN_SPLIT_CSV

    # Load training dataframe
    train_df = pd.read_csv(train_csv)

    # Define augmentation parameters (should match those used in training)
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
        rescale=1./255
    )

    print(f"🚀 Starting augmentation of {len(train_df)} images, {augmentations_per_image} augmentations each...")

    for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
        filename = row['filename']
        label = row['label']

        # Load original image
        img_path = os.path.join(data_dir, "train", filename)
        img = load_img(img_path, target_size=config.TARGET_SIZE)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)  # shape (1, height, width, channels)

        # Create output dir for this class if not exists
        class_dir = os.path.join(output_dir, label)
        os.makedirs(class_dir, exist_ok=True)

        # Generate and save augmented images
        i = 0
        for batch in datagen.flow(x, batch_size=1):
            augmented_img = batch[0]  # get numpy array image from batch

            # Rescale back to [0,255] and convert to uint8 before saving
            augmented_img = (augmented_img * 255).astype('uint8')

            aug_filename = f"{os.path.splitext(filename)[0]}_aug_{i}.png"
            save_path = os.path.join(class_dir, aug_filename)
            save_img(save_path, augmented_img)

            i += 1
            if i >= augmentations_per_image:
                break

    print(f"✅ Augmented images saved in '{output_dir}'")

if __name__ == "__main__":
    # Define output folder for augmented images
    augmented_data_dir = os.path.join(config.DATA_DIR, "train_augmented")
    create_augmented_dataset(augmented_data_dir, augmentations_per_image=5)
