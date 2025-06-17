import tensorflow as tf
import numpy as np
import os
import config
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)

# Path to the test image (replace with your own image path)
img_path = '/content/download.jpg'  # Example: 'Sheep_Classification_Images/train/sheep1.jpg'

# Load and preprocess the image
img = image.load_img(img_path, target_size=config.TARGET_SIZE)
img_array = image.img_to_array(img)
img_array = img_array / 255.0  # Normalize
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Predict
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)[0]

# Map class index to label
class_indices = {v: k for k, v in model.class_indices.items()} if hasattr(model, 'class_indices') else None

# Fallback: use validation generator to get class mapping
if class_indices is None:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import pandas as pd

    val_df = pd.read_csv(config.VAL_SPLIT_CSV)
    val_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
        val_df,
        directory=os.path.join(config.DATA_DIR, "train"),
        x_col="filename",
        y_col="label",
        class_mode="categorical",
        target_size=config.TARGET_SIZE,
        batch_size=1,
        shuffle=False
    )
    class_indices = {v: k for k, v in val_gen.class_indices.items()}

# Print predicted class
print(f"✅ Predicted class: {class_indices[predicted_class]}")
