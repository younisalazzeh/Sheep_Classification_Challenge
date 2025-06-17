import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import config
import os

def create_transfer_learning_model(num_classes):
    base_model = MobileNetV2(input_shape=(*config.TARGET_SIZE, 3),
                             include_top=False,
                             weights='imagenet')
    base_model.trainable = False  # Freeze base model layers

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model

def main():
    # Load dataframes
    train_df = pd.read_csv(config.TRAIN_SPLIT_CSV)
    val_df = pd.read_csv(config.VAL_SPLIT_CSV)

    # Data augmentation parameters, you can adjust or import from config
    augmentation_params = dict(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest"
    )

    train_datagen = ImageDataGenerator(rescale=1./255, **augmentation_params)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=os.path.join(config.DATA_DIR, "train"),
        x_col="filename",
        y_col="label",
        target_size=config.TARGET_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )

    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=os.path.join(config.DATA_DIR, "train"),
        x_col="filename",
        y_col="label",
        target_size=config.TARGET_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    num_classes = len(train_df['label'].unique())
    model = create_transfer_learning_model(num_classes)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_generator,
        epochs=config.EPOCHS,
        validation_data=val_generator,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
        ],
        verbose=1
    )

    model.save(config.MODEL_SAVE_PATH)
    print(f"✅ Model saved at: {config.MODEL_SAVE_PATH}")

    # Print final accuracies
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"📊 Final Training Accuracy: {final_train_acc:.4f}")
    print(f"📊 Final Validation Accuracy: {final_val_acc:.4f}")

if __name__ == "__main__":
    main()
