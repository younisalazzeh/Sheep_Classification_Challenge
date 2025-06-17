import tensorflow as tf
import pandas as pd
import os
import matplotlib.pyplot as plt
import config
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from Data_augmentation import get_data_generators

def create_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(config.TARGET_SIZE[0], config.TARGET_SIZE[1], 3))
    base_model.trainable = False  # Freeze base

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model

def train_one_fold(fold):
    print(f"\n🚀 Training Fold {fold}")
    train_csv = config.FOLD_CSV_TEMPLATE.format(fold, 'train')
    val_csv = config.FOLD_CSV_TEMPLATE.format(fold, 'val')

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    num_classes = train_df['label'].nunique()

    train_gen, val_gen = get_data_generators()

    train_generator = train_gen.flow_from_dataframe(
        dataframe=train_df,
        directory=os.path.join(config.DATA_DIR, 'train'),
        x_col='filename',
        y_col='label',
        target_size=config.TARGET_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical'
    )

    val_generator = val_gen.flow_from_dataframe(
        dataframe=val_df,
        directory=os.path.join(config.DATA_DIR, 'train'),
        x_col='filename',
        y_col='label',
        target_size=config.TARGET_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical'
    )

    model = create_model(num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3),
        ModelCheckpoint(
            filepath=os.path.join(config.MODEL_SAVE_DIR, f'model_fold{fold}.h5'),
            save_best_only=True,
            monitor='val_accuracy'
        )
    ]

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # Save accuracy plot
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'Accuracy - Fold {fold}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plot_path = config.PLOT_SAVE_PATH.format(fold)
    plt.savefig(plot_path)
    print(f"📈 Saved accuracy plot: {plot_path}")
    plt.close()

if __name__ == "__main__":
    for fold in range(1, config.N_SPLITS + 1):
        train_one_fold(fold)
