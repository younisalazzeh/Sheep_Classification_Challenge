import os

# مسارات البيانات
DATA_DIR = r'Sheep_Classification_Images\\'
TRAIN_LABELS_CSV = os.path.join(DATA_DIR, 'train_labels.csv')
TRAIN_SPLIT_CSV = os.path.join(DATA_DIR, 'train_split.csv')
VAL_SPLIT_CSV = os.path.join(DATA_DIR, 'val_split.csv')
AUGMENTED_DATA_DIR = os.path.join(DATA_DIR, "train_augmented")

# data augmentation parms
AUGMENTATION_PARAMS = {
    "rotation_range": 20,
    "width_shift_range": 0.2,
    "height_shift_range": 0.2,
    "shear_range": 0.2,
    "zoom_range": 0.2,
    "horizontal_flip": True,
    "brightness_range": [0.8, 1.2],
    "fill_mode": "nearest"
}


# إعداد تقسيم البيانات
TEST_SIZE = 0.2
RANDOM_STATE = 42

# إعدادات النموذج
TARGET_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 20

# مسارات حفظ النموذج والتقارير
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "sheep_classifier_model.h5")
CLASSIFICATION_REPORT_FILE = 'classification_report.txt'


