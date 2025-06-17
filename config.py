import os

# مسارات البيانات
DATA_DIR = r'/content/Arabian_Sheep_Image_Classification_Challenge/Sheep_Classification_Images/'
TRAIN_LABELS_CSV = os.path.join(DATA_DIR, 'train_labels.csv')
TRAIN_SPLIT_CSV = os.path.join(DATA_DIR, 'train_split.csv')
VAL_SPLIT_CSV = os.path.join(DATA_DIR, 'val_split.csv')
AUGMENTED_DATA_DIR = os.path.join(DATA_DIR, "train_augmented")

# class name
CLASS_NAMES = ['Barbari', 'Najdi', 'Harri', "Sawakni", "Roman","Goat","Naeimi"]  # Example class names

# Output fold CSVs
FOLD_CSV_TEMPLATE = os.path.join(DATA_DIR, 'fold_{}_{}.csv')  # Format: fold_1_train.csv or fold_1_val.csv

# Image size and batch
TARGET_SIZE = (150, 150)  # Compatible with transfer learning models
BATCH_SIZE = 32
EPOCHS = 25
N_SPLITS = 5  # Number of K folds

# Model save path
MODEL_SAVE_DIR = os.path.join(DATA_DIR, 'models')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Accuracy plots
PLOT_SAVE_PATH = os.path.join(DATA_DIR, 'accuracy_plot_fold_{}.png')


# مسارات حفظ النموذج والتقارير
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "sheep_classifier_model.h5")
CLASSIFICATION_REPORT_FILE = 'classification_report.txt'


