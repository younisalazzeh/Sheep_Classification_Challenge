import pandas as pd
from sklearn.model_selection import train_test_split
import os
import config

# تحديد المسارات
data_dir = config.DATA_DIR
labels_file = config.TRAIN_LABELS_CSV

# قراءة ملف التسميات
df = pd.read_csv(labels_file)

# تقسيم البيانات إلى تدريب وتحقق
train_df, val_df = train_test_split(df, test_size=config.TEST_SIZE, stratify=df["label"], random_state=config.RANDOM_STATE)

print(f'عدد صور التدريب: {len(train_df)}')
print(f'عدد صور التحقق: {len(val_df)}')

# حفظ ملفات CSV الجديدة
train_df.to_csv(config.TRAIN_SPLIT_CSV, index=False)
val_df.to_csv(config.VAL_SPLIT_CSV, index=False)

print('تم حفظ ملفات train_split.csv و val_split.csv')


