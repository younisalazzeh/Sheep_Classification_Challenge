import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import os
import config

# تحديد المسارات
data_dir = config.DATA_DIR
val_labels_path = config.VAL_SPLIT_CSV
model_path = config.MODEL_SAVE_PATH

# قراءة ملفات التسميات المقسمة
val_df = pd.read_csv(val_labels_path)

# تحميل النموذج المدرب
model = tf.keras.models.load_model(model_path)

# إعداد مولد البيانات للتحقق
val_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=os.path.join(data_dir, "train"),  # مسار الصور للتحقق
    x_col="filename",
    y_col="label",
    target_size=config.TARGET_SIZE,
    batch_size=config.BATCH_SIZE,
    class_mode="categorical",
    shuffle=False  # مهم لتقييم الأداء
)

# الحصول على التنبؤات
Y_pred = model.predict(validation_generator)
y_pred_classes = np.argmax(Y_pred, axis=1)

# الحصول على الفئات الحقيقية
y_true_classes = validation_generator.classes

# الحصول على أسماء الفئات
class_labels = list(validation_generator.class_indices.keys())

# طباعة تقرير التصنيف
report = classification_report(y_true_classes, y_pred_classes, target_names=class_labels)
print("تقرير التصنيف:")
print(report)

# طباعة مصفوفة الارتباك
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
print("مصفوفة الارتباك:")
print(conf_matrix)

# حفظ تقرير التصنيف في ملف
with open(config.CLASSIFICATION_REPORT_FILE, "w") as f:
    f.write(report)

print("تم حفظ تقرير التصنيف في classification_report.txt")


