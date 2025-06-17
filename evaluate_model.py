import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import os
import config

# تحديد المسارات
data_dir = config.DATA_DIR

# تحليل كل طية من النماذج المدربة
for fold in range(1, config.N_SPLITS + 1):
    print(f"\n📊 Evaluating Fold {fold} Model")

    val_labels_path = config.FOLD_CSV_TEMPLATE.format(fold, 'val')
    model_path = os.path.join(config.MODEL_SAVE_DIR, f'model_fold{fold}.h5')

    # قراءة ملف التحقق لهذا الطي
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
        shuffle=False
    )

    # الحصول على التنبؤات
    Y_pred = model.predict(validation_generator)
    y_pred_classes = np.argmax(Y_pred, axis=1)

    # الحصول على الفئات الحقيقية
    y_true_classes = validation_generator.classes
    class_labels = list(validation_generator.class_indices.keys())

    # تقرير التصنيف
    report = classification_report(y_true_classes, y_pred_classes, target_names=class_labels)
    print("تقرير التصنيف:")
    print(report)

    # مصفوفة الارتباك
    conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
    print("مصفوفة الارتباك:")
    print(conf_matrix)

    # حفظ تقرير التصنيف
    report_file = f"classification_report_fold_{fold}.txt"
    with open(os.path.join(config.DATA_DIR, report_file), "w") as f:
        f.write(report)

    print(f"✅ تم حفظ التقرير في {report_file}")
