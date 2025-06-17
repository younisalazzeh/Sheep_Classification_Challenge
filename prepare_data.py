import pandas as pd
from sklearn.model_selection import StratifiedKFold
import config
import os

def create_stratified_folds():
    df = pd.read_csv(config.TRAIN_LABELS_CSV)

    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df['filename'], df['label']), 1):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_path = config.FOLD_CSV_TEMPLATE.format(fold, 'train')
        val_path = config.FOLD_CSV_TEMPLATE.format(fold, 'val')

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)

        print(f'✅ Fold {fold} - Train: {len(train_df)}, Val: {len(val_df)}')

if __name__ == "__main__":
    create_stratified_folds()
