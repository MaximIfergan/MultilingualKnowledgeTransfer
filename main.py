import MLCBQA_Model
import pandas as pd

if __name__ == "__main__":

    model_params = {
        "MODEL": "mt5-base",
        "MODEL_DIR": "/cs/labs/oabend/maximifergan/MKT/outputs/mT5-base-2-epochs/model_files/",
        "TRAIN_BATCH_SIZE": 8,
        "VALID_BATCH_SIZE": 8,
        "TRAIN_EPOCHS": 4,
        "LEARNING_RATE": 1e-4,
        "MAX_SOURCE_TEXT_LENGTH": 396,
        "MAX_TARGET_TEXT_LENGTH": 32,
        "SEED": 18,
    }

    df = pd.read_csv('Datasets/PreprocessDataset.csv')

    MLCBQA_Model.MT5Trainer(
        dataframe=df,
        source_text="Question",
        target_text="Answer",
        model_params=model_params,
        output_dir="outputs",
    )
