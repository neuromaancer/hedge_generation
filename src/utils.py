from pathlib import Path
from dotenv import dotenv_values
from pytorch_lightning import Trainer, LightningDataModule
from rich import print as rprint
from typing import Dict, List
from datetime import datetime
import pandas as pd

from transformers import PreTrainedModel

config = dotenv_values("../.env")

FULL_TRAIN_DATASET_PATH = Path(config["FULL_TRAIN_DATASET_PATH"])
BALANCED_TRAIN_DATASET_PATH = Path(config["BALANCED_TRAIN_DATASET_PATH"])
FULL_TEST_DATASET_PATH = Path(config["FULL_TEST_DATASET_PATH"])
BALANCED_TEST_DATASET_PATH = Path(config["BALANCED_TEST_DATASET_PATH"])


def define_data_path(data_combination: Dict):
    if data_combination["train"] == "full":
        train_data_path = FULL_TRAIN_DATASET_PATH
    elif data_combination["train"] == "balanced":
        train_data_path = BALANCED_TRAIN_DATASET_PATH
    if data_combination["test"] == "full":
        test_data_path = FULL_TEST_DATASET_PATH
    elif data_combination["test"] == "balanced":
        test_data_path = BALANCED_TEST_DATASET_PATH
    return train_data_path, test_data_path


def predict_and_save(
    generator: PreTrainedModel,
    trainer: Trainer,
    model_name: str,
    data_module: LightningDataModule,
    path_to_save: Path,
    info_to_save: str,
    type: str = "test",
):
    predictions = trainer.predict(generator, data_module)
    prompts = pd.read_csv(data_module.test_data_path).history.values.tolist()
    now = datetime.now().strftime("%m_%d_%Y_%H-%M-%S")
    with open(
        path_to_save / f"{model_name}_predicts_{now}_{type}_{info_to_save}.txt", "w+"
    ) as f:
        for pred, history in zip(predictions, prompts):
            # remove the prompt
            f.write(
                f"{pred[0].replace(history, '').replace('<tutor>', '').replace('<tutee>', '').strip()}\n"
            )


def clean_text(text: str):
    text = text.replace("\n", "")
    return text.replace('"', "")


def combine_preds_targets(
    test_data_path: str, preds_file: str, preds_labels
) -> pd.DataFrame:
    test_df = pd.read_csv(test_data_path)
    with open(preds_file, "r") as f:
        test_df["preds"] = list(map(clean_text, f.readlines()))
    test_df["preds_labels"] = preds_labels
    return test_df.copy()
