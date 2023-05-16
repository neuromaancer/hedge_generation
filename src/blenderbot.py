import warnings
from ast import literal_eval
from pathlib import Path
from typing import Any, Optional

import pytorch_lightning as pl
import torch
from dotenv import dotenv_values
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.trainer import Trainer
from rich import print as rprint
from transformers import (
    BlenderbotSmallForConditionalGeneration,
    BlenderbotSmallTokenizer,
)
from dataset import HedgingDataModule
from utils import define_data_path, predict_and_save
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


rprint(f"We have {torch.cuda.device_count()} GPUs")
warnings.filterwarnings("ignore")
torch.cuda.empty_cache()

config = dotenv_values("../.env")
model_scale = "SMALL"
info_to_save = "Hsize_4"
# hyperparameters
MAX_LEN = int(config["MAX_LEN"])
BATCH_SIZE = int(config["BATCH_SIZE"])
MAX_EPOCHS = int(config["MAX_EPOCHS"])
NUM_WORKERS = int(config["NUM_WORKERS"])
LR = float(config["LR"])
DATA_TYPE_COMBINATION = literal_eval(config["DATA_TYPE_COMBINATION"])


# model configuration
FULL_TRAIN_DATASET_PATH = Path(config["FULL_TRAIN_DATASET_PATH"])
BALANCED_TRAIN_DATASET_PATH = Path(config["BALANCED_TRAIN_DATASET_PATH"])
FULL_TEST_DATASET_PATH = Path(config["FULL_TEST_DATASET_PATH"])
BALANCED_TEST_DATASET_PATH = Path(config["BALANCED_TEST_DATASET_PATH"])
SPECIAL_TOKENS = literal_eval(config["SPECIAL_TOKENS"])
# models' types
BLENDERBOT = config[f"BLENDERBOT_{model_scale}"]

# post parameters
PREDS_OUTPUT_DIR = config["PREDS_OUTPUT_DIR"]
MODELS_OUTPUT_DIR = config["MODELS_OUTPUT_DIR"]
TOKENIZER_OUTPUT_DIR = config["TOKENIZER_OUTPUT_DIR"]


class BlenderBotModel(pl.LightningModule):
    def __init__(self, lr, model_type, tokenizer) -> None:
        super(BlenderBotModel, self).__init__()
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.model = BlenderbotSmallForConditionalGeneration.from_pretrained(
            self.model_type
        )
        self.lr = lr

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def _step(self, batch: Any, batch_idx: int, mode: str = "train"):
        in_ids = batch["in_ids"]
        in_mask = batch["in_mask"]
        labels = batch["out_ids"]
        decoder_attention_mask = batch["out_mask"]

        outputs = self.model(
            input_ids=in_ids,
            attention_mask=in_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

        self.log(
            f"{mode}_loss",
            outputs.loss.mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return outputs.loss

    def training_step(self, batch: Any, batch_idx: int):
        return self._step(batch, batch_idx, mode="train")

    def test_step(self, batch: Any, batch_idx: int):
        return self._step(batch, batch_idx, mode="test")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="val")

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Any:
        predictions = []
        bos_token_id = self.tokenizer.bos_token_id
        outputs = self.model.generate(
            input_ids=batch["in_ids"],
            attention_mask=batch["in_mask"],
            early_stopping=True,
            min_length=1,
            max_length=10,
            no_repeat_ngram_size=2,
        )
        for input_, output in zip(batch["in_ids"], outputs):
            pred = (
                self.tokenizer.decode(
                    output,
                    clean_up_tokenization_spaces=True,
                )
                .replace("__unk__", " ")
                .replace("__end__", "")
                .replace("__start__", "")
                .replace("__null__ ", "")
                .strip()
            )
            predictions.append(pred)
        return predictions


if __name__ == "__main__":
    # Blenderbot
    model_type = BLENDERBOT
    model_name = f"blenderbot_{model_scale.lower()}"
    #! need to be changed sometime from {test, final}
    training_type = "final"
    data_combination = DATA_TYPE_COMBINATION
    data_combination_str = (
        f"train_{data_combination['train']}_test_{data_combination['test']}"
    )
    train_data_path, test_data_path = define_data_path(data_combination)

    tokenizer = BlenderbotSmallTokenizer.from_pretrained(
        model_type, truncation=True, do_lower_case=True
    )
    # add special tokens to the tokenizer
    special_tokens = SPECIAL_TOKENS
    if isinstance(special_tokens, str):
        special_tokens = literal_eval(special_tokens)
    # special_tokens.append("\n")
    special_tokens_dict = {"additional_special_tokens": special_tokens}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    data_module = HedgingDataModule(
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        batch_size=BATCH_SIZE,
        max_len=MAX_LEN,
        num_workers=NUM_WORKERS,
        tokenizer=tokenizer,
    )
    generator = BlenderBotModel(model_type=model_type, tokenizer=tokenizer, lr=LR)
    generator.model.resize_token_embeddings(len(tokenizer))
    gen_ckpt = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="val_loss_{val_loss:.2f}_epoch_{epoch:02d}",
        auto_insert_metric_name=False,
    )
    tb_logger = pl_loggers.TensorBoardLogger(
        f"../logs/{model_name}/",
        name=f"type_{training_type}_datatype_{data_combination_str}",
    )
    gen_trainer = Trainer(
        gpus=-1,
        logger=tb_logger,
        max_epochs=10,
        callbacks=[gen_ckpt, EarlyStopping(monitor="val_loss", mode="min")],
        # limit_train_batches=1,
        # limit_val_batches=2,  # debug parameters
    )
    gen_trainer.tune(generator)
    gen_trainer.fit(generator, data_module)

    # save the model with huggingface save_pretrained
    model_to_save = Path(f"{MODELS_OUTPUT_DIR}/{model_name}/{training_type}/model/")
    model_to_save.mkdir(parents=True, exist_ok=True)
    tokenizer_to_save = Path(f"{TOKENIZER_OUTPUT_DIR}/{model_name}/")
    tokenizer_to_save.mkdir(parents=True, exist_ok=True)

    generator.model.save_pretrained(model_to_save)
    tokenizer.save_pretrained(tokenizer_to_save)
    path_to_save = Path(f"{PREDS_OUTPUT_DIR}/{model_name}")
    path_to_save.mkdir(parents=True, exist_ok=True)
    predict_and_save(
        generator=generator,
        trainer=gen_trainer,
        model_name=model_name,
        data_module=data_module,
        path_to_save=path_to_save,
        info_to_save=info_to_save,
        type=training_type,
    )
