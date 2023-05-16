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
from transformers import BartForConditionalGeneration, BartTokenizer
from dataset import HedgingDataModule
from utils import define_data_path, predict_and_save

rprint(f"We have {torch.cuda.device_count()} GPUs")
warnings.filterwarnings("ignore")
torch.cuda.empty_cache()

config = dotenv_values("../.env")
model_scale="BASE"
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

BART = config[f"BART_{model_scale}"]

# post parameters
PREDS_OUTPUT_DIR = config["PREDS_OUTPUT_DIR"]
MODELS_OUTPUT_DIR = config["MODELS_OUTPUT_DIR"]
TOKENIZER_OUTPUT_DIR = config["TOKENIZER_OUTPUT_DIR"]

BAD_WORDS = literal_eval(config["BAD_WORDS"])



class ReRankBART(pl.LightningModule):
    def __init__(self, lr, model_type, tokenizer) -> None:
        super(ReRankBART, self).__init__()
        self.save_hyperparameters()
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.lr = lr
        self.bart = BartForConditionalGeneration.from_pretrained(self.model_type)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def _step(self, batch: Any, batch_idx: int, mode: str = "train"):
        in_ids = batch["in_ids"]
        in_mask = batch["in_mask"]
        labels = batch["out_ids"]
        labels[labels[:, :] == self.bart.model.config.pad_token_id] = -100
        decoder_attention_mask = batch["out_mask"]

        outputs = self.bart(
            input_ids=in_ids,
            attention_mask=in_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )
        loss = outputs.loss
        self.log(
            f"{mode}_loss",
            loss.mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

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
        outputs = self.bart.generate(
            input_ids=batch["in_ids"],
            attention_mask=batch["in_mask"],
            early_stopping=True,
            max_length=500,
        )
        for output in outputs:
            pred = self.tokenizer.decode(
                output, clean_up_tokenization_spaces=True,
            ).replace("<s>", "").replace("</s>", "").strip()
            predictions.append(pred)
        return predictions


if __name__ == "__main__":
    # BART
    model_type = BART
    
    model_name = f"bart_{model_scale.lower()}"
    #! need to be changed sometime from {test, final}
    training_type = "final"
    data_combination = DATA_TYPE_COMBINATION
    data_combination_str = (
        f"train_{data_combination['train']}_test_{data_combination['test']}"
    )
    train_data_path, test_data_path = define_data_path(data_combination)

    bart_tokenizer = BartTokenizer.from_pretrained(
        model_type, truncation=True, do_lower_case=True
    )
    # add special tokens to the tokenizer
    special_tokens = SPECIAL_TOKENS
    if isinstance(special_tokens, str):
        special_tokens = literal_eval(special_tokens)
    special_tokens_dict = {"additional_special_tokens": special_tokens}
    num_added_toks = bart_tokenizer.add_special_tokens(special_tokens_dict)

    data_module = HedgingDataModule(
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        batch_size=BATCH_SIZE,
        max_len=MAX_LEN,
        num_workers=NUM_WORKERS,
        tokenizer=bart_tokenizer,
    )
    generator = ReRankBART(model_type=model_type, tokenizer=bart_tokenizer, lr=LR)
    generator.bart.resize_token_embeddings(len(bart_tokenizer))
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
    bartgen_trainer = Trainer(
        gpus=-1,
        logger=tb_logger,
        max_epochs=10,
        callbacks=[gen_ckpt, EarlyStopping(monitor="val_loss", mode="min")],
        # limit_train_batches=1,
        # limit_val_batches=2,  # debug parameters
    )
    bartgen_trainer.tune(generator)
    bartgen_trainer.fit(generator, data_module)

    # save the model with huggingface save_pretrained
    model_to_save = Path(f"{MODELS_OUTPUT_DIR}/{model_name}/{training_type}/model/")
    model_to_save.mkdir(parents=True, exist_ok=True)
    tokenizer_to_save = Path(
        f"{TOKENIZER_OUTPUT_DIR}/{model_name}/"
    )
    tokenizer_to_save.mkdir(parents=True, exist_ok=True)

    generator.bart.save_pretrained(model_to_save)
    bart_tokenizer.save_pretrained(tokenizer_to_save)
    path_to_save = Path(f"{PREDS_OUTPUT_DIR}/{model_name}")
    path_to_save.mkdir(parents=True, exist_ok=True)
    predict_and_save(
        generator=generator,
        trainer=bartgen_trainer,
        model_name=model_name,
        data_module=data_module,
        path_to_save=path_to_save,
        info_to_save=info_to_save,
        type=training_type,
    )

