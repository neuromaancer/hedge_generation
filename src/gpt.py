import warnings
from ast import literal_eval
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from dotenv import dotenv_values
from pytorch_lightning import LightningDataModule, loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.trainer import Trainer
from rich import print as rprint
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
)
from dataset import ConversationalDataModule
from utils import define_data_path, combine_preds_targets
import re



rprint(f"We have {torch.cuda.device_count()} GPUs")
warnings.filterwarnings("ignore")
torch.cuda.empty_cache()
model_scale = "SMALL"
info_to_save = "Hsize_4"

config = dotenv_values("../.env")
# hyperparameters
MAX_LEN = int(config["MAX_LEN"])
BATCH_SIZE = int(config["BATCH_SIZE"])
MAX_EPOCHS = int(config["MAX_EPOCHS"])
NUM_WORKERS = int(config["NUM_WORKERS"])
LR = float(config["LR"])
DATA_TYPE_COMBINATION = literal_eval(config["DATA_TYPE_COMBINATION"])
COMBINED_PREDS_OUTPUT_DIR = Path(config["COMBINED_PREDS_OUTPUT_DIR"])

# model configuration
SPECIAL_TOKENS = literal_eval(config["SPECIAL_TOKENS"])
# models' types
GPT2 = config["GPT2"]
DIALOGPT = config[f"DIALOGPT_{model_scale}"]

# post parameters
PREDS_OUTPUT_DIR = config["PREDS_OUTPUT_DIR"]
MODELS_OUTPUT_DIR = config["MODELS_OUTPUT_DIR"]
TOKENIZER_OUTPUT_DIR = config["TOKENIZER_OUTPUT_DIR"]

train_data_path, test_data_path = define_data_path(DATA_TYPE_COMBINATION)


class GPTModel(pl.LightningModule):
    def __init__(
        self, lr: float, model_type: str, tokenizer: PreTrainedTokenizer
    ) -> None:
        super(GPTModel, self).__init__()
        self.save_hyperparameters()
        self.model_type = model_type
        self.model = AutoModelForCausalLM.from_pretrained(self.model_type)
        self.lr = lr
        self.tokenizer = tokenizer

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def _step(self, batch: Any, batch_idx: int, mode: str = "train"):
        in_ids = batch["in_ids"]
        attention_mask = batch["in_attention_mask"]

        outputs = self.model(
            input_ids=in_ids, attention_mask=attention_mask, labels=in_ids
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

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="val")

    def predict_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Any:
        predictions = []
        outputs = self.model.generate(
            input_ids=batch["his_ids"],
            attention_mask=batch["his_attention_mask"],
            early_stopping=True,
            max_length=500,
            no_repeat_ngram_size=2,
        )
        for output in outputs:
            pred = self.tokenizer.decode(
                output,
                clean_up_tokenization_spaces=True,
            ).replace("<|endoftext|>", "")
            predictions.append(pred)
        return predictions


def predict_and_save(
    generator: GPTModel,
    trainer: Trainer,
    model_name: str,
    data_module: LightningDataModule,
    folder_to_save: str,
    info_to_save: str,
    type: str = "test",
    testset_type: str = "balanced",
):
    predictions = trainer.predict(generator, data_module)
    prompts = pd.read_csv(data_module.test_data_path).history.values.tolist()
    now = datetime.now().strftime("%m_%d_%Y_%H-%M-%S")
    to_save = f"{folder_to_save}/preds_{model_name}_test-type_{testset_type}__{now}_{type}_{info_to_save}.txt"
    with open(to_save, "w+") as f:
        for pred, history in zip(predictions, prompts):
            # remove the prompt
            history = re.sub(" +", " ", history)
            history = history.replace(" .", ".")
            output = (
                pred[0]
                .replace(history, "")
                .replace("<tutor>", "")
                .replace("<tutee>", "")
                .strip()
            )
            # rprint("[blue]result: ", pred[0])
            # rprint("[green]history: ", history)
            # rprint("[red]output: ", output)
            f.write(f"{output}\n")
    return to_save


if __name__ == "__main__":
    # DIALOGPT
    model_type = DIALOGPT
    model_name = "dialogpt"
    training_type = "final"
    data_combination = DATA_TYPE_COMBINATION
    data_combination_str = (
        f"train_{data_combination['train']}_test_{data_combination['test']}"
    )

    train_data_path, test_data_path = define_data_path(data_combination)
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    #! gpt model needs *left* padding not *right* padding
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    # add special tokens to the tokenizer
    special_tokens = SPECIAL_TOKENS
    if isinstance(special_tokens, str):
        special_tokens = literal_eval(special_tokens)
    special_tokens_dict = {"additional_special_tokens": special_tokens}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    # prepare the dataset
    data_module = ConversationalDataModule(
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        batch_size=BATCH_SIZE,
        max_len=MAX_LEN,
        num_workers=NUM_WORKERS,
        tokenizer=tokenizer,
    )
    # test for data_module
    # data_module.prepare_data()
    # data_module.setup()
    # rprint("history", data_module.train_set.history)
    # rprint("input", data_module.train_set.input)
    # input("Press Enter to continue...")
    # initialize the model
    generator = GPTModel(lr=LR, model_type=model_type, tokenizer=tokenizer)
    generator.model.resize_token_embeddings(len(tokenizer))
    gen_ckpt = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="val_loss_{val_loss:.2f}_epoch_{epoch:02d}",
        auto_insert_metric_name=False,
    )
    logger = pl_loggers.TensorBoardLogger(
        f"../logs/{model_name}/",
        name=f"type_{training_type}_datatype_{data_combination_str}",
    )
    trainer = Trainer(
        gpus=-1,
        precision=16,
        logger=logger,
        max_epochs=MAX_EPOCHS,
        callbacks=[gen_ckpt, EarlyStopping(monitor="val_loss", mode="min")],
        # limit_train_batches=1,
        # limit_val_batches=2,  # debug parameters
    )
    trainer.tune(generator)
    trainer.fit(generator, data_module)

    # save the model with huggingface save_pretrained
    model_to_save = Path(f"{MODELS_OUTPUT_DIR}/{model_name}/{training_type}/model/")
    model_to_save.mkdir(parents=True, exist_ok=True)
    tokenizer_to_save = Path(f"{TOKENIZER_OUTPUT_DIR}/{model_name}/")
    tokenizer_to_save.mkdir(parents=True, exist_ok=True)

    generator.model.save_pretrained(model_to_save)
    tokenizer.save_pretrained(tokenizer_to_save)
    folder_to_save = Path(f"{PREDS_OUTPUT_DIR}/{model_name}")
    folder_to_save.mkdir(parents=True, exist_ok=True)
    preds_file_path = predict_and_save(
        generator=generator,
        trainer=trainer,
        model_name=model_name,
        data_module=data_module,
        folder_to_save=folder_to_save,
        type=training_type,
        testset_type=data_combination["test"],
        info_to_save=info_to_save,
    )
    now = datetime.now()
    now = now.strftime("%m_%d_%Y_%H-%M-%S")
    combine_result_df = combine_preds_targets(
        test_data_path=test_data_path, preds_file=preds_file_path
    )
    combine_path = Path(f"{COMBINED_PREDS_OUTPUT_DIR}/dialogpt_{model_scale.lower()}")
    combine_path.mkdir(parents=True, exist_ok=True)
    combine_result_df.to_csv(
        f"{COMBINED_PREDS_OUTPUT_DIR}/dialogpt_{model_scale.lower()}/{DATA_TYPE_COMBINATION['test']}_predictions_{now}_{info_to_save}.csv",
        index=False,
    )
