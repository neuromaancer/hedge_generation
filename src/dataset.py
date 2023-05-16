import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
import pandas as pd
from transformers import PreTrainedTokenizer
from typing import Optional
from sklearn.model_selection import train_test_split
from rich import print as rprint


class HedgingGenDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: PreTrainedTokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.data = df
        self.text = ("<tutor> " + self.data.text).values
        # self.text = self.data.text.values
        self.history = self.data.history.values
        self.label = self.data.label.values
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        history = str(self.history[index])
        history = " ".join(history.split())
        in_ = self.tokenizer.encode_plus(
            history,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True,
        )

        text = str(self.text[index])
        text = " ".join(text.split())
        out_ = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True,
        )

        in_ids = in_["input_ids"]
        in_mask = in_["attention_mask"]
        in_token_type_ids = in_["token_type_ids"]

        out_ids = out_["input_ids"]
        out_mask = out_["attention_mask"]

        return {
            "in_ids": torch.tensor(in_ids, dtype=torch.long),
            "in_mask": torch.tensor(in_mask, dtype=torch.long),
            "in_token_type_ids": torch.tensor(in_token_type_ids, dtype=torch.long),
            "out_ids": torch.tensor(out_ids, dtype=torch.long),
            "out_mask": torch.tensor(out_mask, dtype=torch.long),
            "label": torch.tensor(self.label[index], dtype=torch.long),
        }


class HedgingDataModule(LightningDataModule):
    def __init__(
        self,
        train_data_path: str,
        test_data_path: str,
        batch_size: int,
        max_len: int,
        num_workers: int,
        tokenizer: PreTrainedTokenizer,
    ) -> None:
        super().__init__()
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        self.prepare_data()

    def prepare_data(self) -> None:
        self.data = pd.read_csv(self.train_data_path)
        self.test_data = pd.read_csv(self.test_data_path)

    def setup(self, stage: Optional[str] = None) -> None:
        # Create Training, Validation and Test Datasets

        train_data, val_data = train_test_split(
            self.data, test_size=0.25, random_state=8
        )
        # Create Different Datasets
        self.train_set = HedgingGenDataset(train_data, self.tokenizer, self.max_len)
        self.val_set = HedgingGenDataset(val_data, self.tokenizer, self.max_len)
        self.test_set = HedgingGenDataset(self.test_data, self.tokenizer, self.max_len)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set, batch_size=1, shuffle=False, num_workers=self.num_workers,
        )


class ConversationDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, df: pd.DataFrame, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.df = df
        df["input"] = df["history"] + " <tutor> " + df["text"] + " <|endoftext|>"
        self.history = (df["history"] + " <tutor> ").values
        rprint(self.history)
        self.input = df["input"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        input_ = str(self.input[index])
        history_ = str(self.history[index])
        input_ = " ".join(input_.split())
        history_ = " ".join(history_.split())
        in_ = self.tokenizer.encode_plus(
            input_,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )
        prompt = self.tokenizer.encode_plus(
            history_,
            None,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
        )

        in_ids = in_["input_ids"]
        in_attention_mask = in_["attention_mask"]
        prompt_ids = prompt["input_ids"]
        prompt_attention_mask = prompt["attention_mask"]

        return {
            "in_ids": torch.tensor(in_ids, dtype=torch.long),
            "in_attention_mask": torch.tensor(in_attention_mask, dtype=torch.long),
            "his_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "his_attention_mask": torch.tensor(prompt_attention_mask, dtype=torch.long),
        }


class ConversationalDataModule(LightningDataModule):
    def __init__(
        self,
        train_data_path: str,
        test_data_path: str,
        batch_size: int,
        max_len: int,
        num_workers: int,
        tokenizer: PreTrainedTokenizer,
    ) -> None:
        super().__init__()
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        self.prepare_data()

    def prepare_data(self) -> None:
        self.data = pd.read_csv(self.train_data_path)
        self.test_data = pd.read_csv(self.test_data_path)

    def setup(self, stage: Optional[str] = None) -> None:
        # Create Training, Validation and Test Datasets

        train_data, val_data = train_test_split(
            self.data, test_size=0.25, random_state=8
        )
        # Create Different Datasets
        self.train_set = ConversationDataset(self.tokenizer, train_data, self.max_len)
        self.val_set = ConversationDataset(self.tokenizer, val_data, self.max_len)
        self.test_set = ConversationDataset(
            self.tokenizer, self.test_data, self.max_len
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set, batch_size=1, shuffle=False, num_workers=self.num_workers,
        )
