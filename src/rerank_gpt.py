from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch
from rich import print as rprint
import pandas as pd
from datetime import datetime
from dotenv import dotenv_values
from pathlib import Path
from utils import define_data_path, combine_preds_targets
from ast import literal_eval
import re

# configuration
config = dotenv_values("../.env")
TOKENIZER_OUTPUT_DIR = Path(config["TOKENIZER_OUTPUT_DIR"])
MODELS_OUTPUT_DIR = Path(config["MODELS_OUTPUT_DIR"])
DATA_TYPE_COMBINATION = literal_eval(config["DATA_TYPE_COMBINATION"])
BAD_WORDS = literal_eval(config["BAD_WORDS"])
# BAD_WORDS.append("<laughter>")
BERT_CLF = config["BERT_CLF"]
PREDS_OUTPUT_DIR = Path(config["PREDS_OUTPUT_DIR"])
COMBINED_PREDS_OUTPUT_DIR = Path(config["COMBINED_PREDS_OUTPUT_DIR"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data_path, test_data_path = define_data_path(DATA_TYPE_COMBINATION)

tokenizer = AutoTokenizer.from_pretrained(f"{TOKENIZER_OUTPUT_DIR}/dialogpt/")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

generator_model = AutoModelForCausalLM.from_pretrained(
    f"{MODELS_OUTPUT_DIR}/dialogpt/final/model", return_dict_in_generate=True
)
clf_model = BertForSequenceClassification.from_pretrained(BERT_CLF)

pred_mode = 2
# pred_mode = 4
test_df = pd.read_csv(test_data_path)
predictions = []
preds_labels = []
bad_words_ids = [tokenizer(bad_word).input_ids for bad_word in BAD_WORDS]
len_ = len(test_df)
count_wrong = 0

clf_model.eval()
with torch.no_grad():
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with (
        open(
            f"{PREDS_OUTPUT_DIR}/wrongs/rerank_dialogpt/{DATA_TYPE_COMBINATION['test']}_wrongs_rerank_dialogpt_{now}.csv",
            "w+",
        ) as f1,
        open(
            f"{PREDS_OUTPUT_DIR}/rights/rerank_dialogpt/{DATA_TYPE_COMBINATION['test']}_rights_rerank_dialogpt_{now}.csv",
            "w+",
        ) as f2,
    ):
        for idx, row in test_df.iterrows():
            history = row["history"]
            text = row["text"]
            label = row["label"]
            rprint(history)
            # The context of the sentence.
            # prompt = tokenizer.encode_plus(
            #     history + " <tutor> ",
            #     None,
            #     add_special_tokens=True,
            #     max_length=330,
            #     padding="max_length",
            #     return_token_type_ids=True,
            #     truncation=True,
            # )
            prompt = tokenizer(history + " <tutor> ", add_special_tokens=True)
            prompt_size = len(prompt["input_ids"])
            outputs = generator_model.generate(
                input_ids=torch.tensor([prompt["input_ids"]], dtype=torch.long),
                attention_mask=torch.tensor(
                    [prompt["attention_mask"]], dtype=torch.long
                ),
                early_stopping=True,
                num_beams=100,
                # do_sample=True,
                # top_p=0.95,
                # typical_p=0.95,
                bad_words_ids=bad_words_ids,
                no_repeat_ngram_size=2,
                repetition_penalty=1.5,
                num_return_sequences=100,
                max_length=prompt_size + 10,
                output_scores=True,
                min_length=prompt_size + 1,
            )
            # rprint(outputs)
            # rprint(outputs.scores[0][0].size())
            # rprint(len(outputs.scores))
            pred_text = ""
            # zipped = zip(outputs.sequences_scores, outputs.sequences)
            zipped = zip(outputs.scores, outputs.sequences)
            # sorted_res = sorted(zipped, key=lambda x: x[0], reverse=True)
            sorted_res = list(zipped)
            lm_rank_1 = ""
            lm_rank_1_label = 0
            for rank, (sentence_score, sequence) in enumerate(sorted_res):
                # rprint(tokenizer.decode(
                #             sequence, clean_up_tokenization_spaces=True
                #         ))
                # history = re.sub(" +", " ", history)
                # history = history.replace(" .", ".")
                generated = tokenizer.decode(sequence)

                generated = (
                    generated.replace("<|endoftext|>", "")
                    # .replace("  ", " ")
                    # .replace("  ", " ")
                )
                # generated = re.sub(" +", " ", generated)
                rprint("[blue]generated: ", generated)
                # rprint("[cyan]history: ", history)
                # if history in generated:
                #     rprint("[red]history in generated")
                # else:
                #     rprint("history [green]not in generated")
                candidate = (
                    generated.replace(history, "")
                    .replace("  ", " ")
                    .replace("  ", " ")
                    .replace("  ", " ")
                    .replace("  ", " ")
                    .replace("<tutor>", "")
                    # .replace("<tutor>", "")
                    .replace("<tutee>", "")
                    # .replace("<laughter>", "")
                    # .replace("  ", " ")
                    .strip()
                )
                rprint("[yellow]candidate: ", candidate)
                if rank == 0:
                    lm_rank_1 = candidate
                clf_inputs = bert_tokenizer(candidate, return_tensors="pt")
                clf_outputs = clf_model(**clf_inputs)
                pred = clf_outputs.logits.argmax(dim=1, keepdim=True).item()
                if rank == 0:
                    lm_rank_1_label = pred
                # convert the 4 classes predictions to 2 classes predictions
                if (pred_mode == 2) and (int(pred) != 0):
                    pred = 1
                if int(pred) == int(label):
                    pred_text = candidate
                    f2.write(str(idx + 2))  # line number starts from 2
                    f2.write(",")
                    f2.write(history)
                    rprint("[red]history: ", history)
                    f2.write(",")
                    f2.write(text)
                    f2.write(",")
                    f2.write(pred_text)
                    rprint("[blue]pred_text: ", pred_text)
                    f2.write(",")
                    f2.write(str(label))
                    f2.write("\n")
                    preds_labels.append(pred)
                    break
            if pred_text == "":
                count_wrong += 1
                # if no corresponding prediction is found, pick the a first best one
                pred_text = lm_rank_1
                # write wrongs to log file
                f1.write(str(idx + 2))  # line number starts from 2
                f1.write(",")
                f1.write(history)
                f1.write(",")
                f1.write(text)
                f1.write(",")
                f1.write(pred_text)
                f1.write(",")
                f1.write(str(label))
                f1.write("\n")
                rprint(
                    f"[red]{round(count_wrong/len_, ndigits=3)*100}% lines with wrong labels"
                )
                rprint(f"text with wrong label: [bright_blue]{pred_text}")
                rprint(f"right label: [blue]{label}")
                rprint(f"target text: [cyan]{text}")
                preds_labels.append(lm_rank_1_label)
            predictions.append(pred_text)
            rprint(f"[green]{idx+1}/{len_} is finished")

now = datetime.now()
now = now.strftime("%m_%d_%Y_%H-%M-%S")
preds_file_path = f"{PREDS_OUTPUT_DIR}/rerank_dialogpt/{DATA_TYPE_COMBINATION['test']}_predictions_{now}.txt"
with open(
    preds_file_path,
    "w+",
) as f:
    for pred in predictions:
        f.write(pred + "\n")

combine_result_df = combine_preds_targets(
    test_data_path=test_data_path, preds_file=preds_file_path, preds_labels=preds_labels
)

combine_result_df.to_csv(
    f"{COMBINED_PREDS_OUTPUT_DIR}/rerank_dialogpt/{DATA_TYPE_COMBINATION['test']}_predictions_{now}.csv",
    index=False,
)
