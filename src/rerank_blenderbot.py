from transformers import (
    BlenderbotSmallForConditionalGeneration,
    BertForSequenceClassification,
    BlenderbotSmallTokenizer,
    BertTokenizer,
)
import torch
from rich import print as rprint
import pandas as pd
from datetime import datetime
from dotenv import dotenv_values
from pathlib import Path
from utils import define_data_path, combine_preds_targets
from ast import literal_eval

# configuration
config = dotenv_values("../.env")
TOKENIZER_OUTPUT_DIR = Path(config["TOKENIZER_OUTPUT_DIR"])
MODELS_OUTPUT_DIR = Path(config["MODELS_OUTPUT_DIR"])
DATA_TYPE_COMBINATION = literal_eval(config["DATA_TYPE_COMBINATION"])
BAD_WORDS = literal_eval(config["BAD_WORDS"])
BERT_CLF = config["BERT_CLF"]
PREDS_OUTPUT_DIR = Path(config["PREDS_OUTPUT_DIR"])
COMBINED_PREDS_OUTPUT_DIR = Path(config["COMBINED_PREDS_OUTPUT_DIR"])
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_data_path, test_data_path = define_data_path(DATA_TYPE_COMBINATION)
info_to_save = "beam_search_blenderbot_small"

## * debug mode
# test_data_path = "/scratch2/aabulimiti/hg/HedgingGeneration/data/testset/test_tmp.csv"

tokenizer = BlenderbotSmallTokenizer.from_pretrained(
    f"{TOKENIZER_OUTPUT_DIR}/blenderbot_small/"
)
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

generator_model = BlenderbotSmallForConditionalGeneration.from_pretrained(
    f"{MODELS_OUTPUT_DIR}/blenderbot_small/final/model", return_dict_in_generate=True
)
clf_model = BertForSequenceClassification.from_pretrained(BERT_CLF)

# 2 classes vs 4 classes
pred_mode = 2
# pred_mode = 4
test_df = pd.read_csv(test_data_path)
predictions = []
bad_words_ids = [tokenizer(bad_word).input_ids for bad_word in BAD_WORDS]
len_ = len(test_df)
count_wrong = 0


clf_model.eval()
with torch.no_grad():
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with (
        open(
            f"{PREDS_OUTPUT_DIR}/wrongs/rerank_blenderbot/{DATA_TYPE_COMBINATION['test']}_wrongs_rerank_blenderbot_{now}_{info_to_save}.csv",
            "w+",
        ) as f1,
        open(
            f"{PREDS_OUTPUT_DIR}/rights/rerank_blenderbot/{DATA_TYPE_COMBINATION['test']}_rights_rerank_blenderbot_{now}_{info_to_save}.csv",
            "w+",
        ) as f2,
    ):
        for idx, row in test_df.iterrows():
            history = row["history"]
            text = row["text"]
            label = row["label"]
            # The context of the sentence.
            his = tokenizer.encode_plus(
                history,
                None,
                add_special_tokens=True,
                max_length=330,
                pad_to_max_length=True,
                return_token_type_ids=True,
                truncation=True,
            )
            outputs = generator_model.generate(
                input_ids=torch.tensor([his["input_ids"]], dtype=torch.long),
                attention_mask=torch.tensor([his["attention_mask"]], dtype=torch.long),
                early_stopping=True,
                num_beams=50,
                # do_sample=True,
                # top_p=0.95,
                # typical_p=0.95,
                bad_words_ids=bad_words_ids,
                no_repeat_ngram_size=2,
                repetition_penalty=1.5,
                num_return_sequences=50,
                max_length=10,
                output_scores=True,
                min_length=2,
            )
            pred_text = ""
            zipped = zip(outputs.sequences_scores, outputs.sequences)
            sorted_res = sorted(zipped, key=lambda x: x[0], reverse=True)
            # rprint(sorted_res)
            for rank, (sentence_score, sequence) in enumerate(sorted_res):
                #! replace <tutor>
                candidate = (
                    tokenizer.decode(sequence, clean_up_tokenization_spaces=True)
                    .replace("__unk__", " ")
                    .replace("__end__", "")
                    .replace("__start__", "")
                    .replace("__null__ ", "")
                    .replace("<tutor> ", "")
                    .strip()
                )
                clf_inputs = bert_tokenizer(candidate, return_tensors="pt")
                clf_outputs = clf_model(**clf_inputs)
                pred = clf_outputs.logits.argmax(dim=1, keepdim=True).item()
                # convert the 4 classes predictions to 2 classes predictions
                if (pred_mode == 2) and (int(pred) != 0):
                    pred = 1

                if int(pred) == int(label):
                    pred_text = candidate
                    f2.write(str(idx + 2))  # line number starts from 2
                    f2.write(",")
                    f2.write(history)
                    f2.write(",")
                    f2.write(text)
                    f2.write(",")
                    f2.write(pred_text)
                    rprint(f"[red]{pred_text}")
                    f2.write(",")
                    f2.write(str(label))
                    f2.write(",")
                    f2.write(str(rank))
                    rprint(rank)
                    f2.write("\n")
                    break
            if pred_text == "":
                count_wrong += 1
                # if no corresponding prediction is found, pick the a first best one
                pred_text = tokenizer.decode(sorted_res[0][1], skip_special_tokens=True)
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
                f1.write(",")
                f1.write(str(rank))
                f1.write("\n")
                rprint(
                    f"[red]{round(count_wrong/len_, ndigits=3)*100}% lines with wrong labels"
                )
                rprint(f"text with wrong label: [bright_blue]{pred_text}")
                rprint(f"right label: [blue]{label}")
                rprint(f"target text: [cyan]{text}")
            predictions.append(pred_text)
            rprint(f"[green]{idx+1}/{len_} is finished")

now = datetime.now()
now = now.strftime("%m_%d_%Y_%H-%M-%S")
preds_file_path = f"{PREDS_OUTPUT_DIR}/rerank_blenderbot/{DATA_TYPE_COMBINATION['test']}_predictions_{now}_{info_to_save}.txt"
with open(
    preds_file_path,
    "w+",
) as f:
    for pred in predictions:
        f.write(pred + "\n")

combine_result_df = combine_preds_targets(
    test_data_path=test_data_path, preds_file=preds_file_path
)

combine_result_df.to_csv(
    f"{COMBINED_PREDS_OUTPUT_DIR}/rerank_bart/{DATA_TYPE_COMBINATION['test']}_predictions_{now}_{info_to_save}.csv",
    index=False,
)
