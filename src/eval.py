import warnings
from pathlib import Path
from rich import print as rprint
from ast import literal_eval
import numpy as np
import pandas as pd
import pretty_errors
import torch
from dotenv import dotenv_values
from nlgeval import NLGEval, compute_metrics
from sklearn.metrics import accuracy_score, classification_report
from summ_eval.bert_score_metric import BertScoreMetric
from summ_eval.chrfpp_metric import ChrfppMetric
from transformers import BertForSequenceClassification, BertTokenizer

from BARTScore.bart_score import BARTScorer
from datetime import datetime
from utils import define_data_path

pretty_errors.configure(
    separator_character="*",
    filename_display=pretty_errors.FILENAME_EXTENDED,
    line_number_first=True,
    display_link=True,
    lines_before=5,
    lines_after=2,
    line_color=f"{pretty_errors.RED}> {pretty_errors.default_config.line_color}",
    code_color=f"  {pretty_errors.default_config.line_color}",
    truncate_code=True,
    display_locals=True,
)


warnings.filterwarnings("ignore")
torch.cuda.empty_cache()

config = dotenv_values("../.env")
EVAL_MODEL = config["EVAL_MODEL"]
REFS_FILE = Path(config["REFS_FILE"])
PREDS_FILE = Path(config["PREDS_FILE"])
DATA_TYPE_COMBINATION = literal_eval(config["DATA_TYPE_COMBINATION"])
EVAL_LOG_FILE = Path(config["EVAL_LOG_FILE"])
COMMENTS_LOG_FILE = config["COMMENTS_LOG_FILE"]
COMBINED_PREDS_OUTPUT_DIR = config["COMBINED_PREDS_OUTPUT_DIR"]


train_data_path, test_data_path = define_data_path(DATA_TYPE_COMBINATION)

now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

eval_df = pd.read_csv(EVAL_LOG_FILE)
test_df = pd.read_csv(test_data_path)

eval_dict = {
    "time": now,
    "model": EVAL_MODEL,
    "trainset": train_data_path.name,
    "testset": test_data_path.name,
    "comments": COMMENTS_LOG_FILE,
}


with open(PREDS_FILE, "r") as f:
    preds = f.readlines()
    # test_df["preds"] = preds
    preds = list(map(lambda s: s.strip(), preds))
    test_df["preds"] = preds
    rprint("[red]length of preds: ", len(preds))
with open(
    REFS_FILE,
    "r",
) as f:
    refs = f.readlines()
    refs = list(map(lambda s: s.strip(), refs))
    rprint("[red]length of refs: ", len(refs))


bart_scorer = BARTScorer(device="cuda:0", checkpoint="facebook/bart-large-cnn")
bart_scorer.load(path="../models/bartscore.pth")
bart_score = np.mean(bart_scorer.score(preds, refs, batch_size=4))
eval_dict["bart_score"] = round(bart_score, ndigits=3)

rprint(f"BART Score: {bart_score}")

bert = BertScoreMetric()
charfp = ChrfppMetric()

bert_dict = bert.evaluate_batch(preds, refs)
rprint(f"BERT: {bert_dict}")
eval_dict["bert_score"] = round(bert_dict["bert_score_f1"], ndigits=3) * 100

charfp_dict = charfp.evaluate_batch(preds, refs)
rprint(f"CHARFP: {charfp_dict}")
eval_dict["charfp"] = round(charfp_dict["chrf"], ndigits=3)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("../models/bert_clf")

preds = []
model.eval()
with torch.no_grad():
    with open(PREDS_FILE, "r") as f:
        for line in f:
            line = line.strip()
            inputs = tokenizer(line, return_tensors="pt")
            outputs = model(**inputs)
            pred = outputs.logits.argmax(dim=1, keepdim=True)
            preds.append(pred.item())

predictions = []
for pred in preds:
    if pred != 0:
        predictions.append(1)
    else:
        predictions.append(0)

test_df["preds_labels"] = predictions
refs = test_df["label"].tolist()
rprint("len refs", len(refs))
rprint("len preds", len(predictions))

acc = accuracy_score(refs, predictions)
rprint("accuracy: ", acc)
eval_dict["acc"] = round(acc, ndigits=3)

rprint(set(predictions))

report = classification_report(y_true=refs, y_pred=predictions, output_dict=True)
rprint(report)
eval_dict["weighted_f1"] = round(report["weighted avg"]["f1-score"], ndigits=3)
eval_dict["marco_f1"] = round(report["macro avg"]["f1-score"], ndigits=3)

rprint(len(predictions))
rprint(len(refs))


n = NLGEval()
metrics_dict = compute_metrics(
    PREDS_FILE, [REFS_FILE], no_skipthoughts=True, no_glove=True
)

rprint(f"{metrics_dict}")
eval_dict["bleu_1"] = round(metrics_dict["Bleu_1"] * 100, ndigits=3)
eval_dict["bleu_2"] = round(metrics_dict["Bleu_2"] * 100, ndigits=3)
eval_dict["bleu_3"] = round(metrics_dict["Bleu_3"] * 100, ndigits=3)
eval_dict["rouge-L"] = round(metrics_dict["ROUGE_L"] * 100, ndigits=3)

eval_df = eval_df.append(eval_dict, ignore_index=True)
rprint(eval_df)
eval_df.to_csv(EVAL_LOG_FILE, index=False)
path_to_save = Path(f"{COMBINED_PREDS_OUTPUT_DIR}/{EVAL_MODEL}")
path_to_save.mkdir(parents=True, exist_ok=True)
test_df = test_df[["history", "text", "label", "preds_labels", "preds"]]
test_df.to_csv(
    path_to_save / f"{EVAL_MODEL}_{now}_{COMMENTS_LOG_FILE}.csv", index=False
)
