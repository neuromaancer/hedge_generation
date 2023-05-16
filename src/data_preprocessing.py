from venv import create
import pandas as pd
from rich import print as rprint
from ast import literal_eval
from pathlib import Path
import random


def count_turn(l):
    turn_num = 0
    speaker = l[0][0]
    rprint(l[1][0])
    rprint(f"Speaker: {speaker}")
    for idx in range(1, len(l)):
        if l[idx][0] != speaker:
            turn_num += 1
            speaker = l[idx][0]
    return turn_num


def label_str_to_int(str_):
    if str_ == "Nothing":
        return 0
    elif str_ in ["IDQ", "IDS", "IDA"]:
        return 1


def create_turns(
    df,
    turn_column="turn",
    utterance_column="Text",
    group_by=["Dyad", "Session"],
):
    df[turn_column] = ""
    df["turn_label"] = 0
    groups = df.groupby(by=group_by)
    for idx, group in groups:
        turn = []
        labels = []
        first = group.head(1)
        speaker = literal_eval(first[utterance_column].values[0])[0]
        for i, row in group.iterrows():
            t = literal_eval(row[utterance_column])
            label = label_str_to_int(row["Label"])
            if t[0] == speaker:
                turn.append(t)
                labels.append(label)
            else:
                df.at[i - 1, turn_column] = str(turn)
                df.at[i - 1, "turn_label"] = max(labels)
                turn.clear()
                labels.clear()
                turn.append(t)
                labels.append(label)
                speaker = t[0]
            if i == group.tail(1).index[0]:
                df.at[i, turn_column] = turn
                df.at[i, "turn_label"] = max(labels)
    return df


def find_history_upper_bound(
    series, df, idx, null_block_type="", history_size=4, turn_column="turn"
):
    first_index = series.head(1).index[0]
    size = 0
    if idx - history_size <= first_index:
        return first_index
    expand_window = 0
    while size < history_size:
        if idx - history_size - expand_window == first_index:
            return first_index
        l = df[idx - history_size - expand_window : idx][turn_column].values.tolist()
        size = sum(item != null_block_type for item in l)
        expand_window += 1

    return idx - history_size - expand_window + 1


def create_history_by_turn(
    df: pd.DataFrame,
    rolling=4,
    group_by=["Dyad", "Session"],
    turn_column="turn",
    history_column="history",
) -> pd.DataFrame:
    df[history_column] = ""
    groups = df.groupby(by=group_by)
    for idx, group in groups:
        for i, row in group.iterrows():
            upper_bound = find_history_upper_bound(
                series=group[turn_column],
                df=df,
                idx=i,
                null_block_type="",
                history_size=rolling,
            )
            history = df[turn_column][upper_bound:i].values
            df.at[i, history_column] = history
    return df


def decide_role(people, period):
    if period == "T1":
        if people == "P1":
            return "tutor"
        if people == "P2":
            return "tutee"
    elif period == "T2":
        if people == "P1":
            return "tutee"
        if people == "P2":
            return "tutor"


if __name__ == "__main__":
    raw_data_path = Path("../data/raw/indirectness_dataset.csv")
    #! important parameter
    history_size = 4
    raw_df = pd.read_csv(raw_data_path)
    df_with_turn = create_turns(raw_df)
    df_with_history = create_history_by_turn(df_with_turn, rolling=history_size)
    rprint(df_with_history.head())
    input("Press Enter to continue...")
    drop_lines = []
    for i, row in df_with_history.iterrows():
        people = literal_eval(row["Text"])[0]
        role = decide_role(people, row["Period"])
        if role == "tutee":
            drop_lines.append(i)

    df_wo_tutee = df_with_history.drop(drop_lines)
    df_wo_tutee["history_sentence"] = ""
    df_wo_tutee["turn_sentence"]    = ""
    df_wo_tutee["numerical_label"] = 0
    df_wo_tutee["text"] = ""
    for idx, row in df_wo_tutee.iterrows():
        string_history = str(row["history"])
        turn = str(row["turn"])
        p1_role = decide_role("P1", row["Period"])
        p2_role = decide_role("P2", row["Period"])
        df_wo_tutee.at[idx, "history_sentence"] = (
            string_history.replace("[", "")
            .replace("]", "")
            .replace(")", "")
            .replace("(", "")
            .replace("'", "")
            .replace(",", "")
            .replace("\n", "")
            .replace('"', "")
            .replace("\\", "")
            .replace("P1", f"<{p1_role}>")
            .replace("P2", f"<{p2_role}>")
            .replace("{laughter}", "<laughter>")
            .replace("laughter", "<laughter>")
            .replace("<<laughter>>", "<laughter>")
            .replace("cos", "because")
            .replace("\{sfx\}", "")
            .replace("sfx", "")
            .replace("inaudible", "")
            .replace("inhale", "")
            .replace("exhale", "")
            .replace("  ", " ")
            .strip()
        )
        df_wo_tutee.at[idx, "turn_sentence"] = (
            turn.replace("[", "")
            .replace("]", "")
            .replace(")", "")
            .replace("(", "")
            .replace("'", "")
            .replace(",", "")
            .replace("\n", "")
            .replace('"', "")
            .replace("\\", "")
            .replace("P1", "")
            .replace("P2", "")
            .replace("{laughter}", "<laughter>")
            .replace("laughter", "<laughter>")
            .replace("<<laughter>>", "<laughter>")
            .replace("cos", "because")
            .replace("\{sfx\}", "")
            .replace("sfx", "")
            .replace("inaudible", "")
            .replace("inhale", "")
            .replace("exhale", "")
            .replace("  ", " ")
            .strip()
        )
        if row["Label"] == "Nothing":
            df_wo_tutee.at[idx, "numerical_label"] = 0
        elif row["Label"] == "IDQ":
            df_wo_tutee.at[idx, "numerical_label"] = 1
        elif row["Label"] == "IDS":
            df_wo_tutee.at[idx, "numerical_label"] = 2
        elif row["Label"] == "IDA":
            df_wo_tutee.at[idx, "numerical_label"] = 3

        df_wo_tutee.at[idx, "text"] = (
            str(row["Text"])
            .replace(")", "")
            .replace("(", "")
            .replace("P1", "")
            .replace("P2", "")
            .replace("'", "")
            .replace(",", "")
            .replace("\n", "")
            .replace('"', "")
            .replace("\\", "")
            .replace("{laughter}", "<laughter>")
            .replace("laughter", "<laughter>")
            .replace("<<laughter>>", "<laughter>")
            .replace("cos", "because")
            .replace("\{sfx\}", "")
            .replace("sfx", "")
            .replace("inaudible", "")
            .replace("inhale", "")
            .replace("exhale", "")
            .replace("  ", " ")
            .strip()
        )

    hedging_df = df_wo_tutee[
        [
            "history_sentence",
            "text",
            "numerical_label",
            "turn_sentence",
            "turn_label",
            "Session",
            "Dyad",
            "Period",
        ]
    ]
    hedging_df.rename(
        columns={
            "history_sentence": "history",
            "turn_sentence": "turn",
            "numerical_label": "label",
            "Session": "session",
            "Dyad": "dyad",
            "Period": "period",
        },
        inplace=True,
    )
    hedging_df.head(10)
    #! important number: pick one session as the test set
    test_session = 3
    train_hedging_df = hedging_df[hedging_df["session"] != test_session]
    test_hedging_df = hedging_df[hedging_df["session"] == test_session]
    rprint(len(train_hedging_df))
    rprint(len(test_hedging_df))
    rprint(len(train_hedging_df) / len(hedging_df))
    rprint(train_hedging_df["label"].value_counts())
    rprint(test_hedging_df["label"].value_counts())
    hd_df_0 = train_hedging_df[train_hedging_df["label"] == 0].sample(504)
    hd_df_1 = train_hedging_df[train_hedging_df["label"] == 1]
    balanced_train_hedging_df = pd.concat([hd_df_0, hd_df_1])
    hd_df_0 = test_hedging_df[test_hedging_df["label"] == 0].sample(185)
    hd_df_1 = test_hedging_df[test_hedging_df["label"] == 1]
    balanced_test_hedging_df = pd.concat([hd_df_0, hd_df_1])

    train_hedging_df = train_hedging_df.dropna()
    test_hedging_df = test_hedging_df.dropna()
    balanced_test_hedging_df = balanced_test_hedging_df.dropna()
    balanced_train_hedging_df = balanced_train_hedging_df.dropna()

    rprint(len(train_hedging_df))
    rprint(len(test_hedging_df))
    rprint(len(train_hedging_df) / len(hedging_df))
    rprint(train_hedging_df["label"].value_counts())
    rprint(test_hedging_df["label"].value_counts())

    train_hedging_df.to_csv(
        f"../data/trainset/full_train_hedging_session_wise_history_size_{history_size}_turn_level_label.csv"
    )
    test_hedging_df.to_csv(
        f"../data/testset/full_test_hedging_session_wise_history_size_{history_size}_turn_level_label.csv"
    )
    balanced_train_hedging_df.to_csv(
        f"../data/trainset/balanced_train_hedging_session_wise_history_size_{history_size}_turn_level_label.csv"
    )
    balanced_test_hedging_df.to_csv(
        f"../data/testset/balanced_test_hedging_session_wise_history_size_{history_size}_turn_level_label.csv"
    )

    train_hedging_df = pd.read_csv(
        f"../data/trainset/full_train_hedging_session_wise_history_size_{history_size}_turn_level_label.csv"
    )
    test_hedging_df = pd.read_csv(
        f"../data/testset/full_test_hedging_session_wise_history_size_{history_size}_turn_level_label.csv"
    )
    balanced_train_hedging_df = pd.read_csv(
        f"../data/trainset/balanced_train_hedging_session_wise_history_size_{history_size}_turn_level_label.csv"
    )
    balanced_test_hedging_df = pd.read_csv(
        f"../data/testset/balanced_test_hedging_session_wise_history_size_{history_size}_turn_level_label.csv"
    )

    train_hedging_df = train_hedging_df.dropna()
    test_hedging_df = test_hedging_df.dropna()
    balanced_test_hedging_df = balanced_test_hedging_df.dropna()
    balanced_train_hedging_df = balanced_train_hedging_df.dropna()

    rprint(len(train_hedging_df))
    rprint(len(test_hedging_df))
    rprint(len(train_hedging_df) / len(hedging_df))
    rprint(train_hedging_df["label"].value_counts())
    rprint(test_hedging_df["label"].value_counts())
    train_hedging_df.to_csv(
        f"../data/trainset/full_train_hedging_session_wise_history_size_{history_size}_turn_level_label.csv"
    )
    test_hedging_df.to_csv(
        f"../data/testset/full_test_hedging_session_wise_history_size_{history_size}_turn_level_label.csv"
    )
    balanced_train_hedging_df.to_csv(
        f"../data/trainset/balanced_train_hedging_session_wise_history_size_{history_size}_turn_level_label.csv"
    )
    balanced_test_hedging_df.to_csv(
        f"../data/testset/balanced_test_hedging_session_wise_history_size_{history_size}_turn_level_label.csv"
    )
