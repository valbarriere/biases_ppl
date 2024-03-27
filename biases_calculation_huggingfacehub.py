"""
Soit je compare les changements par rapport aux predictions d'avant, en calculant:
- la diff moyenne de proba pos
- la diff moyenne de proba neg
- diff entre Ppos-Pneg en moyenne
- la difference des F1-pro, F1-neg ?
- le pourcentage de prediction differentes des predictions normales sur les positif/negatif (i.e. #FP et FN sur pos/neg quand on compare par rapport aux pred originales)

Valentin Barriere, 02/22
"""
import os
import argparse
import pickle as pkl
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.special import softmax
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, AutoConfig

from CountryGenderNamePerturbation import PerturbedExamples

CACHE_DIR = os.getenv("CACHE_DIR", None)
DATA_PATH = os.getenv("DATA_PATH", "xlm-t/data/sentiment/english")
proxies = None


MAX_SEQ_LEN_INI = 256


def read_text_data_to_df(text_file_path, labels_file_path, encoding="utf-8"):
    with open(text_file_path, "r", encoding=encoding) as file:
        text_data = file.readlines()

    with open(labels_file_path, "r", encoding=encoding) as file:
        labels_data = file.readlines()

    text_data = [line.strip() for line in text_data]
    labels_data = [line.strip() for line in labels_data]

    df = pd.DataFrame({"tweet": text_data, "label": labels_data})
    return df

def read_tsv_data_to_df(tsv_file_path, text_col, label_col, encoding="utf-8"):
    tsv_df = pd.read_csv(tsv_file_path, sep="\t", quotechar='"', encoding=encoding, header=0)

    text_data = tsv_df[text_col].values
    labels_data = tsv_df[label_col].values

    df = pd.DataFrame({"tweet": text_data, "label": labels_data})
    return df


def load_df_data(
    tweet_label_df,
    label_type,
    dict_lab={"positive": 0, "negative": 1, "else": 2},
    SPARSE_CATEGORICAL=True,
    multi_labels=False,
    cumsum_label_vectors=False,
):
    """
    Load train data. You need the names of the columns on the first line of the CSV...
    """
    df = tweet_label_df

    # normally never happening
    train_tweets = df["tweet"].fillna("CVxTz").values  # why 'CVxTz'?

    list_unique = df["label"].unique()
    if label_type == "int":
        try:
            df["label"] = df["label"].astype(np.int32)
        except ValueError as e:
            print(list_unique)
            raise e("Some labels couldn't be transformed to int!")
        for label in df["label"].unique():
            if label not in dict_lab.values():
                raise ValueError(f"Label {label} not found in label2id dict!")
        # if unlabeled data only
        if (len(list_unique) == 1) and (list_unique[0] == -1):
            train_y = df["label"].astype(np.int32).values
        else:
            train_y = df["label"].values
    else:
        # TODO: no return inside a if... and no function inside a function
        if multi_labels:
            def func_mlabels(mlabels):
                y = np.zeros((len(dict_lab))).astype(np.int32)
                for lab in mlabels.split(","):
                    y[dict_lab[lab]] = 1
                # return ','.join([str(dict_lab[lab]) for lab in mlabels.split(',')])
                return y

            train_y = np.concatenate(
                [
                    k[np.newaxis, :]
                    for k in df["label"].astype(str).str.lower().map(func_mlabels).values
                ]
            )
        else:
            if dict_lab == {"regression": 0}:
                train_y = df["label"].values
            else:
                # Remark: we can keep else with another values
                if "else" in dict_lab.keys():
                    train_y = (
                        df["label"]
                        .str.lower()
                        .map(dict_lab)
                        .fillna(dict_lab["else"])
                        .astype(np.int32)
                        .values
                    )
                else:
                    # to be sure that there is no other label than the ones in the dict_label
                    # print(dict_lab, df['label'].unique(), df['label'].isnull().sum())
                    # print(df[df['label'].map(dict_lab).isnull()]['label'])
                    if df["label"].str.lower().map(dict_lab).isnull().sum() != 0:
                        print(dict_lab)
                        print(df["label"].str.lower().map(dict_lab).isnull().sum())
                        list_unique = df["label"].unique()
                        print(list_unique)
                        if (len(list_unique) != 1) or (list_unique[0] != "-1"):
                            raise Exception(
                                "Error: dict_lab not working well with the dataset"
                            )
                        else:
                            print("Dataset of unlabeled data...ok!")

                    # impossible if multi-labels
                    train_y = df["label"].str.lower().map(dict_lab).astype(np.int32).values

                    if len(dict_lab) != len(df["label"].unique()):
                        print("There might be a problem with the labels...")
                        print(
                            f'dict_lab : {dict_lab},\ndf["label"].unique() : {df["label"].unique()}\n\n'
                        )

                # create the one-hot vectors
                if not SPARSE_CATEGORICAL:
                    b = np.zeros((train_y.size, train_y.max() + 1), dtype=np.int32)
                    b[np.arange(train_y.size), train_y] = 1
                    train_y = b
                    if cumsum_label_vectors:
                        train_y = 1 - np.cumsum(train_y, axis=1)[:, :-1]

    return (train_tweets, train_y)


def create_input_array(sentences, tokenizer, MAX_SEQ_LEN=MAX_SEQ_LEN_INI):
    """
    This one for the new transformers version
    """
    # only for roberta and xlm-roberta
    if "roberta" in tokenizer.name_or_path:
        # if True:
        print("Using transfo3 tokenizer, for RoBERTa models")
        return create_input_array_transfo3(sentences, tokenizer, MAX_SEQ_LEN)
    else:
        print("Using transfo4 tokenizer, for non-RoBERTa models")
        sentences = [k.split(" </s> ") for k in sentences]
        # if there is no ' </s> ', need to input as a list
        if sum([len(s) != 1 for s in sentences]) == 0:
            sentences = [s[0] for s in sentences]
        encoded_inputs = tokenizer(
            list(sentences),
            padding="max_length",
            truncation=True,
            return_tensors="tf",
            max_length=MAX_SEQ_LEN,
        )
        return [
            encoded_inputs["input_ids"],
            encoded_inputs["attention_mask"],
            encoded_inputs["token_type_ids"],
        ]


def create_input_array_transfo3(sentences, tokenizer, MAX_SEQ_LEN=MAX_SEQ_LEN_INI):
    """
    Allow to use more than 2 sentences in the same input, with different token_type_ids
    Need to recode properly this function. Like, no need to return_tensors="tf"

    !!!!! MAXIMUM NUMBER OF SENTENCES is 4 !!!!!
    """
    # Case 2 sentences in input (like for QA, or putting context)
    if ("</s>" not in sentences[0]) or (tokenizer.not_use_token_type_ids):
        # if '</s>' not in sentences[0]:
        encoded_inputs_1 = tokenizer(
            list(sentences),
            padding="max_length",
            truncation=True,
            return_tensors="tf",
            max_length=MAX_SEQ_LEN,
        )
        separate_sentences = False
        encoded_inputs_2 = None
    else:
        separate_sentences = True
        # maximum nb of separation tokens found in a sentence
        nb_sent = np.max([k.count("</s>") for k in sentences]) + 1
        more_than_two_sent = nb_sent > 2

        dict_list_sentences = {
            k: [] for k in range(4)
        }  # TODO: changer le 4 code en dur
        # dict_list_sentences = {k: [] for k in range(nb_sent)}

        for sent in sentences:
            if not sent:
                print("***" * 100 + "ONE TEXT ELEMENT IS NULL !!!" + "***" * 100)
                for k in range(nb_sent):
                    dict_list_sentences[k].append("CVxTz")
            else:
                sent_split = sent.split("</s>")
                # in order to have always nb_sep + 1
                # si impair
                if (nb_sent - len(sent_split)) != 0:
                    # si impair
                    if len(sent_split) // 2:
                        sent_split += [None] + (nb_sent - len(sent_split) - 1) * [""]
                    else:
                        sent_split += (nb_sent - len(sent_split)) * [""]
                # sent_split += (nb_sent - len(sent_split))*['']
                for k in range(nb_sent):
                    dict_list_sentences[k].append(sent_split[k])

        try:
            encoded_inputs_1 = tokenizer(
                dict_list_sentences[0],
                dict_list_sentences[1],
                padding="max_length",
                truncation=True,
                return_tensors="tf",
                max_length=MAX_SEQ_LEN,
            )
        except:
            assert True
            print(
                "WARNING: Adding a special pad_token --> <pad>: tokenizer.add_special_tokens({'pad_token': '<pad>'})"
            )
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            encoded_inputs_1 = tokenizer(
                dict_list_sentences[0],
                dict_list_sentences[1],
                padding="max_length",
                truncation=True,
                return_tensors="tf",
                max_length=MAX_SEQ_LEN,
            )

        # If there are 3 sentences at least one time, else encoded_inputs_2 in useless everytime
        if more_than_two_sent:
            if dict_list_sentences[3]:
                encoded_inputs_2 = tokenizer(
                    dict_list_sentences[2],
                    dict_list_sentences[3],
                    padding="max_length",
                    truncation=True,
                    return_tensors="tf",
                    max_length=MAX_SEQ_LEN,
                )
            else:
                encoded_inputs_2 = tokenizer(
                    dict_list_sentences[2],
                    padding="max_length",
                    truncation=True,
                    return_tensors="tf",
                    max_length=MAX_SEQ_LEN,
                )
        else:
            encoded_inputs_2 = None

    # Adding token_type_ids
    list_encoded_inputs = []
    for encoded_inputs in [encoded_inputs_1, encoded_inputs_2]:
        if encoded_inputs:
            # OLD stuff, I coded the good token_type_ids for RoBERTa (function tokenization_roberta.py) so return_token_type_ids=True when calling tokenizer
            # You don't have token_type_ids with roberta..
            if "token_type_ids" not in encoded_inputs.keys():
                # encoded_inputs['token_type_ids'] = None
                encoded_inputs["token_type_ids"] = np.zeros(
                    encoded_inputs["input_ids"].shape, dtype=np.int32
                )
                if separate_sentences:
                    for idx_sent, ts_sent in tqdm(
                        enumerate(encoded_inputs["input_ids"])
                    ):
                        bool_sent_2 = 0
                        for idx_tok, tok in enumerate(ts_sent):
                            if tok == tokenizer.sep_token_id:
                                bool_sent_2 += 1
                            if (
                                bool_sent_2 == 2
                            ):  # there are 2 consecutive <s> between 2 sentences
                                encoded_inputs["token_type_ids"][idx_sent, idx_tok] = 1

            encoded_inputs["token_type_ids"] = tf.convert_to_tensor(
                encoded_inputs["token_type_ids"]
            )

        list_encoded_inputs.append(encoded_inputs)

    encoded_inputs_1, encoded_inputs_2 = list_encoded_inputs

    # Merging the encoding inputs together
    if encoded_inputs_2:
        input_ids = encoded_inputs_1["input_ids"].numpy()
        attention_mask = encoded_inputs_1["attention_mask"].numpy()
        token_type_ids = encoded_inputs_1["token_type_ids"].numpy()

        input_ids_2 = encoded_inputs_2["input_ids"].numpy()
        attention_mask_2 = encoded_inputs_2["attention_mask"].numpy()
        token_type_ids_2 = encoded_inputs_2["token_type_ids"].numpy()

        a, b = np.where(input_ids_2 == tokenizer.sep_token_id)
        # a, b  = np.where(attention_mask_2)

        for idx_sent in range(input_ids_2.shape[0]):
            pos_max = b[np.max(np.where(a == idx_sent))] + 1
            # case where the are more than 2 sentences
            if pos_max > 3:
                len_sent_1 = np.max(np.where(attention_mask[idx_sent])) + 1

                # maximum size
                pos_max = min(pos_max, MAX_SEQ_LEN - len_sent_1)
                if pos_max:
                    # so the last one has a 0 for token_type_ids (idk why I made it like that)
                    token_type_ids_2[idx_sent, pos_max - 1] = -2
                    token_type_ids[
                        idx_sent, len_sent_1 - 1 : len_sent_1 - 1 + pos_max
                    ] = (token_type_ids_2[idx_sent, :pos_max] + 2)

                    attention_mask[idx_sent, len_sent_1 : len_sent_1 + pos_max] = 1

                    new_input = input_ids_2[idx_sent, :pos_max]
                    new_input[0] = 2
                    input_ids[idx_sent, len_sent_1 : len_sent_1 + pos_max] = new_input

        encoded_inputs = {
            "input_ids": tf.convert_to_tensor(input_ids),
            "attention_mask": tf.convert_to_tensor(attention_mask),
            "token_type_ids": tf.convert_to_tensor(token_type_ids),
        }

    else:
        encoded_inputs = encoded_inputs_1

    return [
        encoded_inputs["input_ids"],
        encoded_inputs["attention_mask"],
        encoded_inputs["token_type_ids"],
    ]

def _kl_div(P, Q, mean_of_divs=True):
    """
    2 options: mean of the KL on each example or mean of the probabilities and KL global
    """
    P = np.array(P)
    Q = np.array(Q)

    # mean of the divs
    if mean_of_divs:
        kl = np.mean(np.sum((P * np.log(P / Q)), axis=1))
    else:
        # div of the means
        P = np.mean(P, axis=1)
        Q = np.mean(Q, axis=1)
        kl = np.sum((P * np.log(P / Q)))

    return kl


def symetric_kl(P, Q, mean_of_divs=True):
    return (
        _kl_div(P, Q, mean_of_divs=mean_of_divs)
        + _kl_div(Q, P, mean_of_divs=mean_of_divs)
    ) / 2


def get_model_tokenizer_and_config(model_name_or_path,
                                   model_gen=TFAutoModelForSequenceClassification,
                                   tok_gen=AutoTokenizer):
    tokenizer = tok_gen.from_pretrained(
        model_name_or_path, proxies=proxies, cache_dir=CACHE_DIR
    )
    model = model_gen.from_pretrained(
        model_name_or_path, proxies=proxies, cache_dir=CACHE_DIR
    )

    config = AutoConfig.from_pretrained(model_name_or_path)
    return model, tokenizer, config


def _calculate_sentiment_bias(
    model,
    X_text,
    y,
    tokenizer,
    dict_lab,
    data_path,
    list_countries=[],
    n_duplicates=10,
    dict_pos_neg={"positive": "yes", "negative": "no"},
):
    """
    The function itself, taking model, X_text, y as inputs
    """

    str_pos = "positive"
    str_neg = "negative"
    # if not positive negative, then you need a mapping between the variables and pos/neg
    if not (("positive" in dict_lab.keys()) and ("negative" in dict_lab.keys())):
        # check that all the labels are in the dict_pos_neg so that can be a direct mapping
        for k in dict_pos_neg.values():
            assert k in list(
                dict_lab.keys()
            ), "dict_pos_neg does not contain the string names of positive/negative classes"

        print("new mapping:", dict_pos_neg)
        str_pos = dict_pos_neg[str_pos]
        str_neg = dict_pos_neg[str_neg]
    perturber = (
        PerturbedExamples(list_countries)
        if len(list_countries)
        else PerturbedExamples()
    )
    print("Perturbing examples...")
    perturbed_X_text = perturber.all_countries(X_text, y, n_duplicates)
    nb_sent_modified = len(perturbed_X_text["Original"][0])
    print("...Obtaining %d perturbated sentences" % nb_sent_modified)
    dict_results = {country: {} for country in perturbed_X_text.keys()}

    path_dump_perturbed = os.path.join(data_path, "perturbed_text_stances.pkl")
    with open(path_dump_perturbed, "wb") as fp:
        pkl.dump(perturbed_X_text, fp)

    if nb_sent_modified > 1:
        print("Calculating probas...")
        # print(perturbed_X_text['Original'][0])

        input_arrays = create_input_array(perturbed_X_text["Original"][0], tokenizer)
        try:  # for TFAutoModelForSequenceClassification
            proba_ini = softmax(model.predict(input_arrays), axis=1)
        except:
            proba_ini = softmax(model.predict(input_arrays)[0], axis=1)
        y_hat_ini = [
            kkk
            for kk in [n_duplicates * [k] for k in np.argmax(proba_ini, axis=1)]
            for kkk in kk
        ]

        # mean of the difference between pos and neg
        dict_results["Original"] = 100 * np.mean(
            proba_ini[:, dict_lab[str_pos]] - proba_ini[:, dict_lab[str_neg]]
        )

        for country, value_gender in tqdm(perturbed_X_text.items()):
            if country != "Original":
                for gender, (examples, labels) in value_gender.items():
                    input_arrays = create_input_array(examples, tokenizer)
                    # print(model.predict(input_arrays))
                    try:
                        proba = softmax(model.predict(input_arrays), axis=1)
                    except:  # for TFAutoModelForSequenceClassification
                        proba = softmax(model.predict(input_arrays)[0], axis=1)
                    dict_results[country][gender] = {
                        "proba": np.round(
                            100
                            * np.mean(
                                proba[:, dict_lab[str_pos]]
                                - proba[:, dict_lab[str_neg]]
                            )
                            - dict_results["Original"],
                            2,
                        )
                    }
                    y_hat_pred = np.argmax(proba, axis=1)
                    CM = confusion_matrix(y_hat_ini, y_hat_pred)

                    dump_CM_file = f"CM_stances_en_{country}.pkl"
                    path_dump_CM = os.path.join(data_path, dump_CM_file)
                    with open(path_dump_CM, "wb") as fp:
                        pkl.dump(CM, fp)
                    # gives the percentage of increasing/decreasing
                    # the number of new prediction - the number of past prediction / the number of past prediction
                    percentage_changing = (
                        np.sum(CM, axis=0) - np.sum(CM, axis=1)
                    ) / np.sum(CM, axis=1)

                    # case where there was only one class outputed by the system, i.e. if very few entities were detected
                    if len(CM) == 1:
                        print(percentage_changing)
                        percentage_changing = [1 for _ in dict_lab.values()]

                    # print(y_hat_ini, y_hat_pred, CM)
                    for lab, idx_lab in dict_lab.items():
                        dict_results[country][gender][lab] = 100 * np.round(
                            percentage_changing[idx_lab], 2
                        )

                    dict_results[country][gender]["KL_sym"] = symetric_kl(
                        np.repeat(proba_ini, n_duplicates, axis=0), proba
                    )
                    dict_results[country][gender]["KL_sym_mean"] = symetric_kl(
                        np.repeat(proba_ini, n_duplicates, axis=0),
                        proba,
                        mean_of_divs=False,
                    )
                    # dict_results[country][gender]['Sinkhorn'] = kl_div(proba_ini, proba)

        # creating the df
        del dict_results["Original"]
        midx = pd.MultiIndex.from_product(
            [
                list(dict_results.values())[0].keys(),
                list(list(dict_results.values())[0].values())[0].keys(),
            ]
        )
        df_bias = pd.DataFrame(
            [
                [key3 for key2 in key1.values() for key3 in key2.values()]
                for key1 in dict_results.values()
            ],
            index=dict_results.keys(),
            columns=midx,
        )

    else:
        print("Not even one sentence detected with an entity... Return empty dataframe")
        df_bias = pd.DataFrame([[]])

    df_bias["n_sent_modified"] = nb_sent_modified
    return df_bias


def check_file_exists_while_parsing(file_path):
    if not os.path.exists(file_path):
        raise argparse.ArgumentTypeError(f"{file_path} does not exist")
    return file_path


def parse_arguments():
    parser = argparse.ArgumentParser(description="Receives arguments to run biases calculation over defined datasets.")
    parser.add_argument("--model_name", type=str, help="Name of the model to run", default="cardiffnlp/twitter-xlm-roberta-base-sentiment")
    parser.add_argument("--data_path", type=str, help="Path where data is stored", default=DATA_PATH)
    parser.add_argument("--data_type", type=str, choices=["txt", "tsv"], help="Type of data (txt or tsv)", default="txt")
    parser.add_argument("--data_tsv", type=str, help="File containing the data (required if data_type is tsv)")
    parser.add_argument("--text_col", type=str, help="Column containing the text data in the tsv file (required if data_type is tsv)")
    parser.add_argument("--label_col", type=str, help="Column containing the label data in the tsv file (required if data_type is tsv)")
    parser.add_argument("--text_file", type=str, help="File containing the text data (required if data_type is txt)", default="test_text.txt")
    parser.add_argument("--label_file", type=str, help="File containing the label data (required if data_type is txt)", default="test_labels.txt")
    parser.add_argument("--label_type", type=str, choices=["str", "int"], help="Type of labels (str or int)", default="int")
    parser.add_argument(
        "--list_countries",
        help="countries to test",
        type=str,
        default=["United_Kingdom", "France", "Spain", "Germany"],
        nargs="+",
    )
    parser.add_argument(
        "--n_duplicates", help="how many n_duplicates", type=int, default=10
    )
    parser.add_argument("--test", help="test", default=False, action="store_true")
    args = parser.parse_args()
    if args.data_type == "tsv":
        if not (args.data_tsv and args.text_col and args.label_col):
            parser.error("When data_type is tsv, data_tsv, text_col, and label_col are required.")
        data_tsv_path = os.path.join(args.data_path, args.data_tsv)
        args.data_tsv = check_file_exists_while_parsing(data_tsv_path)
    else:
        if not (args.text_file and args.label_file):
            parser.error("When data_type is txt, text_file and label_file are required.")
        text_file_path = os.path.join(args.data_path, args.text_file)
        args.text_file = check_file_exists_while_parsing(text_file_path)
        label_file_path = os.path.join(args.data_path, args.label_file)
        args.label_file = check_file_exists_while_parsing(label_file_path)
    return args


def main(args):
    model_name = args.model_name
    
    model, tokenizer, config = get_model_tokenizer_and_config(model_name)

    data_path = args.data_path
    data_type = args.data_type
    if data_type == "txt":
        text_path = args.text_file
        label_path = args.label_file
        df = read_text_data_to_df(text_path, label_path)
    else:
        tsv_path = args.data_tsv
        text_col = args.text_col
        label_col = args.label_col
        df = read_tsv_data_to_df(tsv_path, text_col, label_col)


    label_type = args.label_type
    dict_lab = {k.lower():v for k,v in config.label2id.items()}
    X_text, y = load_df_data(df, label_type, dict_lab, )

    if args.test:
        X_text = X_text[:200]

    df_bias = _calculate_sentiment_bias(
        model,
        X_text,
        y,
        tokenizer,
        dict_lab,
        data_path,
        list_countries=args.list_countries,
        n_duplicates=args.n_duplicates,
    )
    if not args.test:
        df_bias.to_csv(os.path.join(data_path, "biases.tsv"), sep="\t")
    else:
        print(df_bias)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
