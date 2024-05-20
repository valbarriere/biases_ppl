
import os
import argparse
from dotenv import load_dotenv

from biases_calculator import BiasesCalculator 


load_dotenv()
PATH_DATA = os.getenv("PATH_DATA", None)


def check_file_exists_while_parsing(file_path):
    if not os.path.exists(file_path):
        raise argparse.ArgumentTypeError(f"{file_path} does not exist")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name",
                        help="The name of the model name", 
                        type=str, 
                        default="cardiffnlp/twitter-xlm-roberta-base-sentiment", 
                 )
    parser.add_argument("-c", "--name_corpora", 
                        help="The name of the folder containing the corpora", 
                        type=str, 
                        default="Biases")
    parser.add_argument("--path_corpora", 
                        help="The path of the folders containing all the corpora", 
                        type=str, 
                        default=PATH_DATA)
    parser.add_argument("--data_tsv", 
                        type=str, 
                        default="tweets_test_spanish_val.tsv")
    parser.add_argument("--text_col",
                        type=str,
                        help="Column containing the text data in the tsv file",
                        default="tweet")
    parser.add_argument("--label_col",
                        type=str,
                        help="Column containing the label data in the tsv file",
                        default="label")
    parser.add_argument("--label_type",
                        type=str,
                        choices=["str", "int"],
                        help="Type of labels (str or int)",
                        default="str",)
    parser.add_argument("--list_countries", 
                        help="countries to test", 
                        type=str, 
                        default=['United_Kingdom', "France", 'Spain', 'Germany'], 
                        nargs='+')
    parser.add_argument("--n_duplicates", 
                        help="how many n_duplicates", 
                        type=int, 
                        default=10)
    parser.add_argument("--test", 
                        help="test", 
                        default=False, 
                        action='store_true')
    args = parser.parse_args()
    input_data_file = args.data_tsv
    path_corpus = os.path.join(args.path_corpora, args.name_corpora)
    data_file_path = os.path.join(path_corpus, input_data_file)
    check_file_exists_while_parsing(data_file_path)
    return args

def main(args):
    path_corpus = os.path.join(args.path_corpora, args.name_corpora)
    biases_calculator = BiasesCalculator(args.model_name,
                                         path_corpus,
                                         args.data_tsv,
                                         args.text_col,
                                         args.label_col,
                                         args.label_type,
                                         args.list_countries,
                                         args.n_duplicates)
    biases_calculator.load_model_tokenizer_and_config()
    df = biases_calculator.read_tsv_data_to_df() # Reads the data and loads it as tweet, label
    X_text, y = biases_calculator.load_df_data(df)
    if args.test:
        X_text = X_text[:200]
    df_bias = biases_calculator._calculate_sentiment_bias(
        X_text,
        y
    )

    if args.test:
        print(df_bias)
    else:
        df_bias.to_csv(os.path.join(path_corpus, f"biases_{args.data_tsv}"), sep="\t")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
