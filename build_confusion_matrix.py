import os
import pandas as pd 
import seaborn as sn
import argparse
from matplotlib import pyplot as plt
from dotenv import load_dotenv

load_dotenv()
PATH_DATA = os.getenv("PATH_DATA", None)

lan2code = {
    'arabic': 'AR',
    'english' : 'EN',
    'french' : 'FR',
    'german' : 'DE',
    'hindi' : 'HI',
    'italian' : 'IT',
    'portuguese' : 'PT',
    'spanish' : 'ES',
} # Languages available in XLM-T

list_entities = ["Morocco",
                "United_Kingdom",
                "Germany",
                "France",
                "Spain",
                "Italy",
                "Portugal",
                "Hungary",
                "Poland",
                "Turkey"] # Languages used in biases calc

def standardize(dfnorm):
    return (2*dfnorm - dfnorm.min() - dfnorm.max())/(dfnorm.max() - dfnorm.min())

def normalize(dfnorm):
    return (dfnorm - dfnorm.mean())/(dfnorm.std())

def main():
    # Usage python build_confusion_matrix.py data "biases_tweets_test_{}.tsv"
    parser = argparse.ArgumentParser(description="Process some strings.")
    parser.add_argument('name_corpora', type=str, help='The name of the corpora')
    parser.add_argument('tsv_file_pattern', type=str, help='The file pattern of the biased tsv data files')
    args = parser.parse_args()
    data_path = os.path.join(PATH_DATA, args.name_corpora)
    lan2file = {lan2code[lan]: os.path.join(data_path, args.tsv_file_pattern.format(lan)) for lan in lan2code.keys()}
    dict_df = {lan : standardize(pd.read_csv(lan2file[lan], sep='\t', index_col=0, header=[0,1])['male']['proba'][list_entities]) 
               for lan in lan2file.keys()}
    dfmat = pd.DataFrame(dict_df).T
    dfmat.rename(columns={'United_Kingdom' : 'United Kingdom'}, inplace=True)
    sn.set(font_scale=1.4)
    plt.figure(figsize = (12,10))
    hm = sn.heatmap(dfmat, annot=True, cmap="Blues")
    plt.xticks(rotation=45)
    fig = hm.get_figure()
    fig.savefig("./Standardized_matrix_10lang.png", bbox_inches='tight', dpi=1280)

if __name__ == "__main__":
    main()