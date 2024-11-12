import os
import argparse
import json
from biases_calculation_huggingfacehub_PPL import main_probas
from PPL_Positivity import main_ppl
from generate_table3 import main_local_level

PATH_DATA = os.getenv("PATH_DATA", None)

def load_args_from_json(json_file):
    """Load arguments from a JSON file if specified."""
    with open(json_file, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to JSON config file with arguments", type=str)

    # Set default values and help messages in the parser
    parser.add_argument("--model_name", help="The name of the model", type=str, default="cardiffnlp/twitter-xlm-roberta-base-sentiment")
    parser.add_argument("--name_corpora", help="The name of the folder containing the corpora", type=str, default="Biases")
    parser.add_argument("--path_corpora", help="The path of the folders containing all the corpora", type=str, default=PATH_DATA)
    parser.add_argument("--data_tsv", help="The tsv file containing the data", type=str, default="tweets_test_spanish_val.tsv")
    parser.add_argument("--list_countries", help="Countries to test", type=str, nargs='+', 
                        default=['United_Kingdom', 'Ireland', 'United_States', 'Canada', 'Australia', 'New_Zealand', 
                                 'South_Africa', 'India', 'Germany', 'France', 'Spain', 'Italy', 'Portugal', 'Hungary', 
                                 'Poland', 'Turkey', 'Morocco'])
    parser.add_argument("--n_duplicates", help="How many n_duplicates", type=int, default=10)
    parser.add_argument("--proba_only", help="Calculate probability only", action='store_true', default=False)
    parser.add_argument("--male_only", help="Use male only", action='store_false', default=True)
    parser.add_argument("--perturb", help="Apply perturbation to data", action='store_false', default=True)
    parser.add_argument("--emotion_task", help="If the task is emotion classification", action='store_true', default=False)
    parser.add_argument("--ner_type", help="NER tool to use (spacy or hf)", type=str, choices=["spacy", "hf"], default="spacy")
    parser.add_argument("--ner_name", help="Name of the NER model to use", type=str, default="xx_ent_wiki_sm")
    parser.add_argument("--base_model_name", help="The name of the base model (Pretrained, not finetuned)", type=str, default="cardiffnlp/twitter-xlm-roberta-base")
    parser.add_argument("--list_gender", help="Gender to test", type=str, nargs='+', default=[])
    parser.add_argument("--verbose", help="Enable verbose output", action='store_true', default=False)
    parser.add_argument("--existing_dic", help="If already started experiments before", action=argparse.BooleanOptionalAction)
    
    # Parse command-line arguments
    args = parser.parse_args()

    # If JSON config is provided, load arguments from JSON and overwrite defaults
    if args.config:
        json_args = load_args_from_json(args.config)
        for key, value in json_args.items():
            # Only override `path_corpora` if it is not `None` in JSON
            if key == "path_corpora" and value is None:
                continue
            setattr(args, key, value)

    # Perturb and calculate probs
    main_probas(args)
    # Calculate perplexity
    main_ppl(args)
    # Calculate local correlation
    main_local_level(args)
