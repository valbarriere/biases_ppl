import os
import time
import argparse
import pickle as pkl
from transformers import AutoTokenizer
from dotenv import load_dotenv
from vllm import LLM, SamplingParams
from utils import read_csv_val
load_dotenv()


CONTENT_SYSTEM = "You are a helpful and honest assistant. Please, respond concisely and truthfully."
CACHE_DIR = os.getenv("CACHE_DIR", None)
BOOL_TEST=True



def get_task_and_tweet_template_from_task(task_to_label):
    if task_to_label == 'sentiment':
        task = "sentiment"
        label_str = "’positive’, ’neutral’ or ’negative’"
        task_str = "Sentiment"
    else:
        task = "degree of toxicity"
        label_str = "’not toxic’, ’slightly toxic’, ’moderately toxic’, ’very toxic’ or ’extremely toxic’"
        task_str = "Toxicity"
    template_dict =  {
        "task":task,
        "label_str": label_str,
        "task_str": task_str
    }
    return template_dict


def create_input_ids(list_tweets, tokenizer, template_dict, return_text_prompts=False):
    list_content_user = []
    for tweet in list_tweets:
        content_user = f"""
        Rate the {template_dict["task"]} in the text. 
        Possible values are {template_dict["label_str"]}. Write only one value, without more explanation.
        Text: ’{tweet}’ 
        {template_dict["task_str"]}:
        """
        list_content_user.append(content_user)

    messages = [(
        {"role": "system", "content": CONTENT_SYSTEM},
        {"role": "user", "content": content_user},) for content_user in list_content_user
    ]
    
    # return the prompts and not the dictionnary
    if return_text_prompts:
        inputs_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            padding=True, 
            tokenize=False,
        )
    else:
        inputs_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True, 
        )

    return inputs_ids

def create_prompts(list_tweets, tokenizer, template_dict):
    list_content_user = []
    for tweet in list_tweets:
        content_user = f"""
        Rate the {template_dict["task"]} in the text. 
        Possible values are {template_dict["label_str"]}. Write only one value, without more explanation.
        Text: ’{tweet}’ 
        {template_dict["task_str"]}:
        """
        list_content_user.append(content_user)

    messages = [(
        {"role": "system", "content": CONTENT_SYSTEM},
        {"role": "user", "content": content_user},) for content_user in list_content_user
    ]

    prompts = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        padding=True, 
        tokenize=False,
    )
    
    return prompts

def annotate_file(model_id, task, file_path):
    model_name = model_id.split("/")[1]

    vllm_sampling_parameters = {
        "temperature": 0.0,
        "top_p": 0.95,
    }
    sampling_params = SamplingParams(**vllm_sampling_parameters)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=CACHE_DIR,
        padding_side='left',
    )
    
    tokenizer.pad_token = tokenizer.eos_token

    template_dict = get_task_and_tweet_template_from_task(task)
    tweets_df = read_csv_val(file_path)

    llm = LLM(model=model_id,
            dtype="float16",
            download_dir=CACHE_DIR,
            )
    
    list_tweets = tweets_df["tweet"].fillna("CVxTz").values
    t = time.time()
    responses = []
    
    prompts = create_input_ids(list_tweets, tokenizer, template_dict, return_text_prompts=True)


    outputs = llm.generate(
        prompts = prompts,
        sampling_params=sampling_params,
        use_tqdm=BOOL_TEST,
    )
    
    responses = [opt.outputs[0].text for opt in outputs]
    
    print("Labelling done: %d examples in %.2f minutes"%(len(list_tweets), (time.time()-t)/60.))

    tweets_df["label"] = responses
    
    file_name = f"{file_path}-{model_name}_labeled"

    tweets_df.to_csv(file_name, sep="\t") 
    print(f"Data stored in {file_name}")

def main():
    parser = argparse.ArgumentParser(description="Annotator for sentiment or toxicity tasks")

    parser.add_argument('-m', '--model_id', type=str, required=True, help="Name of the HF model to use as annotator.")
    parser.add_argument('-t', '--task', type=str, required=True, choices=["sentiment", "toxic"], help="Annotation task to be done (sentiment or toxic)")
    parser.add_argument('-f', '--file_path', type=str, required=True, help="Path to the input file to be annotated. A tsv file is expected with a tweet column")

    args = parser.parse_args()

    print(f"Model Name: {args.model_id}")
    print(f"Task: {args.task}")
    print(f"Input File: {args.file_path}")

    annotate_file(args.model_id, args.task, args.file_path)

if __name__ == "__main__":
    main()
