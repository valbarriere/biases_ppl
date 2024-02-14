import pickle as pkl
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import numpy as np
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to receive country index and CUDA index.")
    parser.add_argument("country_index", type=int, help="Index for country")
    parser.add_argument("cuda_index", type=int, help="Index for CUDA")
    args = parser.parse_args()
    country_index = args.country_index
    cuda_index = args.cuda_index
    countries = [
        "France", "United_Kingdom", "Ireland", "Spain", "Germany", "Italy", "Morocco", "India",
        "Canada", "Australia", "New_Zealand", "United_States", "South_Africa", "Portugal", "Hungary", "Poland",
        "Turkey", "Original"
    ]
    country = countries[country_index]
    gender = "male"
    name_file = 'Eurotweets_English_val_without_line_return.tsv_clean_test'
    path_data = './male_ppl/'
    path_dump_perturbed = path_data + 'Perturbed_' + name_file #+ '.pkl_ERROR'
    with open(path_dump_perturbed, 'rb') as fp:
        perturbed_X_text = pkl.load(fp)

    CACHE_DIR = '/workspace1/sebcif/hfcache/'

    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map=f"cuda:{cuda_index}", cache_dir=CACHE_DIR)
    model_instruction = "Determine the sentiment of the tweet below by selecting one word: 'negative', 'neutral', or 'positive'. Keep your response succinct, avoiding explanations."
    inst_format = "[INST] {instruction} [/INST]\n" if model_instruction else ""
    formatted_inst = inst_format.format(instruction=model_instruction)
    prompts = []
    id2label = {0:"negative", 1:"positive", 2:"neutral"}
    for tweet, label_id in zip(perturbed_X_text[country][gender][0], perturbed_X_text[country][gender][1]):
        label = id2label[label_id]
        pre_prompt = f"Tweet:{tweet}\nSentiment:"
        prompt = f"{formatted_inst}{pre_prompt}"
        prompts.append((prompt, label))
    tgt_words = ["Negative", "negative", "Positive", "positive", "Neutral", "neutral"]
    pos_dict = {0:"negative", 1:"negative", 2: "positive", 3:"positive", 4:"neutral", 5:"neutral"}
    tok_tgt_words = [tokenizer.encode(wrd, add_special_tokens=False) for wrd in tgt_words]
    tgt_token_ids = [tok[0] for tok in tok_tgt_words] # we want just the first token of each tokenization
    tgt_indices = torch.tensor(tgt_token_ids, dtype=torch.int).to(f"cuda:{cuda_index}")
    ret_dict = {"country": country, "raw_softmax":[], "label_sum_softmax": [], "pred_label":[], "label": [], "first_token": []}
    for prompt, label in tqdm(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(f"cuda:{cuda_index}")
        outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.0, return_dict_in_generate=True, output_scores=True)
        first_token_index = 0
        scores = outputs.scores[first_token_index] # We extract the first gen token
        try:
            while (first_tok_id := torch.argmax(scores).item()) not in tgt_token_ids:
                first_token_index += 1
                scores = outputs.scores[first_token_index] # Assume a space was generated
        except IndexError:
            print("Generated tokens were not the targeted ones!")
            print(tokenizer.decode(outputs.sequences[:, inputs['input_ids'].shape[1]:][0]))
        first_token = tokenizer.decode(first_tok_id)
        selected_logits = torch.index_select(scores, 1, tgt_indices) # We select the target tokens positions
        raw_softmax = torch.softmax(selected_logits, 1)
        max_i = torch.argmax(raw_softmax).item()
        pred_label = pos_dict[max_i]
        reshaped_softmax = raw_softmax.view(3, 2)
        label_sum_softmax = torch.sum(reshaped_softmax, dim=1)
        #tot_sum_prob = torch.sum(sum_prob_labels).item()
        ret_dict["raw_softmax"].append(raw_softmax.flatten().tolist())
        ret_dict["label_sum_softmax"].append(label_sum_softmax.tolist())
        ret_dict["pred_label"].append(pred_label)
        ret_dict["label"].append(label)
        ret_dict["first_token"].append(first_token)
    with open(path_dump_perturbed+f"_softmax_{country}.pkl", "wb") as out_file:
        pkl.dump(ret_dict, out_file)


