import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

MAX_SEQ_LEN_INI=256

def create_input_array_transfo3(sentences, tokenizer, MAX_SEQ_LEN=MAX_SEQ_LEN_INI):
    """
    Allow to use more than 2 sentences in the same input, with different token_type_ids
    Need to recode properly this function. Like, no need to return_tensors="tf"

    !!!!! MAXIMUM NUMBER OF SENTENCES is 4 !!!!! 
    """
    # Case 2 sentences in input (like for QA, or putting context)
    if ('</s>' not in sentences[0]) or (tokenizer.not_use_token_type_ids):
        encoded_inputs_1 = tokenizer(list(sentences), padding='max_length', truncation=True, return_tensors="tf", max_length=MAX_SEQ_LEN)
        separate_sentences = False
        encoded_inputs_2 = None
    else:
        separate_sentences = True
        # maximum nb of separation tokens found in a sentence
        nb_sent = np.max([k.count('</s>') for k in sentences]) + 1
        more_than_two_sent = nb_sent > 2

        dict_list_sentences = {k: [] for k in range(4)} # TODO: changer le 4 code en dur 

        for sent in sentences:
            if not sent:
                print('***'*100+'ONE TEXT ELEMENT IS NULL !!!'+'***'*100)
                for k in range(nb_sent):
                    dict_list_sentences[k].append('CVxTz')
            else:
                sent_split = sent.split('</s>')
                # in order to have always nb_sep + 1 
                # si impair 
                if (nb_sent - len(sent_split)) != 0:
                    # si impair 
                    if len(sent_split) // 2:
                        sent_split += [None] + (nb_sent - len(sent_split) - 1)*['']
                    else:
                        sent_split += (nb_sent - len(sent_split))*['']
                # sent_split += (nb_sent - len(sent_split))*['']
                for k in range(nb_sent):
                    dict_list_sentences[k].append(sent_split[k])
        
        try:
            encoded_inputs_1 = tokenizer(dict_list_sentences[0],dict_list_sentences[1] , padding='max_length', truncation=True, return_tensors="tf", max_length=MAX_SEQ_LEN)
        except:
            assert True
            print("WARNING: Adding a special pad_token --> <pad>: tokenizer.add_special_tokens({'pad_token': '<pad>'})")
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
            encoded_inputs_1 = tokenizer(dict_list_sentences[0],dict_list_sentences[1] , padding='max_length', truncation=True, return_tensors="tf", max_length=MAX_SEQ_LEN)

        # If there are 3 sentences at least one time, else encoded_inputs_2 in useless everytime
        if more_than_two_sent:
            if dict_list_sentences[3]:
                encoded_inputs_2 = tokenizer(dict_list_sentences[2],dict_list_sentences[3] , padding='max_length', truncation=True, return_tensors="tf", max_length=MAX_SEQ_LEN)
            else: 
                encoded_inputs_2 = tokenizer(dict_list_sentences[2], padding='max_length', truncation=True, return_tensors="tf", max_length=MAX_SEQ_LEN)
        else:
            encoded_inputs_2 = None

    # Adding token_type_ids
    list_encoded_inputs = []
    for encoded_inputs in [encoded_inputs_1, encoded_inputs_2]:

        if encoded_inputs:
            # OLD stuff, I coded the good token_type_ids for RoBERTa (function tokenization_roberta.py) so return_token_type_ids=True when calling tokenizer
            # You don't have token_type_ids with roberta..
            if 'token_type_ids' not in encoded_inputs.keys():
                # encoded_inputs['token_type_ids'] = None
                encoded_inputs['token_type_ids'] = np.zeros(encoded_inputs['input_ids'].shape, dtype=np.int32)
                if separate_sentences:
                    for idx_sent, ts_sent in tqdm(enumerate(encoded_inputs['input_ids'])):
                        bool_sent_2 = 0
                        for idx_tok, tok in enumerate(ts_sent):
                            if tok == tokenizer.sep_token_id:
                                bool_sent_2 += 1
                            if bool_sent_2 == 2: # there are 2 consecutive <s> between 2 sentences 
                                encoded_inputs['token_type_ids'][idx_sent, idx_tok] = 1

            encoded_inputs['token_type_ids'] = tf.convert_to_tensor(encoded_inputs['token_type_ids'])

        list_encoded_inputs.append(encoded_inputs)

    encoded_inputs_1, encoded_inputs_2 = list_encoded_inputs

    # Merging the encoding inputs together 
    if encoded_inputs_2:
        input_ids = encoded_inputs_1['input_ids'].numpy()
        attention_mask = encoded_inputs_1['attention_mask'].numpy()
        token_type_ids = encoded_inputs_1['token_type_ids'].numpy()

        input_ids_2 = encoded_inputs_2['input_ids'].numpy()
        attention_mask_2 = encoded_inputs_2['attention_mask'].numpy()
        token_type_ids_2 = encoded_inputs_2['token_type_ids'].numpy()

        a, b  = np.where(input_ids_2 == tokenizer.sep_token_id)

        for idx_sent in range(input_ids_2.shape[0]):
            pos_max = b[np.max(np.where(a == idx_sent))] + 1
            # case where the are more than 2 sentences
            if pos_max > 3:
                len_sent_1 = np.max(np.where(attention_mask[idx_sent])) + 1

                # maximum size
                pos_max = min(pos_max, MAX_SEQ_LEN - len_sent_1)
                if pos_max:
                    # so the last one has a 0 for token_type_ids (idk why I made it like that)
                    token_type_ids_2[idx_sent,pos_max-1] = -2
                    token_type_ids[idx_sent, len_sent_1-1:len_sent_1-1+pos_max] = token_type_ids_2[idx_sent,:pos_max] + 2

                    attention_mask[idx_sent, len_sent_1:len_sent_1+pos_max] = 1

                    new_input = input_ids_2[idx_sent,:pos_max]
                    new_input[0] = 2
                    input_ids[idx_sent, len_sent_1:len_sent_1+pos_max] = new_input

        encoded_inputs = {'input_ids': tf.convert_to_tensor(input_ids), 
        'attention_mask': tf.convert_to_tensor(attention_mask), 
        'token_type_ids': tf.convert_to_tensor(token_type_ids),
        }

    else:
        encoded_inputs = encoded_inputs_1

    return [encoded_inputs['input_ids'], 
            encoded_inputs['attention_mask'], 
            encoded_inputs['token_type_ids']]

def create_input_array(sentences, tokenizer, MAX_SEQ_LEN=MAX_SEQ_LEN_INI):
    """
    This one for the new transformers version
    """
    # only for roberta and xlm-roberta
    if 'roberta' in tokenizer.name_or_path:
        print('Using transfo3 tokenizer, for RoBERTa models')
        return create_input_array_transfo3(sentences, tokenizer, MAX_SEQ_LEN)
    else:
        print('Using transfo4 tokenizer, for non-RoBERTa models')
        sentences = [k.split(' </s> ') for k in sentences]
        # if there is no ' </s> ', need to input as a list
        if sum([len(s) != 1 for s in sentences]) == 0:
            sentences = [s[0] for s in sentences]
        encoded_inputs = tokenizer(list(sentences), padding='max_length', truncation=True, return_tensors="tf", max_length=MAX_SEQ_LEN)
        return [encoded_inputs['input_ids'], 
                encoded_inputs['attention_mask'], 
                encoded_inputs['token_type_ids']]

def read_csv_val(fn, encoding = 'utf-8'):
	return pd.read_csv(fn, sep='\t', quotechar='"', encoding=encoding, header=0)

def loadTsvData(inputTsvFile, dict_lab={'positive':0, 'negative':1, 'else':2}, enc="utf-8", SPARSE_CATEGORICAL=True, multi_labels=False, cumsum_label_vectors=False, verbose=False, unlabeled=False):
    """
    Load train data. You need the names of the columns on the first line of the CSV...
    """
    df = read_csv_val(inputTsvFile, encoding=enc)

    # normally never happening
    train_tweets = df["tweet"].fillna("CVxTz").values # why 'CVxTz'?
    if unlabeled:
        train_y = np.full(train_tweets.size, -1)
        return (train_tweets, train_y)
    # if unlabeled data only 
    list_unique = df['label'].unique()
    if (len(list_unique) == 1) and (list_unique[0] == -1):
        train_y = df['label'].astype(np.int32).values

    # TODO: no return inside a if... and no function inside a function 
    elif multi_labels:
        def func_mlabels(mlabels):
            y = np.zeros((len(dict_lab))).astype(np.int32)
            for lab in mlabels.split(','):
                y[dict_lab[lab]] = 1
            return y

        train_y = np.concatenate([k[np.newaxis,:] for k in df['label'].astype(str).str.lower().map(func_mlabels).values])

    else:
        if dict_lab == {'regression':0}:
            train_y = df['label'].values
        else:
            # Remark: we can keep else with another values
            if 'else' in dict_lab.keys():
                train_y = df['label'].str.lower().map(dict_lab).fillna(dict_lab['else']).astype(np.int32).values
            else:
                # to be sure that there is no other label than the ones in the dict_label
                if df['label'].str.lower().map(dict_lab).isnull().sum() != 0:
                    print(dict_lab)
                    print(df['label'].str.lower().map(dict_lab).isnull().sum())
                    list_unique = df['label'].unique()
                    print(list_unique)
                    if (len(list_unique) != 1) or (list_unique[0] != '-1'):
                        raise Exception('Error: dict_lab not working well with the dataset')
                    else:
                        print("Dataset of unlabeled data...ok!")

                #impossible if multi-labels
                train_y = df['label'].str.lower().map(dict_lab).astype(np.int32).values

                if len(dict_lab) != len(df['label'].unique()):
                    print('There might be a problem with the labels...')
                    print('dict_lab : {},\ndf["label"].unique() : {}\n\n'.format(dict_lab, df['label'].unique()))

            # create the one-hot vectors
            if not SPARSE_CATEGORICAL:
                b = np.zeros((train_y.size, train_y.max()+1), dtype=np.int32)
                b[np.arange(train_y.size),train_y] = 1
                train_y = b
                if cumsum_label_vectors:
                    train_y = 1 - np.cumsum(train_y, axis=1)[:,:-1]

        if verbose: print('Load "{}", {} samples, {} labels'.format(inputTsvFile.split('/')[-1], len(train_tweets), len(train_y)))
    return (train_tweets, train_y)