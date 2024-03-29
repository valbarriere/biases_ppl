from tensorflow.python.ops import image_ops
from Transformers_utils import read_csv_val, to_csv_val
import numpy as np
import os
from tqdm import tqdm
import tensorflow as tf

# je sais pas a quoi ca sert
AUTOTUNE = tf.data.experimental.AUTOTUNE

MAX_SEQ_LEN_INI=256

def loadTsvData(inputTsvFile, dict_lab={'positive':0, 'negative':1, 'else':2}, enc="utf-8", SPARSE_CATEGORICAL=True, multi_labels=False, cumsum_label_vectors=False, verbose=False):
    """
    Load train data. You need the names of the columns on the first line of the CSV...
    """
    df = read_csv_val(inputTsvFile, encoding=enc)

    # normally never happening
    train_tweets = df["tweet"].fillna("CVxTz").values # why 'CVxTz'?

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
            #return ','.join([str(dict_lab[lab]) for lab in mlabels.split(',')])
            return y

        train_y = np.concatenate([k[np.newaxis,:] for k in df['label'].astype(str).str.lower().map(func_mlabels).values])
        # return (train_tweets, train_y)

    else:
        if dict_lab == {'regression':0}:
            train_y = df['label'].values
        else:
            # Remark: we can keep else with another values
            if 'else' in dict_lab.keys():
                train_y = df['label'].str.lower().map(dict_lab).fillna(dict_lab['else']).astype(np.int32).values
            else:
                # to be sure that there is no other label than the ones in the dict_label
                # print(dict_lab, df['label'].unique(), df['label'].isnull().sum())
                # print(df[df['label'].map(dict_lab).isnull()]['label'])
                if df['label'].str.lower().map(dict_lab).isnull().sum() != 0:
                    print(dict_lab)
                    print(df['label'].str.lower().map(dict_lab).isnull().sum())
                    list_unique = df['label'].unique()
                    print(list_unique)
                    if (len(list_unique) != 1) or (list_unique[0] != '-1'):
                        raise('Error: dict_lab not working well with the dataset')
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


class DatasetLoader():
    """
    Load the data from the tsv files 
    """

    def __init__(self, bool_multimodal, dict_args, model_name, batch_size, multi_labels, cumsum_label_vectors, 
        SPARSE_CATEGORICAL, dict_lab, path_corpus, tokenizer, bool_sample_weight_dev, verbose, using_token_type_ids_roberta=True, force_token_creation=False,
                max_seq_len=128):
        """
        """

        self.bool_multimodal = bool_multimodal
        self.dict_args = dict_args
        self.model_name = model_name

        self.batch_size = batch_size

        self.multi_labels = multi_labels

        self.cumsum_label_vectors = cumsum_label_vectors

        self.SPARSE_CATEGORICAL = SPARSE_CATEGORICAL

        self.dict_lab = dict_lab

        self.path_corpus = path_corpus
        # for the dumps
        self.path_dump = path_corpus + 'dump/'
        self.path_dump_input = self.path_dump + 'inputs/'

        self.tokenizer = tokenizer

        self.bool_sample_weight_dev = bool_sample_weight_dev

        self.verbose = verbose 

        self.using_token_type_ids_roberta = using_token_type_ids_roberta
        
        self.force_token_creation = force_token_creation
        
        self.max_seq_len = max_seq_len
        
        # for selftraining, tokenizer = None
        if self.tokenizer:
            self.tokenizer.not_use_token_type_ids = not bool(self.using_token_type_ids_roberta)
            self.use_token_type_ids_str = ( ('roberta' in model_name) and (self.tokenizer.not_use_token_type_ids) )*'_no_token_type_ids'
        else:
            self.use_token_type_ids_str = ( ('roberta' in model_name) and (not bool(self.using_token_type_ids_roberta)) )*'_no_token_type_ids'


    def load_dataset(self, inputFile_tr):
        """
        """

        train_tweets, train_y = loadTsvData(self.path_corpus + inputFile_tr, self.dict_lab, SPARSE_CATEGORICAL=self.SPARSE_CATEGORICAL, 
                                            multi_labels = self.multi_labels, cumsum_label_vectors=self.cumsum_label_vectors, verbose=self.verbose)       # load the data

        # to get the already tokenized data
        str_id_saving_files_tr = '_{}_{}'.format(self.model_name,inputFile_tr) + self.use_token_type_ids_str
        
        # for the selftraining --> useless, if not existing then the tokenizer is just a None
        # if not hasattr(self.tokenizer, 'not_use_token_type_ids'):
            # self.tokenizer.not_use_token_type_ids = not bool(self.using_token_type_ids_roberta)

        train_x = create_or_load_input(self.path_dump_input, str_id_saving_files_tr, train_tweets, self.tokenizer, self.force_token_creation, max_seq_len=self.max_seq_len)

        if self.bool_multimodal:

            a, b, c = train_x
            # TODO: change since only work for ResNet trained for emotion recognition
            # scores_or_feats = 'scores'  if 'scores' in str_id_training else 'features'
            # d = load_MM_input_from_disk_old(path_dump = path_dump, scores_or_feats=scores_or_feats, type_ds = 'one_target_train')


            # d, image_size = load_MM_array_or_path(path_corpus = path_corpus, name_tsv = inputFile_tr, path_dump_input = path_dump_input, 
                # use_tf_dataset = dict_args['use_tf_dataset'], model_name_vision = dict_args['model_name_vision'])
            d, image_size = load_MM_array_or_path(path_corpus = self.path_corpus, name_tsv = inputFile_tr, path_dump_input = self.path_dump_input, 
                use_tf_dataset = self.dict_args['use_tf_dataset']*self.dict_args['path_in_tfds'], model_name_vision = self.dict_args['model_name_vision'])                
            train_x = (a, b, c, d)

            self.image_size = image_size

        else:
            self.image_size = None

        return train_x, train_y


    def load_train(self, list_inputFile_tr):
        """
        list_inputFile_tr: list of paths containing the names of the tsv files to load
        """

        ## TRAIN DATA
        train_y_multidataset = []
        # tt = {0: [], 2: [], 3: []}
        train_x_multidataset = [ [], [], [] ]
        #train_x_multidataset = None
        image_size = None
        if self.bool_multimodal: train_x_multidataset.append([])
        for inputFile_tr in list_inputFile_tr:

            train_x, train_y = self.load_dataset(inputFile_tr)
            train_y = [k if k != '-1'  else -1 for k in train_y]

            # if several train datasets list of list of the features to concatenate the features together
            for k_val, val in enumerate(train_x):   
                train_x_multidataset[k_val].append(val)

            #if not train_x_multidataset:
            #    train_x_multidataset = train_x.copy()
            #else:
            #    for k_val, val in train_x.items():
            #       if val: # for RoBERTa no token_type_ids, val is None
            #            train_x_multidataset[k_val] = np.concatenate((train_x_multidataset[k_val], val), axis=0)


            train_y_multidataset.append(train_y)

        train_y = np.concatenate(train_y_multidataset, axis = 0)
        # print(train_x_multidataset, train_x)
        train_x = tuple([np.concatenate(train_x_multidataset_type_feat, axis=0) for train_x_multidataset_type_feat in train_x_multidataset])

        if self.dict_args['use_tf_dataset']:
            # train_ds = create_tf_dataset(train_x, train_y, image_size, batch_size=batch_size)
            # train_ds = create_tf_dataset(train_x, train_y, image_size, batch_size=batch_size, path_in_tfds=dict_args['path_in_tfds']) 
            train_ds = create_tf_dataset_DA(train_x, train_y, self.image_size, batch_size=batch_size, path_in_tfds=self.dict_args['path_in_tfds'], data_augmentation_images=True) 

            self.train_data = train_ds
            return train_ds

        else:

            self.train_data = (train_x, train_y)
            return train_x, train_y


    def load_dev(self, inputFile_dev):
        """
        path_corpus + inputFile_dev
        """

        dev_x, dev_y = self.load_dataset(inputFile_dev)

        # Used to weight the languages on the dev set when it's unbalances and you want an homogenous model (only used once)
        if self.bool_sample_weight_dev:

            df = read_csv_val(self.path_corpus + inputFile_dev)
            array_lang = df.lang.values
            # put a weight proportionally inverse to the frequency of the lang in the dev set (to not favorise a language over another) 
            dict_ct = {k:v for k,v in zip(np.unique(array_lang, return_counts=True)[0], np.unique(array_lang, return_counts=True)[1])}
            avg_samples = np.sum(list(dict_ct.values()))/len(dict_ct)
            dict_ct = {k:avg_samples/v for k, v in dict_ct.items()}
            sample_weight_validation = np.array([dict_ct[lan] for lan in array_lang])

            dev_data = (dev_x, dev_y, sample_weight_validation)                
        else:
            dev_data = (dev_x, dev_y)                # pack into (x, y) tuple 
            sample_weight_validation=None

        self.dev_data = (dev_x, dev_y)

        if self.dict_args['use_tf_dataset']:
            # dev_ds = create_tf_dataset(dev_x, dev_y, image_size, sample_weight_validation, batch_size=batch_size)
            # dev_ds = create_tf_dataset(dev_x, dev_y, image_size, sample_weight=sample_weight_validation, batch_size=batch_size, path_in_tfds=dict_args['path_in_tfds'])
            dev_ds = create_tf_dataset_DA(dev_x, dev_y, self.image_size, sample_weight=sample_weight_validation, batch_size=self.batch_size, path_in_tfds=self.dict_args['path_in_tfds'], data_augmentation_images=False)

            dev_data = dev_ds

            self.dev_ds = dev_ds
            return dev_ds

        return self.dev_data


    def calculate_class_weight(self, bool_class_weight):
        """
        """

        if bool_class_weight:
            # if class_weight
            # TODO: be careful this does not work with multi-labels, bug            
            train_y = self.train_data[1]
            # deal with non labeled data
            train_y = train_y[train_y != -1]
            # train_y = [k for k in train_y if k != -1]

            unique_labels, counts_labels = np.unique(train_y, return_counts=True, axis = 0)
            if len(train_y.shape) == 1: # sparse categorical
                label_id = unique_labels
            else:
                label_id = np.argmax(unique_labels, axis = -1)
                
            # print(unique_labels, counts_labels)

            if not self.SPARSE_CATEGORICAL:
                # the way the vectors are constructed means that the first become the last
                class_nb = [counts_labels[k] for k in np.sort(label_id)[::-1]] 
            else:
                class_nb = [counts_labels[k] for k in np.sort(label_id)]

            # if we dont care about neutral --> Dailydialog
            # if "remove_labels" in dict_args.keys():
            for remove_label in self.dict_args['remove_labels']:
                class_nb[dict_lab[remove_label]] = 0

            class_nb /= np.sum(class_nb)
            print('Using class_weight, percentage is:')
            print(class_nb)
            # class_weight = [1/k for k in class_nb]
            val_cw_removed = 1e-1
            class_weight = [1/k if k else val_cw_removed for k in class_nb]
            class_weight = {lab : val for lab, val in enumerate(class_weight)}
        else:
            class_weight = None

        self.class_weight = class_weight

        return class_weight

    def load_test(self, input_data_File):
        """
        """

        (test_x, test_y) = self.load_dataset(input_data_File)

        self.test_data = (test_x, test_y)

        if self.dict_args['use_tf_dataset']:
            test_x = create_tf_dataset_DA(test_x, test_y, image_size=self.image_size, batch_size = 32, path_in_tfds=self.dict_args['path_in_tfds'], data_augmentation_images=False)

        return test_x, test_y


def change_data_form_t5(train_x, is_multimodal=False):
    """
    Change the form on inputs from list to dict. Which is more robust actually... 
    """
    if is_multimodal:
        return {'input_word_ids': train_x[0], 'input_mask': train_x[1], 'decoder_input_ids': train_x[0], 'input_img': train_x[-1]}
    else:
        return {'input_word_ids': train_x[0], 'input_mask': train_x[1], 'decoder_input_ids': train_x[0]}

def create_input_array_old(sentences, tokenizer, MAX_SEQ_LEN=MAX_SEQ_LEN_INI):
    """
    """
    # Case 2 sentences in input (like for QA, or putting context)

    if '</s>' not in sentences[0]:
        encoded_inputs = tokenizer(list(sentences), padding='max_length', truncation=True, return_tensors="tf", max_length=MAX_SEQ_LEN)
        separate_sentences = False
    else:
        separate_sentences = True
        nb_sep = np.max([k.count('</s>') for k in sentences])
        sentences_1 = []
        sentences_2 = []
        sentences_3 = []
        sentences_4 = []
        for sent in sentences:
            nb_sep = np.max([k.count('</s>') for k in sentences])
            if not sent:
                print('***'*100+'ONE TEXT ELEMENT IS NULL !!!'+'***'*100)
                sent_1, sent_2 = 'CVxTz', 'CVxTz'
            else:
                sent_1, sent_2 = sent.split('</s>')
            sentences_1.append(sent_1)
            sentences_2.append(sent_2)

        encoded_inputs = tokenizer(sentences_1, sentences_2, padding='max_length', truncation=True, return_tensors="tf", max_length=MAX_SEQ_LEN, return_token_type_ids=True)

    # OLD stuff, I coded the good token_type_ids for RoBERTa (function tokenization_roberta.py) so return_token_type_ids=True when calling tokenizer
    # You don't have token_type_ids with roberta..
    if 'token_type_ids' not in encoded_inputs.keys():
        # encoded_inputs['token_type_ids'] = None
        encoded_inputs['token_type_ids'] = np.zeros(encoded_inputs['input_ids'].shape)
        if separate_sentences:
            for idx_sent, ts_sent in tqdm(enumerate(encoded_inputs['input_ids'])):
                bool_sent_2 = 0
                for idx_tok, tok in enumerate(ts_sent):
                    if tok == tokenizer.sep_token_id:
                        bool_sent_2 += 1
                    if bool_sent_2 == 2: # there are 2 consecutive <s> between 2 sentences 
                        encoded_inputs['token_type_ids'][idx_sent, idx_tok] = 1

    return [encoded_inputs['input_ids'], 
            encoded_inputs['attention_mask'], 
            encoded_inputs['token_type_ids']]


def create_input_array(sentences, tokenizer, MAX_SEQ_LEN=MAX_SEQ_LEN_INI):
    """
    This one for the new transformers version
    """
    # only for roberta and xlm-roberta
    if 'roberta' in tokenizer.name_or_path:
    #if True:
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
        
def create_input_array_transfo3(sentences, tokenizer, MAX_SEQ_LEN=MAX_SEQ_LEN_INI):
    """
    Allow to use more than 2 sentences in the same input, with different token_type_ids
    Need to recode properly this function. Like, no need to return_tensors="tf"

    !!!!! MAXIMUM NUMBER OF SENTENCES is 4 !!!!! 
    """
    # Case 2 sentences in input (like for QA, or putting context)
    if ('</s>' not in sentences[0]) or (tokenizer.not_use_token_type_ids):
    # if '</s>' not in sentences[0]:
        encoded_inputs_1 = tokenizer(list(sentences), padding='max_length', truncation=True, return_tensors="tf", max_length=MAX_SEQ_LEN)
        separate_sentences = False
        encoded_inputs_2 = None
    else:
        separate_sentences = True
        # maximum nb of separation tokens found in a sentence
        nb_sent = np.max([k.count('</s>') for k in sentences]) + 1
        more_than_two_sent = nb_sent > 2

        dict_list_sentences = {k: [] for k in range(4)} # TODO: changer le 4 code en dur 
        # dict_list_sentences = {k: [] for k in range(nb_sent)}

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

# saving to and loading from disk
def save_input_to_disk(train_sentences, dev="", path_dump = './', File_to_convert=''):
    ids, masks, segments = train_sentences

    # create folder if needed
    if not os.path.isdir(path_dump + dev):
        os.mkdir(path_dump + dev)
    np.save(path_dump + dev + "input_ids" + File_to_convert + ".npy", ids)
    np.save(path_dump + dev + "input_masks" + File_to_convert + ".npy", masks)
    np.save(path_dump + dev + "input_segments" + File_to_convert + ".npy", segments)
    print('Files {} saved...'.format(dev+File_to_convert[1:]))


def load_input_from_disk(dev="", path_dump = './', File_to_convert=''):
    ids = np.load(path_dump + dev + "input_ids" + File_to_convert + ".npy")
    masks = np.load(path_dump + dev + "input_masks" + File_to_convert + ".npy")
    # Sometimes no segment is needed (like RoBERTa model)
    try:
        segments = np.load(path_dump + dev + "input_segments" + File_to_convert + ".npy")
    except:
        segments=None
    print("*.npy files related to {} are loaded...".format(dev+File_to_convert))
    return (ids, masks, segments)

def create_or_load_input_old(path_dump_input, str_id_saving_files_tr, train_tweets, tokenizer):
    """
    Function to load the input if it exists
    """
    if not os.path.isfile(path_dump_input + "input_ids" + str_id_saving_files_tr + ".npy"): # if the file not exsiting create them all (long process)
        train_x = create_input_array_old(train_tweets, tokenizer)  # convert to BERT inputs
        save_input_to_disk(train_x, path_dump = path_dump_input, File_to_convert = str_id_saving_files_tr)
    else:
        train_x = load_input_from_disk(path_dump = path_dump_input, File_to_convert = str_id_saving_files_tr)

    return train_x

def create_or_load_input(path_dump_input, str_id_saving_files_tr, train_tweets, tokenizer, force_token_creation=False, max_seq_len=128):
    """
    New version, using Hugging Face tokenizer 
    """
    str_id_saving_files_tr += "-hftokenizer"

    # adding the max_seq_len in the filename, useful for long files 
    str_id_saving_files_tr = ("_len%d"%max_seq_len)*(max_seq_len != 128) + str_id_saving_files_tr

    if (not os.path.isfile(path_dump_input + "input_ids" + str_id_saving_files_tr + ".npy")) or force_token_creation: # if the file not exsiting create them all (long process)
        train_x = create_input_array(train_tweets, tokenizer, MAX_SEQ_LEN = max_seq_len)  # convert to BERT inputs
        save_input_to_disk(train_x, path_dump = path_dump_input, File_to_convert = str_id_saving_files_tr)
    else:
        train_x = load_input_from_disk(path_dump = path_dump_input, File_to_convert = str_id_saving_files_tr)

    return train_x

def load_MM_input_from_disk_old(dev="", path_dump = './', scores_or_feats='scores', type_ds = 'one_target_test'):
    """
    Old version used with frozen deepemotion network generating features only
    """
    input_img = np.load(path_dump + dev + "input_img_{}_{}.npy".format(scores_or_feats, type_ds))
    print("*.npy files related to {} are loaded...".format(scores_or_feats + '_' + type_ds))
    return input_img   

def load_MM_input_from_disk_CrisisMMD(path_dump = './', scores_or_feats='scores', type_ds = 'one_target_test'):
    """
    New version for CrisisMMD 
    images_efficient2_task_humanitarian_text_img_agreed_lab_dev.npy
    """
    EFFICIENT_TYPE = 2
    type_task = 'humanitarian'
    # input_img = np.load(path_dump + dev + "input_img_{}_{}.npy".format(scores_or_feats, type_ds))
    # TODO: a changer car ya pas de text img agreed pour TSA, on est plus sur CrisisMMD
    input_img = np.load(path_dump + "images_efficient%d_task_%s_text_img_agreed_lab_%s.npy"%(EFFICIENT_TYPE, 
                                                                                       type_task, type_ds))
    print("*.npy files related to {} with shape {} are loaded...".format(scores_or_feats + '_' + type_ds, input_img.shape))
    return input_img   

def load_MM_input_from_disk(path_corpus, name_tsv, path_dump_input):
    """
    New version for all corpora 
    Ne marche pas vraiment car l'image depend du type de reseau non ??? 
    ou au moins de la taille dans lequel elle a ete resize avant ?? 

    """

    path_tsv_data = path_corpus + name_tsv
    # no need for model name here, 
    str_id_saving_files = '_{}'.format(name_tsv)

    file_path = path_dump_input + "input_img" + str_id_saving_files + ".npy"

    if not os.path.isfile(file_path):
        df = read_csv_val(path_tsv_data, encoding="utf-8")
        assert False, '%s not existing...create it!'%file_path
        for fn in df.image_path.values:
            pass
    else:
        input_img = np.load(file_path)

    print("*.npy image files related to {} are loaded...".format(str_id_saving_files))
    return input_img   


def path_to_image(path, image_size, num_channels=3, interpolation='bilinear', resize_image=True):
    """
    resize_image set to false when the image has already been resized and you don't have to do it in the loader 
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_image(
        img, channels=num_channels, expand_animations=False)
    if resize_image:
        img = image_ops.resize_images_v2(img, image_size, method=interpolation)
        img.set_shape((image_size[0], image_size[1], num_channels))
    return img

def resize(image, image_size):
    image = tf.image.resize(image, [image_size[0], image_size[1]])
    # image = tf.cast(image, tf.float32)
    # pourquoi pas image = tf.cast(image, tf.uint8) ????
    image = tf.cast(image, tf.uint8)
    # image = (image / 255.0) # scaling inside the model ?? 
    return image

def path_to_image2(path, image_size):
    """
    N'est plus utilise
    tf.image est mieux que tf.io.decode_png / tf.io.decode_jpg
    """
    if '.png' in path:
        img = tf.io.decode_png(tf.io.read_file(path), channels=3)
    else:
        img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    return resize(img, image_size)

def path_to_image3(path, image_size):
    img = tf.image.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    return resize(img, image_size)

def create_tf_dataset(X, y=[], image_size=None, sample_weight=None, batch_size=32, path_in_tfds=True, SPARSE_CATEGORICAL=True):
    """
    image_size only used when multimodal
    should be the (IMG_SIZE_BEFORE_CROP, IMG_SIZE_BEFORE_CROP)
    If image has already been resized a priori, then resize_image=False and the cropping or resizing will occur inside the keras.preprocessing layer  
    """
    # if no y, because just prediction, we don't care
    if not len(y):
        y=np.zeros(len(X[0]))

    label_mode = "int" if SPARSE_CATEGORICAL else 'categorical'
    num_classes = len(np.unique(y)) # wouldn't work for zero-shot 
    ds_labels = tf.python.keras.preprocessing.dataset_utils.labels_to_dataset(y, label_mode, num_classes)

    # if only text 
    if len(X) <= 3:
        if not sample_weight:
            ds = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(X), ds_labels))
        else:
            ds = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(X), ds_labels, tf.data.Dataset.from_tensor_slices(sample_weight)))
    # if multimodal 
    elif len(X) == 4:
        image_ds = tf.data.Dataset.from_tensor_slices(X[-1])
        if path_in_tfds:
            # image_ds = image_ds.map(lambda x: path_to_image(x, image_size))

            # zip the text and the image + the labels altogether
            if not sample_weight:
                # ds = tf.data.Dataset.zip(((tf.data.Dataset.from_tensor_slices(X[:-1]), image_ds), ds_labels))
                ds = tf.data.Dataset.zip(( tf.data.Dataset.from_tensor_slices(X).map(lambda x, y, z, t: (x, y, z, path_to_image3(t, image_size))), ds_labels))
            else:
                ds = tf.data.Dataset.zip(((tf.data.Dataset.from_tensor_slices(X[:-1]), image_ds), 
                    ds_labels, tf.data.Dataset.from_tensor_slices(sample_weight)))
        else:
            ds = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(X), ds_labels))
            # ds = tf.data.Dataset.zip(((tf.data.Dataset.from_tensor_slices(X[:-1]), image_ds), ds_labels))

        # ds_labels change par tf.data.Dataset.from_tensor_slices(y) .... pourquoi ???? 
        # if not sample_weight: 
            # ds = tf.data.Dataset.zip(((tf.data.Dataset.from_tensor_slices(X[:-1]), ds_path_images), tf.data.Dataset.from_tensor_slices(y)))
        # else:
            # ds = tf.data.Dataset.zip(((tf.data.Dataset.from_tensor_slices(X[:-1]), ds_path_images), 
                # tf.data.Dataset.from_tensor_slices(y), tf.data.Dataset.from_tensor_slices(sample_weight)))

    ds = ds.batch(batch_size)
    ds.prefetch(buffer_size=AUTOTUNE)
    print('Dataset: ', ds)

    return ds


def generate_coord(size_max, size_min):
    """
    """ 
    y2 = tf.random.uniform([1], size_min, size_max)[0]
    x2 = tf.random.uniform([1], size_min, size_max)[0]
    x1 = tf.random.uniform([1], 0, tf.reduce_max(x2-size_min, 0))[0]
    y1 = tf.random.uniform([1], 0, tf.reduce_max(y2-size_min,0))[0]

    return x1, y1, x2, y2

def path_to_image_randomcrop_val(path, image_size=(300,300), scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), 
                                 num_channels=3, interpolation='bilinear', verbose=False):
    """
    Marche sur un dataset entier !! 
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_image(
        img, channels=num_channels, expand_animations=False)

    minval = tf.sqrt(scale)[0]
    maxval = tf.sqrt(scale)[1]

    size_min = tf.random.uniform([1], minval, maxval)
    size_max = scale[1]

    x1, y1, x2, y2 = generate_coord(size_max, size_min)
    w = (x2-x1)
    h = (y2-y1)
    aspect_ratio = w/h
    area = w*h

    while (area < size_min) or ((aspect_ratio<ratio[0]) or (aspect_ratio>ratio[1])):
        x1, y1, x2, y2 = generate_coord(size_max, size_min)
        w = (x2-x1)
        h = (y2-y1)
        aspect_ratio = w/h
        area = w*h
    im = tf.image.crop_and_resize(img[None, ...], [[y1, x1, y2, x2]], [0], crop_size=image_size)[0]
    im = tf.cast(im, np.uint8)
    # if verbose:
        # print(im.shape, x1, y1, x2, y2)
        # plt.imshow(im)
        # plt.show()
    return im 

# TO CHANGE --> Cannot be coded in hard 
# layer_random_crop = tf.keras.layers.experimental.preprocessing.RandomCrop(img_size[0], img_size[1])
layer_random_crop ={380: tf.keras.layers.experimental.preprocessing.RandomCrop(380, 380),
                   260: tf.keras.layers.experimental.preprocessing.RandomCrop(260, 260)}

def path_to_image_centercrop(path_img, img_size, verbose=False):
    """
    Avec training=False fait un centercrop, meme si on utilise randomcrop
    """
    img = tf.io.read_file(path_img)
    img = tf.image.decode_image(
        img, channels=3, expand_animations=False)    
    # print(img.shape)

    # im = tf.cast(layer_random_crop(img[None, ...], training= False)[0], np.uint8)
    im = tf.cast(layer_random_crop[img_size[0]](img[None, ...], training= False)[0], np.uint8)

    if verbose:
        print(im.shape)
        plt.imshow(im)
        plt.show()

    return im

def create_tf_dataset_DA(X, y=[], image_size=None, sample_weight=None, batch_size=32, path_in_tfds=True, SPARSE_CATEGORICAL=True, data_augmentation_images=False):
    """
    image_size only used when multimodal
    should be the (IMG_SIZE_BEFORE_CROP, IMG_SIZE_BEFORE_CROP)
    If image has already been resized a priori, then resize_image=False and the cropping or resizing will occur inside the keras.preprocessing layer  
    data_augmentation
    """
    # if no y, because just prediction, we don't care
    if not len(y):
        y=np.zeros(len(X[0]))

    label_mode = "int" if SPARSE_CATEGORICAL else 'categorical'
    num_classes = len(np.unique(y)) # wouldn't work for zero-shot 
    ds_labels = tf.python.keras.preprocessing.dataset_utils.labels_to_dataset(y, label_mode, num_classes)

    # if only text 
    if len(X) <= 3:
        if not sample_weight:
            ds = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(X), ds_labels))
        else:
            ds = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(X), ds_labels, tf.data.Dataset.from_tensor_slices(sample_weight)))
    # if multimodal 
    elif len(X) == 4:
        image_ds = tf.data.Dataset.from_tensor_slices(X[-1])
        if path_in_tfds:
            if data_augmentation_images:
                fun_load_img = lambda x,y: path_to_image_randomcrop_val(x,y)
                # image_ds = image_ds.map(lambda x: path_to_image_randomcrop_val(x, image_size))
            else:
                # image_ds = image_ds.map(lambda x: path_to_image_centercrop(x))
                # fun_load_img = path_to_image_centercrop
                fun_load_img = lambda x,y: path_to_image_centercrop(x,y)
            # zip the text and the image + the labels altogether
            if not sample_weight:
                # ds = tf.data.Dataset.zip(((tf.data.Dataset.from_tensor_slices(X[:-1]), image_ds), ds_labels))
                # ds = tf.data.Dataset.zip(( tf.data.Dataset.from_tensor_slices(X).map(lambda x, y, z, t: (x, y, z, path_to_image3(t, image_size))), ds_labels))
                ds = tf.data.Dataset.zip(( tf.data.Dataset.from_tensor_slices(X).map(lambda x, y, z, t: (x, y, z, fun_load_img(t, image_size))), ds_labels))

            else:
                ds = tf.data.Dataset.zip(((tf.data.Dataset.from_tensor_slices(X[:-1]), image_ds), 
                    ds_labels, tf.data.Dataset.from_tensor_slices(sample_weight)))
        else:
            ds = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(X), ds_labels))
            # ds = tf.data.Dataset.zip(((tf.data.Dataset.from_tensor_slices(X[:-1]), image_ds), ds_labels))

    ds = ds.batch(batch_size)
    ds.prefetch(buffer_size=AUTOTUNE)
    print('Dataset: ', ds)

    return ds

def load_images_paths(path_corpus, name_tsv, path_dump_input, model_name_vision):
    """
    Return an array containing the path of the images to use (columns : path_img_for_learning)
    """
    path_tsv_data = path_corpus + name_tsv
    # no need for model name here, 
    str_id_saving_files = '_{}'.format(name_tsv)

    df = read_csv_val(path_tsv_data, encoding="utf-8")

    if model_name_vision == "efficient0":
        image_size = (224, 224)
    elif model_name_vision == "efficient1":
        image_size = (240, 240)
    elif model_name_vision == "efficient2":
        image_size = (260, 260)
    elif model_name_vision == "efficient3":
        image_size = (300, 300)
    elif model_name_vision == "efficient4":
        image_size = (380, 380)
    elif model_name_vision == "efficient5":
        image_size = (456, 456)

    # TODO 
    # added for debugging with path_to_img3 and it worked
    # image_size = (int(4/3*image_size[0]), int(4/3*image_size[1]))
    # removed now for path_to_random_crop_val
    # also for debugging --> only work for CrisisMMD
    if not hasattr(df, 'path_img_for_learning'):
        df['path_img_for_learning'] = path_corpus + df['image']

    # return df.path_img_for_learning.map(lambda x: x.replace('_rescaled_side_380', '')), image_size
    return df.path_img_for_learning, image_size

def load_MM_array_or_path(path_corpus, name_tsv, path_dump_input, use_tf_dataset, model_name_vision):
    """
    """
    # Load the images path that will be used by the tf.Dataset
    if use_tf_dataset:
        d, image_size = load_images_paths(path_corpus = path_corpus, name_tsv = name_tsv, path_dump_input = path_dump_input, 
        model_name_vision = model_name_vision)
    else:
        d = load_MM_input_from_disk(path_corpus = path_corpus, name_tsv = name_tsv, path_dump_input = path_dump_input)
        image_size = None

    return d, image_size