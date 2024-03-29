# -*- coding: utf-8 -*-
"""
Created on Thu May  14 22:32:13 2020

@author: Valentin

Script for training the transformers on sentiment analysis over tweets. 

This is the developpement branch. 

This version MM_class is the last version September 2021. It uses other scripts: Transformers_data.py, Transformers_utils.py, Transformers_metrics.py and Transformers_class.py

!!!! WARNING !!!!
Metrics_F1 won't work if multimdodal AND use_tf_dataset
!!!! WARNING !!!!

Version BM de Janvier 2023 

"""
# Prevent from using GPU
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
sys.path.insert(0,'/home/barriva/.local/bin/')

# PATH_DATA = '/home/barriva/data/'
PATH_DATA = '/eos/jeodpp/data/projects/REFOCUS/data/Valentin/'

try: 
    import tensorflow as tf
    # import tensorflow_hub as hub # not used anymore 
    from tensorflow.keras import layers
except:
    raise Exception('Error importing tensorflow')
    # ! pip install "tensorflow>=2.0.0"
    # ! pip install --upgrade tensorflow-hub
    import tensorflow as tf 
    # import tensorflow_hub as hub
    from tensorflow.keras import layers

limit_GPU_memory=False
if limit_GPU_memory:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        gpu_max_memory = 32000
        if gpu_max_memory:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_max_memory)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            print('GPU #%d has RAM limited to %d'%(0, gpu_max_memory))

try:
    from transformers import *
    from transformers import __version__ as vtrans
except:
    raise Exception('Error importing transformer')
    # ! pip install transformers  
    from transformers import *
    from transformers import __version__ as vtrans

from Transformers_class import compile_and_return_model

from Transformers_utils import read_csv_val, to_csv_val, str2bool, reduce_size_str_training_id, return_filename_of_pretrained_model, return_dict_labels_for_training, return_str_paths

from Transformers_data import DatasetLoader, change_data_form_t5, create_tf_dataset_DA, load_MM_array_or_path

# utilises pour le test uniquement !! on pourrait les tej 
from Transformers_data import create_or_load_input, create_tf_dataset_DA, loadTsvData

from Transformers_metrics import Metrics_F1, calculate_and_print_metrics, ReturnBestEarlyStopping, create_y_multilabels

TRANSFO4 = vtrans[0] == '4'

# Regarding the platform we are using 
IS_COLAB = False # in colab, other paths 
IS_LOCAL = False # local
IS_JHUB = True

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
from shutil import copytree
import json
from glob import glob
import re

print("TensorFlow Version:",tf.__version__)
# print("Hub version: ",hub.__version__) # not used anymore 
# print("BERT version: ",bert.__version__)
print("transformers version: ",vtrans)

# random seed for reproducibility
import random
random.seed(42)

# TODO: put all the variables inside that 
import argparse

def return_callbacks(USE_TENSORBOARD, DROP_MODEL, str_id_training, path_corpus, update_freq_tboard, 
    trained_modelFilePath, monitor_value, dict_args, 
    patience, monitor_value_es, min_delta):
    """
    """

    # tensorboard dumps
    if USE_TENSORBOARD:
        # TODO: look if I need those variables ; what exactly can I do with them? Now I just have the loss and accuracy
        write_graph=False
        write_images=False
        histogram_freq=0
        id_cut = str_id_training.find('all-epoch')
        #str_id_training_TB = str_id_training[:id_cut-1] + '/' + str_id_training[id_cut:] # test: maybe file name too long
        #str_id_training_TB = str_id_training.replace('sb-10k_train_val.tsv_multilingual-AND-Sentipolc16-trainval_general.csv_val_multilingual-AND-intertass2018-ES-train-tagged.tsv_val_multilingual-AND-TASS2019_country_ES_train.tsv_val_multilingual-AND-DEFT2015-task1-trainval_val.tsv_multilingual-AND-SemEval2018-task1-Valence-c-En-train.txt_val_multilingual_multilingual_devval.tsv', 'all-real-datasets-multilingual')
        log_dir = path_corpus + "logs/fit/" + str_id_training[1:] +"/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        print('Tensorboard dumps {}'.format(log_dir))
        tensorboard_callback = [tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=histogram_freq, 
                        update_freq=update_freq_tboard, write_graph=write_graph, write_images=write_images)]
    else:
        print('Not using Tensorboard...')
        tensorboard_callback=None  

    # Save best model regarding to the val loss
    if DROP_MODEL: 
        # save_weights_only=True mandatory otherwise it does not work for transformers models
        save_model = tf.keras.callbacks.ModelCheckpoint(
            trained_modelFilePath[:-5]+'-best-model-checkpoint-{}.hdf5'.format(monitor_value), monitor=monitor_value, verbose=0, save_best_only=dict_args['save_best_model_only'],
            save_weights_only=True, mode='auto', save_freq='epoch')
        if tensorboard_callback:
            tensorboard_callback.append(save_model)
        else:
            tensorboard_callback=[save_model]
    else:
        print('Not using ModelCheckpoint...')

    # early stopping callback on monitor_value
    if patience > 0:
        es_callback = ReturnBestEarlyStopping(
            monitor=monitor_value_es, min_delta=min_delta, patience=patience, verbose=0, mode='auto',
            baseline=None, restore_best_weights=False
        )

        if tensorboard_callback:
            tensorboard_callback.append(es_callback)
        else:
            tensorboard_callback=[es_callback]
    else:
        print('Not using EarlyStopping...')

    return tensorboard_callback

def fineTuneBERT(trained_modelFilePath, model, tokenizer, path_corpus, dict_lab, SPARSE_CATEGORICAL=False, model_name='any_model', 
                 cumsum_label_vectors=False, list_inputFile_tr=['AllCombinedforBERT.tsv'], inputFile_dev='tweeti-b.dev.dist.Tweets.tsv.utf8.all', 
                 bool_with_dev=False, bool_sample_weight_dev=False, USE_TENSORBOARD=True, update_freq_tboard=100, batch_size=32,
                 nb_epochs=1, steps_per_epoch=None, bool_class_weight=None, validation_split=0.2, PRINT_F1=False, 
                 DROP_MODEL=True, bool_multimodal=True, monitor_value='val_loss', patience=5, min_delta=0.005, monitor_value_es='val_loss', 
                 test_on_epoch_end = None,verbose=False, display_pg_bar=True, multi_labels = False, dict_args={}):
    """
    Finetune `model` and save its weight in trained_modelFilePath
    model has already been loaded as the the transformer model stored in `modelFilePath`, 
    Fine-tune it on the tsv files in the `list_inputFile_tr`, with `inputFile_dev` as dev set
    store the fine-tuned model on `modelFilePath`, overwrititng the original `modelFilePath` file.
    """

    # type_model and type of file used for test/train 
    # TODO: take into account the dataset used to train the model    
    # list_inputFile_tr_in_id = '-AND-'.join(list_inputFile_tr)
    # str_id_saving_files_tr = '_{}_{}'.format(model_name,list_inputFile_tr_in_id)
    # str_id_saving_files_tr = '_{}_{}'.format(model_name,inputFile_tr)
    # str_id_training = '_{}_{}_{}'.format(model_name,inputFile_tr,inputFile_dev) # in trainingBERT

    str_id_training = '_'.join([''] + os.path.splitext(trained_modelFilePath.split('/')[-1])[0].split('_')[1:])

    # for the dumps
    path_dump = path_corpus + 'dump/'
    path_dump_input = path_dump + 'inputs/'
    
    # Only this I've seen with for Galactica 
    if (not hasattr(tokenizer, 'pad_token')) or (tokenizer.pad_token is None):
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            print('Adding special tokens: pas, bos, eos, unk. Only for Galactica?')
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
            tokenizer.add_special_tokens({'bos_token': '<s>'})
            tokenizer.add_special_tokens({'eos_token': '</s>'})
            tokenizer.add_special_tokens({'unk_token': '<unk>'})
            # model.resize_token_embeddings(len(tokenizer))

    dataset_loader = DatasetLoader(bool_multimodal, dict_args, model_name, batch_size, multi_labels, cumsum_label_vectors, 
        SPARSE_CATEGORICAL, dict_lab, path_corpus, tokenizer, bool_sample_weight_dev, verbose, using_token_type_ids_roberta=dict_args['using_token_type_ids_roberta'], force_token_creation=dict_args['force_token_creation'], max_seq_len = dict_args['max_seq_len'])

    train_data = dataset_loader.load_train(list_inputFile_tr)
    if type(train_data) == tuple:
        train_x, train_y = train_data

    tensorboard_callback = return_callbacks(USE_TENSORBOARD, DROP_MODEL, str_id_training, path_corpus, update_freq_tboard, 
    trained_modelFilePath, monitor_value, dict_args, 
    patience, monitor_value_es, min_delta)

    # if class_weight
    # TODO: be careful this does not work with multi-labels, bug
    class_weight = dataset_loader.calculate_class_weight(bool_class_weight)

    # Train model with or without dev data
    # I think MK was doing a 2 step process, training without dev for 1 epoch, then with dev for 4 epochs
    if bool_with_dev:

        dev_ds = dataset_loader.load_dev(inputFile_dev)
        # if not using tfds 
        if type(dev_ds) == tuple:
            dev_x, dev_y = dataset_loader.dev_data

        # Print the F1 on the validation set at the end of each epoch 
        if PRINT_F1:   
            metrics_f1 = Metrics_F1(validation_data = dataset_loader.dev_data, dict_lab = dict_lab, 
                                    output_fn = path_dump + 'Metrics_' + str_id_training[1:] + '.txt', multi_labels = multi_labels, 
                                    remove_labels = dict_args['remove_labels'], use_tf_dataset = dict_args['use_tf_dataset'])

            # TODO: test if quicker when data are in model.validation (faster gpu access...?)
            # metrics_f1 = Metrics_F1_ini()
            # model.validation_data = dev_data
            # print(model.validation_data[1][0])
            if tensorboard_callback:
                tensorboard_callback.append(metrics_f1)
            else:
                tensorboard_callback = [metrics_f1]

            # I generally never use that ; won't work with multimodal and tf_dataset
            if test_on_epoch_end:
                inputFile_test = test_on_epoch_end
                # type_model and type of file used for test/train 
                str_id_saving_files_test = '_{}_{}'.format(model_name,inputFile_test)

                test_tweets, test_y = loadTsvData(path_corpus + inputFile_test, dict_lab, SPARSE_CATEGORICAL = SPARSE_CATEGORICAL, 
                                                multi_labels = multi_labels, cumsum_label_vectors = cumsum_label_vectors)  # load the data

                test_x = create_or_load_input(path_dump_input, str_id_saving_files_test, test_tweets, tokenizer, max_seq_len = dict_args['max_seq_len'])

                if bool_multimodal:
                    a, b, c = test_x
                    # d = load_MM_input_from_disk_old(path_dump = path_dump, scores_or_feats=scores_or_feats, type_ds = 'one_target_dev')
                    # type_ds = 'dev' if 'dev' in inputFile_test else 'test'
                    # d = load_MM_input_from_disk_CrisisMMD(path_dump = path_dump, scores_or_feats=scores_or_feats, type_ds = type_ds)

                    # d, image_size = load_MM_array_or_path(path_corpus = path_corpus, name_tsv = inputFile_test, path_dump_input = path_dump_input, 
                # use_tf_dataset = dict_args['use_tf_dataset'], model_name_vision = dict_args['model_name_vision'])
                    d, image_size = load_MM_array_or_path(path_corpus = path_corpus, name_tsv = inputFile_test, path_dump_input = path_dump_input, 
                use_tf_dataset = dict_args['use_tf_dataset']*dict_args['path_in_tfds'], model_name_vision = dict_args['model_name_vision'])

                    test_x = (a, b, c, d)

                test_data = (test_x, test_y)  

                if dict_args['use_tf_dataset']:
                    # test_data = create_tf_dataset(test_x, test_y, image_size, batch_size=batch_size)
                    # test_data = create_tf_dataset(test_x, test_y, image_size, batch_size=batch_size, path_in_tfds=dict_args['path_in_tfds'])
                    test_data = create_tf_dataset_DA(test_x, test_y, image_size, batch_size=batch_size, path_in_tfds=dict_args['path_in_tfds'], data_augmentation_images=False)

                metrics_f1_test = Metrics_F1(validation_data = test_data, dict_lab = dict_lab, remove_labels = dict_args['remove_labels'], 
                                             use_tf_dataset = dict_args['use_tf_dataset'])
                tensorboard_callback.append(metrics_f1_test)

        # TODO : test this feature
        if 't5' in model_name:
            train_x = change_data_form_t5(train_x, is_multimodal=bool_multimodal)
            dev_data = (change_data_form_t5(dev_data[0], is_multimodal=bool_multimodal), dev_data[1])

        if verbose:
            print('Train, ', len(train_x), [k.shape for k in train_x])
            try:
                print(len(train_y), len(train_y[0]))
            except:
                print('sparse ', len(train_y))
            try:
                print('Dev, ', len(dev_data[0]), [k.shape for k in dev_data[0]])
                try:
                    print(len(dev_data[1]), len(dev_data[1][0]))
                except:
                    print('sparse ', len(dev_data[1]))
            except: 
                print('Dev, ', len(dataset_loader.dev_data[0]), [k.shape for k in dataset_loader.dev_data[0]])
                
        # TEST VAL <---------------------------------------------- A CHANGER 
        #train_x = (train_x[0], train_x[1], None)
        #a, b, _ = dev_x
        #dev_data = ((a, b, None), dev_data[1])
        # --------------------------------------------------------------------------------------------

        # var class_weight if data is unbalanced
        # steps_per_epoch to make shorter epochs, useful when dataset is huged like 2,9M tweets !! Let's put 10k batches(=320k tweets)?

        if dict_args['use_tf_dataset']:
            model.fit(train_ds, epochs=nb_epochs, batch_size=batch_size, validation_data=dev_ds, 
                      shuffle=True, callbacks=tensorboard_callback, class_weight=class_weight, steps_per_epoch=steps_per_epoch,
                     verbose = display_pg_bar)
        else:
            model.fit(train_x, train_y, epochs=nb_epochs, batch_size=batch_size, validation_data=dataset_loader.dev_data, 
                      shuffle=True, callbacks=tensorboard_callback, class_weight=class_weight, steps_per_epoch=steps_per_epoch,
                     verbose = display_pg_bar)
    else:
        if PRINT_F1:   # marche pas Metrics_F1 ici... On a pas de dev donc ca devrait merder... 
            metrics_f1 = Metrics_F1(dict_lab = dict_lab, output_fn = path_dump + 'Metrics_' + str_id_training[1:] + '.txt',
                                   remove_labels = dict_args['remove_labels'], use_tf_dataset = dict_args['use_tf_dataset'])
            # TODO: test if quicker when data are in model.validation (faster gpu access...?)
            # metrics_f1 = Metrics_F1_ini()
            # model.validation_data = dev_data
            # print(model.validation_data[1][0])
            if tensorboard_callback:
                tensorboard_callback.append(metrics_f1)
            else:
                tensorboard_callback = [metrics_f1]
        # var class_weight if data is unbalanced
        # steps_per_epoch to make shorter epochs, useful when dataset is huged like 2,9M tweets !! Let's put 10k batches(=320k tweets)?
        if dict_args['use_tf_dataset']:
            model.fit(train_ds, epochs=nb_epochs, batch_size=batch_size, validation_split=validation_split, 
                  shuffle=True, callbacks=tensorboard_callback, class_weight=class_weight, steps_per_epoch=steps_per_epoch,
                 verbose = display_pg_bar)            
        else:
            model.fit(train_x, train_y, epochs=nb_epochs, batch_size=batch_size, validation_split=validation_split, 
                  shuffle=True, callbacks=tensorboard_callback, class_weight=class_weight, steps_per_epoch=steps_per_epoch,
                 verbose = display_pg_bar)

    # Saving trained model
    print('Saving model to {}...'.format(trained_modelFilePath))
    model.save_weights(trained_modelFilePath)
    print('Model saved')

    # copy the tensorboard dumps into the JRCBox so I can visualize them on my computer... 
    if USE_TENSORBOARD:
        log_dir = path_corpus + "logs/fit/" + str_id_training[1:] +"/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        str_corpus = path_corpus.split('/')[-1]
        path_tensorboard_for_vis = '/eos/jeodpp/data/projects/EMM/transfer/Valentin/tensorboard/%s/logs/fit/'%str_corpus
        try:
            #TODO: verify that its working. Creating the folder if not existing... 
            path_eos_EMM = path_tensorboard_for_vis + str_id_training[1:] + '/'
            if not os.path.isdir(path_eos_EMM): 
                os.makedirs(path_eos_EMM, exist_ok = True)
                # os.mkdir(path_tensorboard_for_vis + str_id_training[1:] + '/')
            copytree(log_dir, path_eos_EMM + log_dir.split('/')[-1])
        except:
            print('Problem copying the tensorboard dumps to the JRCBox... (NextCloud/eos_EMM/) \nThe dumps are in {}'.format(log_dir))

def reg2class(val, value_min, value_max):
    """
    Transform a value from regression model going from value_min to value_max into a class for ordinal classification (granularity 1)
    """
    return np.min(np.max(round(val)-value_min, 0),value_max-value_min)

def reg2vec(val):
    """
    Use this one for code readability 
    """
    result = [k==round(val) for k in range(value_min, value_max + 1)]
    if val < value_min+0.5:
        result[0] = True
    elif val >= (value_max-0.5):
        result[-1] = True
    return result

def create_array_of_string_labels(y_classes, inv_dict_lab):
    """
    Return array composed of strings of the labels. 
    Works for multi-labels. 
    ['positive', 'negative', ...]
    ['disgust/sadness', 'joy', 'fear/anger', ...]
    """

    # if single label
    if type(y_classes[0]) == np.int64:
        y_strings = np.array([inv_dict_lab[k] for k in y_classes])
    # if single label
    elif len(y_classes[0]) == 1:
        y_strings = np.array([inv_dict_lab[k] for k in y_classes])
    else:
        y_strings = np.array([ '/'.join([inv_dict_lab[idx_k] for idx_k, k in enumerate(vec_labels) if k]) for vec_labels in y_classes])

    return y_strings 

def predict_only(modelFilePath, model, input_data_File, tokenizer, dict_lab, model_name='any_model', 
             path_corpus = './mk/out/', create_new_tsv=False, bool_multimodal=True, ordinal_data=False, dict_args={}):
    """
    Only predict and save the predictions 
    """    
    # type_model and type of file used for test/train 
    str_id_saving_files = '_{}_{}'.format(model_name,input_data_File)

    # TODO: take into account the dataset used to train the model
    # done --> Here is the info on the dataset used to train/dev, the model and the testing set
    str_id_training_and_testing = '_'.join([''] + os.path.splitext(modelFilePath.split('/')[-1])[0].split('_')[1:] + [input_data_File])

    path_dump = path_corpus + 'dump/'
    path_dump_pred = path_dump + 'predictions/'
    path_dump_input = path_dump + 'inputs/'

    # Creating the folders if not existing 
    for path_create in [path_dump, path_dump_pred, path_dump_input, path_dump + 'results/']:
        if not os.path.isdir(path_create):
            os.makedirs(path_create)

    # In order to use testBERT with the trained model we want
    # model.load_weights(modelFilePath, by_name=False, skip_mismatch=True)
    print("Loading '%s'"%modelFilePath)
    model.load_weights(modelFilePath, by_name=True)

    #TEST DATA
    # if pred file already existing, using it to add a new col
    if os.path.isfile(path_corpus + 'pred_' + input_data_File):
        df = read_csv_val(path_corpus + 'pred_' + input_data_File, encoding='utf-8')
        bool_add_col = True
    else:
        df = read_csv_val(path_corpus + input_data_File, encoding='utf-8')
        bool_add_col = False

    print('Predicting with the model {}, over {}...'.format(modelFilePath.split('/')[-1], 'pred_'*bool_add_col+input_data_File))

    # normally never happening
    test_tweets = df["tweet"].fillna("CVxTz").values # why 'CVxTz'?

    # change that, supposed to be in the class DatasetLoader()
    tokenizer.not_use_token_type_ids = not bool(dict_args['using_token_type_ids_roberta'])

    test_x = create_or_load_input(path_dump_input, str_id_saving_files, test_tweets, tokenizer, max_seq_len = dict_args['max_seq_len'])

    image_size = None
    if bool_multimodal:
        a, b, c = test_x
        scores_or_feats = 'scores'  if 'scores' in str_id_training_and_testing else 'features'
        # d = load_MM_input_from_disk_old(path_dump = path_dump, scores_or_feats=scores_or_feats, type_ds = 'one_target_test')
        # for when testing on the dev 
        type_ds = 'dev' if 'dev' in input_data_File else 'test'#; print('DEBUG', type_ds)
        # d = load_MM_input_from_disk_CrisisMMD(path_dump = path_dump, scores_or_feats=scores_or_feats, type_ds = type_ds)

        # d, image_size = load_MM_array_or_path(path_corpus = path_corpus, name_tsv = input_data_File, path_dump_input = path_dump_input, 
                # use_tf_dataset = dict_args['use_tf_dataset'], model_name_vision = dict_args['model_name_vision'])
        d, image_size = load_MM_array_or_path(path_corpus = path_corpus, name_tsv = input_data_File, path_dump_input = path_dump_input, 
                use_tf_dataset = dict_args['use_tf_dataset']*dict_args['path_in_tfds'], model_name_vision = dict_args['model_name_vision'])

        test_x = (a, b, c, d)

    if 't5' in model_name:
        test_x = change_data_form_t5(test_x, is_multimodal=bool_multimodal)

    if dict_args['use_tf_dataset']:
        # test_data = create_tf_dataset(test_x, None, image_size=image_size, batch_size = 32)
        # test_data = create_tf_dataset(test_x, None, image_size=image_size, batch_size = 32, path_in_tfds=dict_args['path_in_tfds'])
        test_data = create_tf_dataset_DA(test_x, [], image_size=image_size, batch_size = 32, path_in_tfds=dict_args['path_in_tfds'], data_augmentation_images=False)
        test_x = test_data

    predictions = model.predict(test_x, batch_size = 32, verbose = dict_args['display_pg_bar'])

    # if multi_labels for the test setting 
    if dict_args['multi_labels_test']:

        if dict_args['threshold_multi_labels'] == 0:
            nb_lab = 1
        else:
            # Here, take maximum nb_lab labels iff they are above the threshold of likelikood
            nb_lab = 2

        y_classes = create_y_multilabels(predictions, dict_args['threshold_multi_labels'], nb_lab=nb_lab)

    else:
        y_classes = predictions.argmax(axis=-1)

    # print(y_classes)
    if dict_args['group_labels_test']:
        # path_corpus + dict_args['group_labels_test'] 
        if 'GoEmotions' in dict_args['name_corpora']:
            #with open('/home/emmproc/data/Valentin/GoEmotions/ekman_mapping.json', 'rb') as f:
            with open(PATH_DATA + 'GoEmotions2/ekman_mapping.json', 'rb') as f:
                js = json.loads(f.read())
            js['neutral'] = ['neutral']
        elif 'Debating_Europe' in dict_args['name_corpora']:
            #with open('/home/emmproc/data/Valentin/Debating_Europe/answer_mapping.json', 'rb') as f:
            with open(PATH_DATA + 'Debating_Europe/answer_mapping.json', 'rb') as f:
                js = json.loads(f.read())

        y_classes = group_labels(y_classes, js, dict_lab)

        dict_lab = {grouped_lab : idx_sublab for idx_sublab, (grouped_lab, list_sublabs) in enumerate(js.items())}

    inv_dict_lab = {v:k for k,v in dict_lab.items()}
    # y_classes_dump = np.array([inv_dict_lab[k] for k in y_classes]) 
    # print(y_classes)

    y_classes_dump = create_array_of_string_labels(y_classes, inv_dict_lab)

    pred_fn = path_dump_pred + 'pred' + '_grouped'*bool(dict_args['group_labels_test'])  + '_multilab'*bool(dict_args['multi_labels_test']) + str_id_training_and_testing + ".npy"
    np.save(pred_fn, y_classes_dump)

    proba_fn = path_dump_pred + 'proba' + '_grouped'*bool(dict_args['group_labels_test'])  + '_multilab'*bool(dict_args['multi_labels_test']) + str_id_training_and_testing + ".npy"
    np.save(proba_fn, predictions)

    print('prediction saved at: {}'.format(proba_fn))

    if create_new_tsv: # create pred_%s.tsv%fname or add it a new column
        if dict_args['training_type'] == 'regression':
            output_values = predictions
        else:
            output_values = y_classes_dump
        df['pred' + '_grouped'*bool(dict_args['group_labels_test']) +'_multilab'*bool(dict_args['multi_labels_test']) + str_id_training_and_testing] = output_values
        #df['proba' + '_grouped'*bool(dict_args['group_labels_test']) +'_multilab'*bool(dict_args['multi_labels_test']) + str_id_training_and_testing] = predictions # marche pas
        print('Creating tsv file with the predictions : %s'%(path_corpus + 'pred_' + input_data_File))
        to_csv_val(df, path_corpus + 'pred_' + input_data_File)

    return 0

def group_labels(y_gold_classes, js, dict_lab):
    """
    Transform vector multi-tags to another smaller vector multi-tags using json mapping 
    Works for sparse categorical or full one-hot vectors 
    """

    dict_transition_idx = {}
    for idx_grouped_lab, (grouped_lab, list_sublabs) in enumerate(js.items()):
        for sublab in list_sublabs:
            # useful when sometimes more labels are available in the json file, than the one used in reality
            if sublab in dict_lab.keys():
                dict_transition_idx[dict_lab[sublab]] = idx_grouped_lab

    print('Grouping labels...')

    # if not sparse categorical 
    if len(y_gold_classes.shape) > 1:

        nb_labels_grouped = len(js)
        inv_dict_lab = {v:k for k,v in dict_lab.items()}

        y_gold_classes_grouped = []
        for y_samp in y_gold_classes:
            y_samp_grouped = [0]*nb_labels_grouped
            for str_lab in [inv_dict_lab[idx_k] for idx_k, k in enumerate(y_samp) if k]:
                for idx_grouped_lab, (grouped_lab, list_sublabs) in enumerate(js.items()):
                    if str_lab in list_sublabs:
                        # y_samp_grouped[idx_sublab] += 1
                        y_samp_grouped[idx_grouped_lab] = 1

            y_gold_classes_grouped.append(y_samp_grouped)
        y_gold_classes = np.array(y_gold_classes_grouped)
    else:

        for idx in range(len(y_gold_classes)):
            y_gold_classes[idx] = dict_transition_idx[y_gold_classes[idx]]

    return y_gold_classes

def predictBERT(modelFilePath, model, input_data_File, tokenizer, dict_lab, model_name='any_model', 
             path_corpus = './mk/out/', bool_multimodal=True, dict_args={}):

    # type_model and type of file used for test/train 
    str_id_saving_files = '_{}_{}'.format(model_name,input_data_File)

    # TODO: take into account the dataset used to train the model
    # done --> Here is the info on the dataset used to train/dev, the model and the testing set
    str_id_training_and_testing = '_'.join([''] + os.path.splitext(modelFilePath.split('/')[-1])[0].split('_')[1:] + [input_data_File])

    path_dump = path_corpus + 'dump/'
    path_dump_pred = path_dump + 'predictions/'
    path_dump_input = path_dump + 'inputs/'

    # Creating the folders if not existing 
    for path_create in [path_dump, path_dump_pred, path_dump_input, path_dump + 'results/']:
        if not os.path.isdir(path_create):
            os.makedirs(path_create)

    # In order to use testBERT with the trained model we want
    model.load_weights(modelFilePath)
    # model.load_weights(modelFilePath, by_name=False, skip_mismatch=True)

    #TEST DATA
    test_tweets, test_y = loadTsvData(path_corpus + input_data_File, dict_lab, multi_labels = dict_args['multi_labels_test'], cumsum_label_vectors = cumsum_label_vectors)       # load the data

    # change that, supposed to be in the class DatasetLoader()
    tokenizer.not_use_token_type_ids = not bool(dict_args['using_token_type_ids_roberta'])

    test_x = create_or_load_input(path_dump_input, str_id_saving_files, test_tweets, tokenizer, max_seq_len = dict_args['max_seq_len'])

    image_size=None
    if bool_multimodal:
        a, b, c = test_x
        scores_or_feats = 'scores'  if 'scores' in str_id_training_and_testing else 'features'
        # d = load_MM_input_from_disk_old(path_dump = path_dump, scores_or_feats=scores_or_feats, type_ds = 'one_target_test')
        # for when testing on the dev 
        type_ds = 'dev' if 'dev' in input_data_File else 'test'#; print('DEBUG', type_ds)
        # d = load_MM_input_from_disk_CrisisMMD(path_dump = path_dump, scores_or_feats=scores_or_feats, type_ds = type_ds)
        # d, image_size = load_MM_array_or_path(path_corpus = path_corpus, name_tsv = input_data_File, path_dump_input = path_dump_input, 
                # use_tf_dataset = dict_args['use_tf_dataset'], model_name_vision = dict_args['model_name_vision'])
        d, image_size = load_MM_array_or_path(path_corpus = path_corpus, name_tsv = input_data_File, path_dump_input = path_dump_input, 
                use_tf_dataset = dict_args['use_tf_dataset']*dict_args['path_in_tfds'], model_name_vision = dict_args['model_name_vision'])

        test_x = (a, b, c, d)

    if 't5' in model_name:
        test_x = change_data_form_t5(test_x, is_multimodal=bool_multimodal)

    if dict_args['use_tf_dataset']:
        # test_x = create_tf_dataset(test_x, None, image_size=image_size, batch_size = 32)
        # test_x = create_tf_dataset(test_x, [], image_size=image_size, batch_size = 32, path_in_tfds=dict_args['path_in_tfds'])
        test_x = create_tf_dataset_DA(test_x, [], image_size=image_size, batch_size = 32, path_in_tfds=dict_args['path_in_tfds'], data_augmentation_images=False)    

    # Computing metrics 
    batch_size_testing = 128
    predictions = model.predict(test_x, batch_size = batch_size_testing, verbose = dict_args['display_pg_bar'])

    return predictions, test_y

def testBERT(modelFilePath, model, input_data_File, tokenizer, dict_lab, model_name='any_model', 
             path_corpus = './mk/out/', bool_multimodal=True, ordinal_data=False, dict_args={}, verbose=False):
    """
    Load the transformer stored in `modelFilePath` 
    Load the tweets in the `inputFile` tsv
    CAREFUL `path_dump` hard-coded as `path_dump = path_corpus + 'dump/'` 
    Saving all the predicted values into path_dump + 'all_results.csv'
    """

    print('Testing with the model {}, over {}...'.format(modelFilePath.split('/')[-1], input_data_File))

    predictions, test_y = predictBERT(modelFilePath, model, input_data_File, tokenizer, dict_lab, model_name, 
             path_corpus, bool_multimodal, dict_args)

    # TODO: take into account the dataset used to train the model
    # done --> Here is the info on the dataset used to train/dev, the model and the testing set
    str_id_training_and_testing = '_'.join([''] + os.path.splitext(modelFilePath.split('/')[-1])[0].split('_')[1:] + [input_data_File])

    path_dump = path_corpus + 'dump/'
    path_dump_pred = path_dump + 'predictions/'

    # See if regression or classification
    task_type = 'classification' if len(predictions[0]) > 1 else 'regression'

    # TODO: what if regression ? still a matrix of size (n_samples, 1) so need to squeeze or flatten instead 
    # if classification pure
    if task_type == 'classification':
        y_classes = predictions.argmax(axis=-1)
        y_reg = None # On calcule quand meme K et r avec y_classes

        if dict_args['use_proba_treshold']:
            # TODO: me semble louche ce predictions.argmax > thresh ; devrait plutot etre predictions > thresh ???
            bool_sample_over_threshold = predictions.argmax(axis=-1) > dict_args['use_proba_treshold']
            y_classes = y_classes[bool_sample_over_threshold]
            test_y = test_y[bool_sample_over_threshold]

    else:# trained as regression; Can be ordinal classification or pure regression
        y_reg = predictions.flatten()

        # if regression over integers its an ordinal classification
        if test_y.dtype == np.int32:
            task_type = 'ordinal_classification'
            value_min = test_y.min()
            value_max = test_y.max()
            y_classes = np.array([reg2vec(val, value_min, value_max) for val in y_reg]).argmax(-1)
            # for the offset, if the regression starts at -3 for example
            y_gold_classes = test_y - value_min
        else: # if pure regression 
            # y_gold_classes is not classes but regression values
            y_gold_classes = test_y
            # y_classes is the regression values
            y_classes = None

    # if NOT sparse_categorical: never happens now, except when using multi-labels ?  
    if len(test_y.shape) > 1:
        y_gold_classes = test_y.argmax(axis=-1)
    else:
        y_gold_classes = test_y

    # the label is dependant of the first part of the input, for exemple "<s> explosion </s> 10 people died in an explosion </s>" will be "explosion -- dead people"
    if 'macro_macro' in dict_args.keys():# macro_macro:
        if dict_args['macro_macro']:
            # load the tweets with metadata in the first subsentence 
            test_tweets, _ = loadTsvData(path_corpus + dict_args['macro_macro'], dict_lab, multi_labels = multi_labels, 
                                         cumsum_label_vectors = cumsum_label_vectors) 

            len_labels = len(dict_lab)
            list_first_subphrases = [k.split(' </s> ')[0] for k in test_tweets]

            inv_dict_lab = {v:k for k, v in dict_lab.items()}
            # each pair of event -- label to a value 
            list_all_possible_labels = set()
            for subph, y_gt, y_pred in zip(list_first_subphrases, y_gold_classes, y_classes):
                list_all_possible_labels.update([subph + ' -- ' + inv_dict_lab[y_gt]])
                # list_all_possible_labels.update([subph + ' -- ' + inv_dict_lab[y_pred]]) # on ne prend pas les FP des classes qui n'apparaissent pas

            # creation the new dict_lab
            new_dict_lab = {lab : idx_lab for idx_lab, lab in enumerate(np.sort(list(list_all_possible_labels)))} 

            y_gold_classes_macro = []
            y_classes_macro = []
            # for every example in the test set, we change for the new label
            for idx_sample, subph in enumerate(list_first_subphrases):
                # si label 
                if subph+' -- ' + inv_dict_lab[y_classes[idx_sample]] in new_dict_lab.keys():
                    #y_gold_classes[idx_sample] = new_dict_lab[subph + ' -- ' + inv_dict_lab[y_gold_classes[idx_sample]]]
                    #y_classes[idx_sample] = new_dict_lab[subph + ' -- ' + inv_dict_lab[y_classes[idx_sample]]]
                    y_gold_classes_macro.append(new_dict_lab[subph + ' -- ' + inv_dict_lab[y_gold_classes[idx_sample]]])
                    y_classes_macro.append(new_dict_lab[subph + ' -- ' + inv_dict_lab[y_classes[idx_sample]]])

            dict_lab = new_dict_lab.copy()
            y_gold_classes = y_gold_classes_macro
            y_classes = y_classes_macro
            # import pickle as pkl
            # pkl.dump(dict_lab, open('dict_lab2.pkl', 'wb'))
            # pkl.dump(y_gold_classes_macro, open('y_gold_classes_macro.pkl', 'wb'))
            # pkl.dump(y_classes_macro, open('y_classes_macro.pkl', 'wb'))
            # print(dict_lab, y_gold_classes_macro, y_classes_macro)

    # if multi_labels for the test setting (you could train on multi and test over signle)
    if dict_args['multi_labels_test']:
        y_gold_classes = test_y

        # 
        if dict_args['threshold_multi_labels'] == 0:
            nb_lab = 1
        else:
            # Here, take maximum nb_lab labels iff they are above the threshold of likelikood
            nb_lab = 3

        y_classes = create_y_multilabels(predictions, dict_args['threshold_multi_labels'], nb_lab=nb_lab)

    if dict_args['group_labels_test']:
        # path_corpus + dict_args['group_labels_test']
        if 'GoEmotions' in dict_args['name_corpora']:
            #with open('/home/emmproc/data/Valentin/GoEmotions/ekman_mapping.json', 'rb') as f:
            with open(PATH_DATA + 'GoEmotions2/ekman_mapping.json', 'rb') as f:
                js = json.loads(f.read())
            js['neutral'] = ['neutral']
        elif 'Debating_Europe' in dict_args['name_corpora']:
            #with open('/home/emmproc/data/Valentin/Debating_Europe/answer_mapping.json', 'rb') as f:
            with open(PATH_DATA + '/Debating_Europe/answer_mapping.json', 'rb') as f:
                js = json.loads(f.read())
        elif 'EIOS' in dict_args['name_corpora']:
            #with open('/home/emmproc/data/Valentin/EIOS/dashboard_mapping.json', 'rb') as f:
            with open(PATH_DATA + 'EIOS/dashboard_mapping.json', 'rb') as f:
                js = json.loads(f.read())
        new_js = {}
        for key, value in js.items():
            new_js[key.lower()] = [k.lower() for k in value]
        js = new_js

        # y_gold_classes = test_y 

        # if multi-labels + multi-class also on the grouped-labels
        # y_classes = predictions > 0

        # If multi-class but SINGLE-label task on the test 
        # Cas ou on arrive sur une tache de classification (test est pas multi_labels)
        # y_classes = np.zeros(y_gold_classes.shape)
        # y_classes[:,np.argmax(predictions, axis=-1)] = 1 
        # for idx_samp, best_lab in enumerate(np.argmax(predictions, axis=-1)):
        #     y_classes[idx_samp, best_lab] = 1

        # group the labels and create the new dict_lab for the grouped labels
        y_gold_classes = group_labels(y_gold_classes, js, dict_lab)
        y_classes = group_labels(y_classes, js, dict_lab)
        dict_lab = {grouped_lab : idx_sublab for idx_sublab, (grouped_lab, list_sublabs) in enumerate(js.items())}
        # if we want to throw away the classes that are not in the test set
        # dict_lab = {grouped_lab : idx_sublab for idx_sublab, (grouped_lab, list_sublabs) in enumerate(js.items()) if grouped_lab in np.unique(y_gold_classes)}

    str_tot, dict_arg_updated = calculate_and_print_metrics(y_gold_classes, y_classes, y_reg, dict_lab, path_dump_pred, str_id_training_and_testing,
                                         task_type, ordinal_data, dict_args)

    # Saving all results in a csv for pandas analysis
    if not os.path.isfile(path_dump+'all_results.csv'):
        df_results = pd.DataFrame(columns = dict_arg_updated.keys())
    else:
        df_results = pd.read_csv(path_dump+'all_results.csv', delimiter='\t')
    idx_newline = len(df_results)
    for col, val in dict_arg_updated.items():
        if type(val) == list: val = str(val)
        df_results.loc[idx_newline, col] = val

    df_results.to_csv(path_dump+'all_results.csv', sep='\t', index=None)

    return str_tot

### Training and Testing 
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument("-c", "--name_corpora", help="The name of the folder containing the corpora", type=str, 
                        default="tweets_alex", choices=['tweets_alex', 'IJCAI2019_data', 'ACL_events', 'Crisis_Benchmark', 'Crisis_MMD', 'French_Crisis', 'Mikhail_covid', 'GoEmotions', 'GoEmotions2', 'WASSA23', 'Debating_Europe', 'Debating_Europe_new', 'Values', 'Daily_Dialog', 'Fakeddit', 'memotion_dataset_7k', 'HumAID', 'Eurotweets', 'ValueNet', 'Empathy_WASSA2022', 'EIOS', 'Multilingual_Crisis', 'SLR', 'XAI_NLLF'])
    parser.add_argument("-t", "--name_corpora_test", help="The name of the folder containing the corpora", type=str, 
                        default="")
    # parser.add_argument("--path_corpora", help="The path of the folders containing all the corpora", type=str, default="/home/emmproc/data/Valentin/")
    parser.add_argument("--path_corpora", help="The path of the folders containing all the corpora", type=str, 
                        default=PATH_DATA)
    parser.add_argument("--other_device", help="If using the script outside jhub", default=False, action='store_true')

    # model architecture
    parser.add_argument("-m", "--model_name", help="The name of the model name", type=str, default="bert-tfhub-fine-tuned", 
                        choices=['bert-tfhub-fine-tuned', 'bert-tfhub', 'bert-base-multilingual-uncased', 'bert-base-multilingual-cased', 'xlm-roberta-base', 
                        'xlm-roberta-large', 'roberta-base', 'roberta-large', 't5-small', 't5-base', 't5-large', 't5-3b', 't5-11b', 'electra-small-discriminator', 
                        'electra-base-discriminator', 'electra-large-discriminator', 'dccuchile/bert-base-spanish-wwm-uncased', 'bert-base-german-cased', 
                                 'bert-base-italian-uncased', 'camembert-base', 'alberto', 'bert-base-uncased', 'tf-flaubert-base-cased', 'lxmert', 
                                 'cardiffnlp-emotion', 'cardiffnlp-sentiment', 'cardiffnlp-pretrained', 'mrm8488/RuPERTa-base', 'uklfr/gottbert-base', 
                                 'tartuNLP/EstBERT', 'skimai/spanberta-base-cased', 'microsoft/mdeberta-v3-base', 'microsoft/deberta-v3-base', 'microsoft/deberta-v3-large', 'microsoft/deberta-v2-xxlarge', 'scibert_scivocab_uncased', 'scibert_scivocab_cased', 'facebook/galactica-125m', 'facebook/galactica-1.3b', 'facebook/galactica-6.7b', '/home/barriva/data/SLR/mlm-16k-roberta', '/home/barriva/data/SLR/mlm-16k-scibv2', '/home/barriva/data/SLR/mlm-16k-scibv2-10epochs'])
    parser.add_argument("--pre_training_text", help="path of the text model if already pre-trained BY OURSELF. All the models are already pre-trained anyway", type=str, default="")
    parser.add_argument("--output_network", help="Type of the output network", type=int, default=7)
    parser.add_argument("--training_type", help="Regression or classification, calculate automatically if it's an ordinal classification when using regression", type=str, 
                        default="", choices = ['classification', 'regression', ''])
    parser.add_argument("--multi_labels", help="If multiple tags for one example", default=False, action='store_true')
    parser.add_argument("--threshold_multi_labels", help="If multiple tags for one example, used to calculate Metrics_F1 during training", type=float, default=0)
    parser.add_argument("--multi_labels_test", help="If multiple tags for one example on the test", default=False, action='store_true')
    parser.add_argument("--using_token_type_ids_roberta", help="Force not using token type ids when available", type=str2bool, default=True)
    parser.add_argument("--max_seq_len", help="length of the input sequences of the transformer", type=int, default=128)

    # debugging
    parser.add_argument("--force_token_creation", help="Force the creation of the token instead of using the saved ones", default=False, action='store_true')
    
    # multimodal and fusion
    parser.add_argument("--multimodal", help="If multimodal setting", default=False, action='store_true') # not used anymore
    parser.add_argument("--model_name_vision", help="The name of the vision model name. Type of EfficientNet, default: efficient2", type=str, default="efficient2", 
                        choices=['efficient0', 'efficient1', 'efficient2', 'efficient3', 'efficient5', 'efficient4'])
    parser.add_argument("--pre_training_vision", help="path of the image model if already pre-trained BY OURSELF. All the models are already pre-trained anyway", type=str, default="")
    parser.add_argument("--fusion", help="Fusion on the scores of the unimodal classifiers or the features", type=str, 
                        default="features", choices=['features', 'scores'])
    parser.add_argument("--first_block_to_uf", help="Type of the output network", type=int, default=7, choices=range(9))

    # datasets
    parser.add_argument("--train_set", help="training sets, you can put several and they will be concatenated", type=str, default=["one-target-train.tsv"], nargs='+')
    parser.add_argument("--dev_set", help="dev set", type=str, default="one-target-dev.tsv")
    parser.add_argument("--test_set", help="testing set, you can put several and they will be tested one-by-one", type=str, default="tweeti_all_lang", nargs='+')
    parser.add_argument("--else_class", help="If there is some classes to be aggregated in a classe else", type=str2bool, default=True)
    parser.add_argument("--using_dev_set", help="Using a dev set", type=str2bool, default=True)
    parser.add_argument("--validation_split", help="Percentage of the train set to use as dev set, when not using dev set", 
                        type=float, default=.1)
    parser.add_argument("--no_test", help="Dont run the tests", default=False, action='store_true')
    parser.add_argument("--use_tf_dataset", help="Use a tf.data.Dataset class", default=False, action='store_true')
    parser.add_argument("--path_in_tfds", help="When using tfds, use the path", type=str2bool, default=True)

    # labels 
    parser.add_argument("--labels_from_first_train", help="take the labels present in the first train as the default labels", default=False, action='store_true')
    parser.add_argument("--sparse_categorical", help="Whether to use sparse categories or not", type=str2bool, 
                        default=True)
    parser.add_argument("--ordinal_data", help="If the data is ordinal, compute Cohen's Kappa and Pearson's r", default=False, action='store_true')
    parser.add_argument("--cumsum_label_vectors", help="If the data is ordinal, create label vectors as cumsum, from (Cheng et al., 2007)", default=False, action='store_true')

    # training phase
    parser.add_argument("--steps_per_epoch", help="If maximum number of steps per epoch", type=int, default=None)
    parser.add_argument("--learning_rate", help="Learning Rate", type=float, default=1e-6)
    parser.add_argument("--class_weight", help="If unbalanced dataset", type=str2bool, default=True)
    parser.add_argument("--nb_epochs", help="nb of epochs", type=int, default=50)
    parser.add_argument("--batch_size_training", help="batch size during training, cannot be more than 32 right now", type=int, default=32)
    parser.add_argument("--min_delta_es", help="Min delta for early stopping", type=float, default=0.005)
    parser.add_argument("--patience", help="Patience for early Stopping, if 0 no early stopping", type=int, default=5)
    parser.add_argument("--monitor_value_es", help="What to monitor for early Stopping (loss or accuracy). RMSE by default for regression", type=str, default='val_loss', choices=['val_loss', 'val_acc', 'val_rmse'])
    parser.add_argument("--continue_training", help="Continue the training of the model", default=False, action='store_true')

    # monitoring
    parser.add_argument("--tensorboard", help="tensorboard dumps for monitoring", type=str2bool, default=False)
    parser.add_argument("--update_freq_tboard", help="Frequency to drop values for TensorBoard", type=int, default=100)
    parser.add_argument("--print_f1", help="Print F1 on the dev set at the end of each epoch", type=str2bool, default=True)
    parser.add_argument("--use_model_checkpoint", help="Save the best model regarding the loss on the dev set", type=str2bool, default=True)
    parser.add_argument("--monitor_value", help="What to monitor to save the best model (loss or accuracy)). RMSE by default for regression", type=str, default='val_acc', choices=['val_loss', 'val_acc'])
    # parser.add_argument("--use_early_stopping", help="Stop the training if no improving", type=str2bool, default=True)

    # other
    parser.add_argument("--test_model_only", help="Test the model without training it", default=False, action='store_true')
    parser.add_argument("--for_test", help="Useful to test my code, put a value and it will add that to the string used as ID to save stuff", type=str, default='')
    parser.add_argument("--verbose", help="verbose", default=False, action='store_true')
    parser.add_argument("--test_on_epoch_end", help="Test the metrics at the end of each epoch", default=False, action='store_true')
    parser.add_argument("--display_pg_bar", help="Display the progress bar, useful set to when using nohup", type=str2bool, default=True)
    parser.add_argument("--save_best_model_only", help="Display the progress bar, useful set to when using nohup", type=str2bool, default=True)
    parser.add_argument("--save_new_model_file", help="Create a new file to save the model in order not to erase the last one", type=str2bool, default=True)

    # testing 
    parser.add_argument("--macro_macro", help="Separating the labels without a same type of another attribute", default='', type=str)
    parser.add_argument("--predict_only", help="Predict only instead of testing", default=False, action='store_true')
    parser.add_argument("--create_new_tsv", help="Create a new tsv with the predictions on a new column", default=False, action='store_true')
    parser.add_argument("--group_labels_test", help="Group labels together", type=str, default='')
    parser.add_argument("--model_to_load", help="If you want to load a model from /home/emmproc/data/Valentin/models/", type=str, default='')
    parser.add_argument("--remove_labels", help="Remove some labels from the score functions", type=str, default=[], nargs='+')

    # new ones 
    parser.add_argument("--bool_weight_language_on_dev", help="Sample the dev set regarding the lang", default=False, action='store_true')
    # parser.add_argument("--use_token_type_ids_roberta", help="use token_type_ids with roberta models", type=str2bool, default=True) # on a deja using_token_type_ids_roberta
    parser.add_argument("--use_proba_treshold", help="", type=float, default=.0)
    parser.add_argument("--calculate_sentiment_bias", help="Test the metrics at the end of each epoch", default=False, action='store_true')

    args = parser.parse_args()

    if args.verbose:
        print(args)

    # used if we want to test model only
    test_model_only = args.test_model_only

    #################################### Initialization variables ####################################

    SPARSE_CATEGORICAL = args.sparse_categorical

    str_corpus = args.name_corpora + '/'
    #str_corpus = 'IJCAI2019_data/'

    # Historically used when I had not the JEODPP account...  now always False 
    if args.other_device:
        assert not IS_JHUB, "Problem !! other_device and IS_JHUB at the same time"
        path_corpus, path_dump = return_str_paths(IS_COLAB, IS_LOCAL, IS_JHUB, str_corpus)
        proxies = None
    else:
        path_corpus = args.path_corpora + str_corpus
        path_dump = path_corpus + 'dump/'
        if args.name_corpora_test:
            path_corpus_test = args.path_corpora + args.name_corpora_test + '/'
        else:
            path_corpus_test = path_corpus
        # No need to specify proxies anymore on JEODPP
        # proxies = {'https': 'https://proxy.cidsn.jrc.it:8888', 'http': 'http://proxy.cidsn.jrc.it:8888'}
        proxies = None

    # Creation of the directory for the dumps
    if not os.path.exists(path_dump):
        os.makedirs(path_dump)
        print('Creating the directory {}'.format(path_dump))

    ############ Model architecture ############

    model_name_ini = args.model_name
    # deal with stuff like mrm8488/RuPERTa-base 
    model_name = model_name_ini.split('/')[-1]

    OUTPUT_NETWORK = args.output_network
    fine_tune_hub = True if model_name == 'bert-tfhub-fine-tuned' else False
    training_type = args.training_type
    multi_labels = args.multi_labels

    ############ Multimodal part ############

    bool_already_fine_tuned_on_tweets = True if args.pre_training_text else False
    bool_already_fine_tuned_on_tweets = bool_already_fine_tuned_on_tweets if args.pre_training_vision else False

    bool_multimodal = True if OUTPUT_NETWORK > 10 else False
    scores_or_feats = args.fusion
    # name_img_file =  'input_img_{}_one_target_test.npy'.format(score_or_feat)

    # TODO: change that, should use the dataset to know what size I should use
    input_img_dim = 0

    # other verification not really needed
    # if bool_multimodal:
        # OUTPUT_NETWORK = 11 if scores_or_feats == 'scores' else 12
        # if 'Crisis_MMD' in path_corpus:
            # OUTPUT_NETWORK = 13

    assert ((bool_multimodal and OUTPUT_NETWORK > 10) or 
            ((not bool_multimodal) and OUTPUT_NETWORK <= 10)), "Multimodal setting and network type are incompatible"

    ############ Datasets ############
    input_data_File_train = args.train_set
    first_file_data_train = input_data_File_train[0]

    ##### FOR Leave One Event Type Out #####
    if 'all_but' in first_file_data_train:
        if 'event-type2' in first_file_data_train:
            fn = 'crisis_consolidated_humanitarian_filtered_lang_en_event-type2_{}_{}.tsv_val'
        else:
            fn = 'crisis_consolidated_humanitarian_filtered_lang_en_{}_{}.tsv_val'

        input_data_File_train = []
        for event in ['None', 'earthquake', 'bombing', 'flood', 'hurricane', 'explosion',
                      'landslide', 'fire', 'crash', 'collapse', 'hazard', 'volcano',
                      'disease', 'shooting']:

            if event not in first_file_data_train:
                  input_data_File_train.append(fn.format(event, 'all')) 

    first_file_data_train = input_data_File_train[0]
    ########################################

    bool_with_dev = args.using_dev_set
    input_data_File_dev = args.dev_set if bool_with_dev else 'Nodev'
    bool_weight_language_on_dev = args.bool_weight_language_on_dev

    # if regression, output of the network is size 1 and the data ordinal
    if training_type == 'regression':
        nb_classes = 1
        if args.verbose: print('\n\nREGRESSION MODEL\n\n')
        dict_lab = {'regression':0}
        ordinal_data = True
        if 'Valence-oc' in first_file_data_train:
            int2str = {-3:'very negative', -2:'negative', -1:'slightly negative',
                      0:'neutral', 1:'slightly positive', 2:'positive', 3:'very positive'}
            dict_lab = {v:k for k,v in int2str.items()}  

        # osef ?
        cumsum_label_vectors = args.cumsum_label_vectors

    else: # can have ordinal data when using classification mode
        # path_corpus_test since can be predict_only 
        path_corpus_dict_lab = path_corpus_test if os.path.isfile(path_corpus_test+first_file_data_train) else path_corpus
        dict_lab = return_dict_labels_for_training(not args.labels_from_first_train, path_corpus_dict_lab+first_file_data_train, str_corpus, args.else_class, multi_labels)

        nb_classes = len(dict_lab)
        if args.verbose: 
            print('not '*(not multi_labels) + 'multi-label')
            print('dict_lab:\n', dict_lab)

        ordinal_data = args.ordinal_data
        cumsum_label_vectors = args.cumsum_label_vectors
        # if cumsum label vectors 
        if ordinal_data and args.cumsum_label_vectors:
            SPARSE_CATEGORICAL=False

    # if testing at the end of every epoch ALSO on the test (use for debugging)
    if args.test_on_epoch_end:
        test_on_epoch_end = args.test_set[0]
    else:
        test_on_epoch_end = None

    ############ Training process ############
    # put a value to stop the training of each epoch at a certain number of batchs 

    nb_epochs = args.nb_epochs
    steps_per_epoch=args.steps_per_epoch
    str_steps_per_epoch = 'all-epoch' if not steps_per_epoch else steps_per_epoch
    learning_rate=args.learning_rate # 3e-5 at the beginning, but wasn't really working 
    # useful if dataset is unbalanced
    bool_class_weight = args.class_weight
    if training_type == 'regression': 
        print('Not using class_weight since regression...')
        bool_class_weight = False
    validation_split = args.validation_split
    batch_size_training = args.batch_size_training
    continue_training = args.continue_training

    ############ Monitoring the training ############

    USE_TENSORBOARD = args.tensorboard
    update_freq_tboard = args.update_freq_tboard
    PRINT_F1 = args.print_f1
    DROP_MODEL = args.use_model_checkpoint
    monitor_value = args.monitor_value
    patience = args.patience
    monitor_value_es = args.monitor_value_es
    min_delta = args.min_delta_es

    if training_type == 'regression': 
        if monitor_value != 'val_loss':
            monitor_value = 'val_rmse'
        if monitor_value_es != 'val_loss':
            monitor_value_es = 'val_rmse'
        PRINT_F1 = False

    ############ Strings used to indentify the training ############ 
    # to handle a training with several datasets
    input_data_File_train_in_id = '-AND-'.join(args.train_set)

    list_training_id = [model_name, input_data_File_train_in_id, input_data_File_dev, str_steps_per_epoch, '{:.0e}'.format(learning_rate)]

    if training_type: # if condition for backward compatibility 
        list_training_id = [training_type] + list_training_id

    if bool_class_weight: list_training_id += ['CW'] 

    if bool_weight_language_on_dev: list_training_id += ['LangW-dev'] 

    list_training_id_before_PT = list_training_id.copy()

    # This is useful to use the list_training_id as it is now and create str_training_id_already_trained to finetune a multimodal model pre-trained in a monomodal way 
    if args.pre_training_text or args.pre_training_vision: 
        str_to_add = 'already-ft-on'
        if args.pre_training_text:
            # if several pre-training possible ; la on prend juste le meme reseau texte en output 7 
            str_to_add += '-text-%s'%args.pre_training_text
            # str_to_add += '-text'
            # This is useful when you want to finetune a multimodal model where the text was beforehand pre-trained in a monomodal way. 
            list_training_id_already_trained = list_training_id + ['output-type{}'.format(7)]
            str_training_id_already_trained = '_'.join(list_training_id_already_trained)
            str_training_id_already_trained = reduce_size_str_training_id(str_training_id_already_trained) 
            already_trained_modelFilePath_text = path_dump + 'fine-tuned_{}.hdf5'.format(str_training_id_already_trained) 
            # to take the best checkpoint model 
            already_trained_modelFilePath_text = already_trained_modelFilePath_text[:-5]+'-best-model-checkpoint-{}.hdf5'.format(monitor_value)
            # DEBUG
            # already_trained_modelFilePath_text = already_trained_modelFilePath_text[:-5]+'.hdf5'

        if args.pre_training_vision:
            # This is useful when you want to finetune a multimodal model where the vision was beforehand pre-trained in a monomodal way. 
            # already_trained_modelFilePath_vision = path_dump + 'vision/' + 'best-{}-checkpoint-val_accuracy-unfreeze7.hdf5'.format(args.model_name_vision)
            already_trained_modelFilePath_vision = path_dump + 'vision/' + 'best-{}.hdf5'.format(args.pre_training_vision) # emotion-efficient3-batch32-checkpoint-val_accuracy-unfreeze7-2e-04
            str_to_add += '-vision-%s'%args.pre_training_vision

        list_training_id += [str_to_add]

    # TODO : a changer on utilise plus socre or feat, on utilise le reseau et on freeze ou pas
    if bool_multimodal: 
        list_training_id += ['unfreeze%d'%args.first_block_to_uf]

    list_training_id += ['output-type{}'.format(OUTPUT_NETWORK)]

    # useful for tests 
    if args.for_test: list_training_id += ['TEST-{}'.format(args.for_test)] 

    # useful to continue a training, but actually not rigorous, you should start again the training
    if continue_training: list_training_id += ['continue-training'] 

    # multi_labels ou pas
    if multi_labels: list_training_id += ['multi-labels'] 

    if args.remove_labels:
        list_training_id += ['remove-'+ '-'.join(args.remove_labels)]

    # if there is a </s> in the first tweet of the first train file, then you need to use token_type_ids for roberta 
    try:
        first_examples = read_csv_val(path_corpus+first_file_data_train).tweet.values[:10]
        # use_token_type_ids_roberta = '</s>' in first_ex
        # number of time it happens
        use_token_type_ids_roberta = np.max([k.count('</s>') for k in first_examples])

        # Force not using token_type_ids
        use_token_type_ids_roberta = use_token_type_ids_roberta * args.using_token_type_ids_roberta
        if use_token_type_ids_roberta: print('Using a special token_type_ids layer')
    except:
        use_token_type_ids_roberta = False
    if not args.using_token_type_ids_roberta:
        list_training_id += ['noTokTypeIds']

    # transfo4
    list_training_id += TRANSFO4*['transfo4']

    str_training_id = '_'.join(list_training_id)

    # the string is too long sometimes.... 
    str_training_id = reduce_size_str_training_id(str_training_id) 

    trained_modelFilePath = path_dump + 'fine-tuned_{}.hdf5'.format(str_training_id) 

    #################################### Initialization model ####################################

    # args.model_name_vision.split('efficient') # useless? Qu'est ce que je voulais faire ici? 
    
    # get the size of the features we are adding
    #### THIS NEED TO BE CHANGED AS IT IS TRUE ONLY FOR A VECTOR
    if bool_multimodal:
        file_path = path_dump + "inputs/input_img_{}.npy".format(args.test_set[0])
        input_feat_vector_dim = np.load(file_path).shape[1]
    else:
        input_feat_vector_dim = None
    
    model, tokenizer = compile_and_return_model(model_name_ini, OUTPUT_NETWORK, nb_classes, 
                             bool_multimodal, learning_rate, proxies=proxies, task_type = training_type, 
                                                model_name_vision = args.model_name_vision, first_block_to_uf = args.first_block_to_uf, 
                                                use_token_type_ids_roberta=use_token_type_ids_roberta, multi_labels = multi_labels, verbose=args.verbose,
                                               MAX_SEQ_LEN=args.max_seq_len, input_feat_vector_dim=input_feat_vector_dim)

    if continue_training:
        already_trained_modelFilePath = '_'.join(trained_modelFilePath.split('_')[:-1])+'.hdf5'
        model.load_weights(already_trained_modelFilePath, by_name=False)

    # TODO change that it means pre_training_text and pre_training_vision
    elif bool_multimodal and bool_already_fine_tuned_on_tweets:

        #path_nn_text_already_fine_tuned = path_dump + 'fine-tuned_roberta-base_task_humanitarian_text_img_agreed_lab_train.tsv_val_task_humanitarian_text_img_agreed_lab_dev.tsv_val_all-epoch_1e-06_CW_output-type7-best-model-checkpoint-val_acc.hdf5'
        #path_nn_text_already_fine_tuned = path_dump + 'fine-tuned_roberta-base_task_humanitarian_text_img_agreed_lab_event-type_train.tsv_val_task_humanitarian_text_img_agreed_lab_event-type_dev.tsv_val_all-epoch_1e-06_CW_output-type7-best-model-checkpoint-val_acc.hdf5'
        #path_nn_vision_already_fine_tuned = path_dump + 'vision/' 'best-{}-checkpoint-val_accuracy-unfreeze7.hdf5'.format(args.model_name_vision)

        model.load_weights(already_trained_modelFilePath_text, by_name=True)
        model.load_weights(already_trained_modelFilePath_vision, by_name=True)

        print('Loading already trained models: ')
        print('Text: ', os.path.split(already_trained_modelFilePath_text)[-1])
        print('Vision: ', os.path.split(already_trained_modelFilePath_vision)[-1])

    elif args.pre_training_vision:
        model.load_weights(already_trained_modelFilePath_vision, by_name=True)
        print('Loading already trained model: ')
        print('Image: ', os.path.split(already_trained_modelFilePath_vision)[-1]) 

    # loading an already fine-tuned model 
    # TODO: put the path into args
    elif args.pre_training_text: 
        # INIT_PATH = '/home/emmproc/'
        INIT_PATH = '/eos/jeodpp/data/projects/REFOCUS/'
        # Interet a utiliser avec zero-shot : 

        fn = return_filename_of_pretrained_model(args.pre_training_text)

        # new way to find the already_trained_file
        # TODO: change because I cannot use a model trained on another dataset 
        if not fn:
            fn = os.path.split(already_trained_modelFilePath_text)[-1]

        # model.load_weights(INIT_PATH + 'data/Valentin/tweets_alex/dump/'+fn)
        model.load_weights(fn, by_name=True, skip_mismatch=True)
        print('Loading an already trained model: ')
        print('Text: ', fn)

        # TODO: Change this one, this is the BERT from tf.hub trained on t4s_forBERT
        # model.load_weights(INIT_PATH + 'out-val/fine-tuned-bert.hdf5')

    #################################### Training ####################################

    if (not test_model_only) and (not args.predict_only):
        # create a new path to save the model, in order not to erase the last one
        if args.save_new_model_file:
            if os.path.isfile(trained_modelFilePath): 
                list_models = glob(trained_modelFilePath[:-5] + '*')
                # if there is at least one
                if len(list_models):
                    list_int = [int(k) for path_model in list_models for k in re.findall(r'_v(\d+)\.h', path_model) if k]
                    if len(list_int):
                        new_v = np.max(list_int) + 1
                    else: 
                        new_v = 2
                    print('Warning: Changing variable trained_modelFilePath by using _v%d'%new_v)
                    trained_modelFilePath = trained_modelFilePath[:-5] + '_v%d'%new_v + trained_modelFilePath[-5:]

        fineTuneBERT(trained_modelFilePath, model, tokenizer, path_corpus, dict_lab=dict_lab, SPARSE_CATEGORICAL=SPARSE_CATEGORICAL, 
                     cumsum_label_vectors=cumsum_label_vectors, model_name=model_name, list_inputFile_tr=input_data_File_train, 
                     inputFile_dev=input_data_File_dev, bool_with_dev=bool_with_dev, bool_sample_weight_dev=bool_weight_language_on_dev, USE_TENSORBOARD=USE_TENSORBOARD, 
                     update_freq_tboard=update_freq_tboard, batch_size=batch_size_training, nb_epochs=nb_epochs, steps_per_epoch=steps_per_epoch, 
                     validation_split=validation_split, bool_class_weight=bool_class_weight, PRINT_F1=PRINT_F1, DROP_MODEL=DROP_MODEL, 
                     bool_multimodal=bool_multimodal, monitor_value=monitor_value, patience=patience, min_delta=min_delta,
                     monitor_value_es=monitor_value_es, verbose=args.verbose, test_on_epoch_end = test_on_epoch_end, display_pg_bar = args.display_pg_bar,
                    multi_labels=multi_labels, dict_args=vars(args))

    if DROP_MODEL:
        saved_trained_modelFilePath = trained_modelFilePath[:-5]+'-best-model-checkpoint-{}.hdf5'.format(monitor_value)
        # DEBUG
        # saved_trained_modelFilePath = trained_modelFilePath[:-5]+'.hdf5'
    else:
        saved_trained_modelFilePath = trained_modelFilePath

    if args.model_to_load:
        # saved_trained_modelFilePath = '/home/emmproc/data/Valentin/models/' + args.model_to_load # 'tf_roberta_english_tweets.hdf5'
        saved_trained_modelFilePath = PATH_DATA + 'models/' + args.model_to_load # 'tf_roberta_english_tweets.hdf5'
        # dict_lab = {'positive' : 0, 'negative' : 1, 'neutral' : 2}
        model.load_weights(saved_trained_modelFilePath, by_name=True, skip_mismatch=True)
        # import pdb; pdb.set_trace()
        print('Loading already trained model: ')
        print('Text: ', saved_trained_modelFilePath)
    # For zero-shot
    # TODO: change, to specific to my code... just for text. 
    elif not os.path.isfile(saved_trained_modelFilePath):
        print('Warning --> %s not existing'%saved_trained_modelFilePath)
        args.zero_shot = False
        if args.zero_shot:
            saved_trained_modelFilePath = fn

    #################################### Predicting only ####################################
    # args.predict_only = False
    if args.predict_only:
        list_files_to_test = args.test_set
        for input_data_File in list_files_to_test:
            predict_only(saved_trained_modelFilePath, model, input_data_File=input_data_File, tokenizer=tokenizer,
                     dict_lab=dict_lab, path_corpus=path_corpus_test, model_name=model_name, bool_multimodal=bool_multimodal,
                               ordinal_data=ordinal_data, dict_args = vars(args), create_new_tsv = args.create_new_tsv)
        import sys
        sys.exit(0)

    #################################### Bias only ####################################

    if args.calculate_sentiment_bias:
        from biases_calculation_huggingfacehub import calculate_sentiment_bias
        list_files_to_test = args.test_set
        tokenizer.not_use_token_type_ids = False
        for input_data_File in list_files_to_test:
            n_duplicates = 50
            list_countries = ['France', 'United_Kingdom', 'Ireland', 'Spain', 'Germany', 'Italy', 'Morocco', 
                              'Hungary', 'Poland', 'Estonia', 'Finland', 'Portugal', 'India', 'Russia', 'Turkey',
                             'Greece', 'Sweden', 'the_Netherlands']
            # Only for English experiement
            list_countries += ['United_States', 'Australia', 'New_Zealand', 'Canada', 'South_Africa']
            df_bias = calculate_sentiment_bias(saved_trained_modelFilePath, model, input_data_File=input_data_File, tokenizer=tokenizer,
                     dict_lab=dict_lab, path_corpus=path_corpus_test, list_countries=list_countries, n_duplicates=n_duplicates)
            print(df_bias)
            df_bias.to_csv(path_corpus_test+'bias_%s'%input_data_File, sep='\t')
        import sys
        sys.exit(0)

    #################################### Testing ####################################

    # list_files_to_test = ['one_target_train.tsv', 'one_target_dev.tsv', 'one_target_test.tsv']
    # list_files_to_test = [args.test_set]

    # always a list unless there is no argument and it's tweeti_separate_lang
    list_files_to_test = args.test_set
    if 'tweeti_all_lang' == args.test_set:
        list_files_to_test = ['tweeti-b.dev.dist.Tweets.tsv.utf8.{}_clean_val'.format(lan_str) for lan_str in ['en', 'de', 'fr', 'it', 'es', 'multilingual']]

    # test also on the developpement set : if we are not just in test mode, and there is a dev set 
    if bool_with_dev and (not test_model_only):
        list_files_to_test.insert(0, args.dev_set)
    str_repport = ""
    for input_data_File in list_files_to_test:
        str_repport += testBERT(saved_trained_modelFilePath, model, input_data_File=input_data_File, tokenizer=tokenizer,
                     dict_lab=dict_lab, path_corpus=path_corpus_test, model_name=model_name, bool_multimodal=bool_multimodal,
                               ordinal_data=ordinal_data, dict_args = vars(args), verbose = args.verbose)

    # Save the results on the test in a txt file
    if not test_model_only:
        fn_results='{}/results/{}.results.txt'
    else:
        fn_results='{}/results_test_model_only/{}.results.txt'

    p, fn = os.path.split(saved_trained_modelFilePath)
    os.makedirs(p+'/results_test_model_only/', exist_ok = True)
    #os.makedirs(p+'/results/', exist_ok = True)
    with open(fn_results.format(p, fn), 'w') as f:
        f.write(str_repport)
    #with open(saved_trained_modelFilePath + '.results.txt', 'w') as f:
    #    f.write(str_repport)