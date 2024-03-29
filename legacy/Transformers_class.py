# -*- coding: utf-8 -*-
"""
Created on 15/11/20

@author: Valentin

"""
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from transformers import *
import numpy as np
from transformers import __version__ as tv
CACHE_DIR = '/home/barriva/data/.cache/torch/transformers' + "3"*(tv.split('.')[0] == "3")

def return_model_name_to_general_type(model_name):
    """
    Return the general type of the model, used to load the model from MODELS_TF_dict
    
    Used in compile_and_return_model()
    """
    if 'camembert' in model_name:
        general_type = 'camembert-transformers'
    elif 'cardiff' in model_name:
        general_type = 'automodel'
    elif 'tf-flaubert' in model_name:
        general_type = 'automodel'
    elif 'gottbert' in model_name:
        general_type = 'automodel'
    elif 'bert-base' in model_name:
        general_type = 'bert-transformers'
    elif 'xlm-roberta' in model_name:
        general_type = 'xlm-roberta-transformers'
    elif 'roberta' in model_name:
        general_type = 'roberta-transformers'
    elif 't5' in model_name:
        general_type = 't5-transformers'
    elif 'electra' in model_name:
        general_type = 'electra-transformers'
    # elif 'longformer' in model_name:
    #     general_type = 'longformer-transformers'
    elif 'bert-tfhub' in model_name:
        general_type = model_name
    elif 'alberto' in model_name:
        general_type =  'automodel'
    else:
        print('Using AutoModel for {}'.format(model_name))
        general_type =  'automodel'
        # raise ValueError('What is the model type? model_name is {}'.format(model_name))
        
    return general_type
    
def return_model_tokenize_to_load(model_name):
    """
    Return the good strings in order to dl the models from the huggingface API
    
    Used in compile_and_return_model()
    """
    if 'xlm-roberta' in model_name:
        model_to_load = 'jplu/tf-xlm-roberta-'
        if 'large' in model_name:
            model_to_load += 'large'
        else:
            model_to_load += 'base'
    elif 'electra' in model_name:
        model_to_load = 'google/' + model_name
    elif 'camembert' in model_name:
        model_to_load = 'jplu/tf-camembert-'
        if 'large' in model_name:
            model_to_load += 'large' # does not exist yet
        else:
            model_to_load += 'base'
    elif 'tf-flaubert' in model_name:
        model_to_load = 'jplu/' + model_name   
    elif 'alberto' in model_name:
        # model_to_load = '/home/emmproc/data/Valentin/models/alberto_uncased_L-12_H-768_A-12_italian_huggingface.co'
        model_to_load = 'm-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0'
    elif 'cardiff' in model_name:
        model_to_load = 'cardiffnlp/twitter-roberta-base'
        if 'emotion' in model_name:
            model_to_load += '-emotion'
        elif 'sentiment' in model_name:
            model_to_load += '-sentiment'
    elif 'scibert' in model_name:
        model_to_load = 'allenai/'+model_name
    else:
        model_to_load = model_name
            
    if 'alberto' in model_name:
        #from models.tokenizer import AlBERTo_Preprocessing, AlBERToTokenizer
        #tokenizer = #goback
        tokenizer_to_load = 'm-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0'
    # elif 'gottbert' in model_name:
        # tokenizer_to_load = '/eos/jeodpp/data/projects/EMM/data/Valentin/models/gottbert-base'
    else:
        #tokenizer = tok_gen.from_pretrained(tokenizer_to_load, proxies=proxies)
        tokenizer_to_load = model_to_load
        
    return model_to_load, tokenizer_to_load
            

def return_model_type_img(EFFICIENT_TYPE):
    """
    Return the classical parameters for the type of EfficientNet used
    https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
    
    Used in build_model_efficient_sequential()
    """
    if EFFICIENT_TYPE == 4:
        model_type_efficient = tf.keras.applications.EfficientNetB4
        IMG_SIZE = 380 # for EfficientNetB4 ; 260 B2 ; 224 B0
    elif EFFICIENT_TYPE == 2:
        model_type_efficient = tf.keras.applications.EfficientNetB2
        IMG_SIZE = 260 # for EfficientNetB4 ; 260 B2 ; 224 B0
    elif EFFICIENT_TYPE == 0:
        model_type_efficient = tf.keras.applications.EfficientNetB0
        IMG_SIZE = 224 # for EfficientNetB4 ; 260 B2 ; 224 B0
    elif EFFICIENT_TYPE == 3:
        model_type_efficient = tf.keras.applications.EfficientNetB3
        IMG_SIZE = 300 # for EfficientNetB4 ; 260 B2 ; 224 B0
    elif EFFICIENT_TYPE == -1:
        model_type_efficient = tf.keras.applications.DenseNet121    
        IMG_SIZE = 224
        
    return model_type_efficient, IMG_SIZE

def data_augmentation(IMG_SIZE=None, random_flip="horizontal_and_vertical", random_rot=0.2):
    return tf.keras.Sequential([
        # tf.keras.layers.experimental.preprocessing.RandomCrop(IMG_SIZE, IMG_SIZE),
        tf.keras.layers.experimental.preprocessing.RandomFlip(random_flip),
        tf.keras.layers.experimental.preprocessing.RandomRotation(random_rot), # normally should be 0.01
    ])

def build_model_efficient_sequential(EFFICIENT_TYPE, first_block_to_uf=8, 
                                     path_model_vision_to_load = None, verbose=1):
    """
    Return the EfficientNet model and the input_img layer 
    first_block_to_uf is the number of the first block to uf, must be an integer between 0 and 7. 0 means fine-tuning the whole network
    
    Used in compile_and_return_model()
    """
    
    model_type_efficient, IMG_SIZE = return_model_type_img(EFFICIENT_TYPE)
    
    rotation_value = 0.2 # hyperparameters
    random_flip = "horizontal_and_vertical"
    # Mettre meme chose que pour le modele emotion 
    # data_augmentation = tf.keras.Sequential([
        # tf.keras.layers.experimental.preprocessing.RandomCrop(IMG_SIZE, IMG_SIZE),
        # tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        # tf.keras.layers.experimental.preprocessing.RandomRotation(rotation_value),
    # ])
    
    # input_img = layers.Input(shape=(None, None, 3))
    input_img = layers.Input(shape=(None, None, 3), dtype=tf.int32, name="input_img")
    inputs_augmented = data_augmentation(IMG_SIZE=IMG_SIZE, random_flip=random_flip, random_rot=rotation_value)(input_img)
    
    # x = img_augmentation(inputs)
    model_img = model_type_efficient(
        include_top=False, # whether to include the fully-connected layer at the top of the network.
        weights='imagenet', 
        input_tensor=inputs_augmented, # optional Keras tensor (i.e. output of `layers.Input()`) to use as image input for the model. 
        # If not, create a `layers.Input()` inside the EfficientNet
        input_shape=(IMG_SIZE, IMG_SIZE, 3), # only to be specified if `include_top` (`input_tensor`???) is False. It should have exactly 3 inputs channels.
        # pooling=None, # when `include_top` is `False`.
        # classes=2048, # number of 'classes'
        # classifier_activation='softmax'
    )

    # We unfreeze the top layers while leaving BatchNorm layers frozen
    # if 0 then all the network will be fine-tuned 
    if first_block_to_uf > 0:
        # Freeze the pretrained weights
        model_img.trainable = False
        # if 8 then all the weights will stay frozen
        if first_block_to_uf < 9:
            bool_start_unfreezing = False
            for idx_layer, layer in enumerate(model_img.layers):
                
                if 'block%d'%first_block_to_uf in layer.name:
                    bool_start_unfreezing = True
                    
                if bool_start_unfreezing or ('top_conv' in layer.name): # only for the case where first_block_to_uf=8
                    if not isinstance(layer, layers.BatchNormalization):
                        layer.trainable = True   
    # this part was for testing only... 
    elif first_block_to_uf == -1:
        model_img.trainable = False
        # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
        for layer in model_img.layers[-13:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model_img.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    outputs = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    
    # load a model pre-trained in an unimodal way
    if path_model_vision_to_load:
        num_classes = 5
        outputs = layers.Dense(num_classes, activation="softmax", name="pred")(outputs)
        model = tf.keras.Model(inputs, outputs, name="ImageEncoder")
        model.load_weights(path_model_vision_to_load)
        outputs = model.layers[-2].output
        # in order to remove the last layer
        model = tf.keras.Model(input_img, outputs, name="ImageEncoder")
    else:
        model = tf.keras.Model(input_img, outputs, name="ImageEncoder")
    
    if verbose:
        trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
        non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])
        print('Total params: {:,}'.format(trainable_count + non_trainable_count))
        print('Trainable params: {:,}'.format(trainable_count))
        print('Non-trainable params: {:,}'.format(non_trainable_count))
    if verbose == 2: 
        model.summary()
        
    return model, input_img

def model_pt_or_tf(str_model_to_load):
    """
    If we load a model that is from pt...
    """
    if (('RuPERTa' in str_model_to_load) or 
        ('EstBERT' in str_model_to_load) or
        ('spanberta' in str_model_to_load) or
        ('dccuchile/bert-base-spanish-wwm-uncased' in str_model_to_load) or
        ('gottbert' in str_model_to_load) or
        ('deberta' in str_model_to_load) or
        ('scibert' in str_model_to_load) or
        ('galactica' in str_model_to_load) or
        ('SLR/mlm' in str_model_to_load)
       ):
        from_pt = True
    else:
        from_pt = False
        
    return from_pt
        
def compile_and_return_model(model_name, OUTPUT_NETWORK, nb_classes, bool_multimodal, learning_rate, 
    proxies=None, MAX_SEQ_LEN=128, task_type = '', model_name_vision = 'efficient2', first_block_to_uf = 7, 
                             use_token_type_ids_roberta=True, multi_labels=False, verbose=False,
                            input_feat_vector_dim=None):
    """
    Create the deep learning model, compile it and return it
    
    TODO: input_img_dim is hard-coded, but this value should come from the loaded dataset !!  
    """
    ### The models from the transformers library  
    
    MODELS_TF_dict = {'bert-transformers' : (TFBertModel, BertTokenizer),
                 'xlm-roberta-transformers' : (TFXLMRobertaModel, XLMRobertaTokenizer),
                 'roberta-transformers' : (TFRobertaModel, RobertaTokenizer),
                # TODO: t5 does not work right now, even with pytorch.... 
                 't5-transformers' : (TFT5Model, T5Tokenizer), # 't5-11b' is supposed the best model at the end of 04/2020
                 'electra-transformers' : (TFElectraModel, ElectraTokenizer), # TODO: do the tests with Electra model 
                  'camembert-transformers' : (TFCamembertModel, CamembertTokenizer), 
                 'automodel' : (TFAutoModel, AutoTokenizer)
                 # 'longformer-transformers' : (TFLongformerModel, LongformerTokenizer), # TODO: do the tests with Longformer model 
             }

    ### Model compiling 

    # if True then label = [1, 2, 3] else [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # SPARSE_CATEGORICAL = False

    if 'lxmert' in model_name:
        from transformers import LxmertTokenizer
        tokenizer = LxmertTokenizer.from_pretrained('unc-nlp/lxmert-base-uncased', cache_dir = CACHE_DIR)
    else: # if using transformers library 
        if verbose:
            print('model type: {}, model name: {}'.format(return_model_name_to_general_type(model_name), model_name))
        model_gen, tok_gen = MODELS_TF_dict[return_model_name_to_general_type(model_name)]
        # tokenizer = tok_gen.from_pretrained(model_name, proxies=proxies)
        
        str_model_to_load, str_tokenizer_to_load = return_model_tokenize_to_load(model_name)

        tokenizer = tok_gen.from_pretrained(str_tokenizer_to_load, proxies=proxies, cache_dir = CACHE_DIR)
        from_pt = model_pt_or_tf(str_model_to_load)
        model_tf_transformer = model_gen.from_pretrained(str_model_to_load, proxies=proxies, cache_dir = CACHE_DIR, from_pt=from_pt)
        
    if ('roberta' in model_name) and use_token_type_ids_roberta:
        nb_sentences = use_token_type_ids_roberta + 1
        print('Adding a special token_type_embeddings layer to RoBERTa, of dimension {}'.format(nb_sentences))
        model_tf_transformer.config.type_vocab_size = nb_sentences 
        from transformers import __version__ as tv
        if int(tv.split('.')[0]) > 3:
            model_tf_transformer.roberta.embeddings.token_type_embeddings = tf.Variable(np.random.normal(0.0,model_tf_transformer.config.initializer_range, (model_tf_transformer.config.type_vocab_size, model_tf_transformer.config.hidden_size)).astype('float32'))
        else:
            model_tf_transformer.roberta.embeddings.token_type_embeddings = tf.keras.layers.Embedding(2, model_tf_transformer.config.hidden_size, embeddings_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=model_tf_transformer.config.initializer_range))
          
        tokenizer.create_token_type_ids = True
        
    elif 'tf-flaubert' in model_name: # if not using dictionnary as input... # TODO: change the input with a dictionnary... 
        model_tf_transformer.transformer.use_lang_emb = False
        
    # tokenizer = BertTokenizer.from_pretrained(model_name)
    # model_tf_transformer = TFBertForSequenceClassification.from_pretrained(model_name) # not the good one, as a another layer for classification, not looking for that 
    # model_tf_transformer = TFBertModel.from_pretrained(model_name)

    #input layers
    # MAX_SEQ_LEN=128
    input_word_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="segment_ids")
    
    if OUTPUT_NETWORK != 900: # TEST FOR INPUT_EMBEDS
        if 'electra' in model_name:
            outputs = model([input_word_ids, input_mask, segment_ids])
            sequence_output = outputs[0]  # The last hidden-states is the first element of the output tuple
        elif 't5' in model_name:
            decoder_input_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="decoder_input_ids")
            #outputs = model(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, decoder_input_ids=input_ids)
            outputs = model(inputs=input_ids, input_mask=input_mask, decoder_input_ids=decoder_input_ids)
            sequence_output = outputs[0]
        else:
            # old configuration of transformers library, now pooled_output is not used if not asked in params
            # sequence_output, pooled_output = model_tf_transformer([input_word_ids, input_mask, segment_ids])
            # TODO: verify if it's working... 
            # sequence_output = model_tf_transformer([input_word_ids, input_mask, segment_ids])[0]
            sequence_output = model_tf_transformer({'input_ids' : input_word_ids, 'attention_mask' : input_mask, 'token_type_ids' : segment_ids})[0]
        

    # TODO: look that 
    if OUTPUT_NETWORK == 1:
        #### NEW Network schema
        x = tf.keras.layers.Dense(512, activation="sigmoid", name="hidden-layer1")(pooled_output)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(512, activation="sigmoid", name="hidden-layer2")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        out = tf.keras.layers.Dense(nb_classes, activation="tanh", name="dense_output")(x)
        # (i.e. from_logits if no sigmoid activation at the last layer ???) there is tanh here so what??..... 
        from_logits = True
        ####
    elif OUTPUT_NETWORK == 2:
        ### Other Network schema --> Here you take a value from the sequence output instead of the pooled output
        x = sequence_output[:, 0, :] # token [CLS] used to get obtain an embedding used for further classification 
        out = tf.keras.layers.Dense(nb_classes, activation='sigmoid')(x)
        # if the output looks like a logits so, not bounded and not like probability distribution
        # (i.e. from_logits if no sigmoid activation at the last layer???)
        from_logits = False
        #### 
    elif OUTPUT_NETWORK == 3:
        ### ALMOST SAME config as in the class RobertaClassificationHead of `transformers` --> take `sequence_output[:, 0, :]`
        # dropout value from https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-config.json
        hidden_size_output = 768
        x = sequence_output[:, 0, :] # take <s> token (equiv. to [CLS])
        # HERE IT LACKS x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(hidden_size_output, activation="tanh", name="hidden-layer1")(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        # x = tf.keras.layers.Dense(512, activation="sigmoid", name="hidden-layer2")(x)
        # x = tf.keras.layers.Dropout(0.1)(x)
        out = tf.keras.layers.Dense(nb_classes, name="dense_output")(x)
        # if the output looks like a logits so, not bounded and not like probability distribution
        # (i.e. from_logits if no sigmoid activation at the last layer???)
        from_logits = True
        #### 
    elif OUTPUT_NETWORK == 7:
        ### SAME config as in the class RobertaClassificationHead of `transformers` --> take `sequence_output[:, 0, :]`
        # dropout value from https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-config.json
        hidden_size_output = 768
        x = sequence_output[:, 0, :] # take <s> token (equiv. to [CLS])
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(hidden_size_output, activation="tanh", name="hidden-layer1")(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        # x = tf.keras.layers.Dense(512, activation="sigmoid", name="hidden-layer2")(x)
        # x = tf.keras.layers.Dropout(0.1)(x)
        out = tf.keras.layers.Dense(nb_classes, name="dense_output")(x)
        # if the output looks like a logits so, not bounded and not like probability distribution
        # (i.e. from_logits if no sigmoid activation at the last layer???)
        from_logits = True
        #### 
    elif OUTPUT_NETWORK == 77:
        assert input_feat_vector_dim != None
        
        x = sequence_output[:, 0, :] # token [CLS] used to get obtain an embedding used for further classification 
        
        input_img = tf.keras.layers.Input(shape=(input_feat_vector_dim,), dtype=tf.float32, name="input_img")
        out_fused_scores = tf.concat([x, input_img], axis=1)

        hidden_out_size = 768
        out_fused_scores = tf.keras.layers.Dropout(0.1)(out_fused_scores)
        out_fused_scores = tf.keras.layers.Dense(hidden_out_size, activation="tanh", name="hidden-layer1")(out_fused_scores)
        out_fused_scores = tf.keras.layers.Dropout(0.1)(out_fused_scores)
        
        out = tf.keras.layers.Dense(nb_classes, name="dense_output")(out_fused_scores)
        # if the output looks like a logits so, not bounded and not like probability distribution
        # (i.e. from_logits if no sigmoid activation at the last layer???)
        from_logits = True
        
    elif OUTPUT_NETWORK == 4:
        ### Other Network schema --> Here you take a value from the sequence output instead of the pooled output
        x = sequence_output[:, 0, :] # token [CLS] used to get obtain an embedding used for further classification
        out = tf.keras.layers.Dense(nb_classes, activation='sigmoid')(x)
        # if the output looks like a logits so, not bounded and not like probability distribution
        # (i.e. from_logits if no sigmoid activation at the last layer???)
        from_logits = False # not used 
        #### 
    elif OUTPUT_NETWORK == 11: # MULTIMODAL : scores 
        ### Other Network schema --> Here you take a value from the sequence output instead of the pooled output
        x = sequence_output[:, 0, :] # token [CLS] used to get obtain an embedding used for further classification
        out_txt = tf.keras.layers.Dense(nb_classes, activation='sigmoid')(x)
        
        input_img_dim = 8
        input_img = tf.keras.layers.Input(shape=(input_img_dim,), dtype=tf.float32, name="input_img")
        out_fused_scores = tf.concat([out_txt, input_img], axis=1)

        out_fused_scores = tf.keras.layers.Dense(nb_classes, activation='sigmoid')(out_fused_scores)
        out = tf.keras.layers.Dense(nb_classes, activation='sigmoid')(out_fused_scores)
        # if the output looks like a logits so, not bounded and not like probability distribution
        # (i.e. from_logits if no sigmoid activation at the last layer???)
        from_logits = False
        #### 
    elif OUTPUT_NETWORK == 12: # MULTIMODAL : features 
        ### Other Network schema --> Here you take a value from the sequence output instead of the pooled output
        x = sequence_output[:, 0, :] # token [CLS] used to get obtain an embedding used for further classification 
        input_img_dim = 2048
        input_img = tf.keras.layers.Input(shape=(input_img_dim,), dtype=tf.float32, name="input_img")
        out_fused_scores = tf.concat([x, input_img], axis=1)

        hidden_out_size = 512
        out_fused_scores = tf.keras.layers.Dense(hidden_out_size, activation='sigmoid')(out_fused_scores)
        out = tf.keras.layers.Dense(nb_classes, activation='sigmoid')(out_fused_scores)
        # if the output looks like a logits so, not bounded and not like probability distribution
        # (i.e. from_logits if no sigmoid activation at the last layer???)
        from_logits = False
        #### 
    elif OUTPUT_NETWORK == 13: # MULTIMODAL : feature fusion and TRAINING THE VISION part 
        ### Other Network schema --> Here you take a value from the sequence output instead of the pooled output
        x = sequence_output[:, 0, :] # token [CLS] used to get obtain an embedding used for further classification 
        
        EFFICIENT_TYPE = int(model_name_vision[-1])
        model_img, input_img = build_model_efficient_sequential(EFFICIENT_TYPE, first_block_to_uf=first_block_to_uf)
        # 2 layers :  
        hidden_out_size = 512
        out_fused_scores = tf.concat([x, model_img.output], axis=1)
        out_fused_scores = tf.keras.layers.Dense(hidden_out_size, activation='sigmoid')(out_fused_scores)
        out = tf.keras.layers.Dense(nb_classes, activation='sigmoid')(out_fused_scores)
        # if the output looks like a logits so, not bounded and not like probability distribution
        # (i.e. from_logits if no sigmoid activation at the last layer???)
        from_logits = False       
        
    elif OUTPUT_NETWORK == 14: # 13 + Attention on image embedding, using the embedding of the first token of the event
        ### Other Network schema --> Here you take a value from the sequence output instead of the pooled output
        x = sequence_output[:, 0, :] # token [CLS] used to get obtain an embedding used for further classification 
        
        x_event = sequence_output[:, 1, :] # token [CLS] used to get obtain an embedding used for further classification 
        hidden_size_output = 768 # size of text embedding
        
        EFFICIENT_TYPE = int(model_name_vision[-1])
        model_img, input_img = build_model_efficient_sequential(EFFICIENT_TYPE, first_block_to_uf=first_block_to_uf)

        # TODO : test --> this looks weird 
        output_img = tf.keras.layers.Dense(hidden_size_output, activation='sigmoid')(model_img.output)
        attention_layer_vision = tf.keras.layers.Attention()
        output_img_att = attention_layer_vision([x_event, output_img])
        
        # 2 layers :  
        hidden_out_size = 512
        out_fused_scores = tf.concat([x, output_img_att], axis=1)
        out_fused_scores = tf.keras.layers.Dense(hidden_out_size, activation='sigmoid')(out_fused_scores)
        out = tf.keras.layers.Dense(nb_classes, activation='sigmoid')(out_fused_scores)
        # if the output looks like a logits so, not bounded and not like probability distribution
        # (i.e. from_logits if no sigmoid activation at the last layer???)
        from_logits = False 
        
    elif OUTPUT_NETWORK == 15: # 14 + Attention on text embedding, using the embedding of image
        ### Other Network schema --> Here you take a value from the sequence output instead of the pooled output
        x = sequence_output[:, 0, :] # token [CLS] used to get obtain an embedding used for further classification 
        
        x_event = sequence_output[:, 1, :] # token [CLS] used to get obtain an embedding used for further classification 
        
        EFFICIENT_TYPE = int(model_name_vision[-1])
        model_img, input_img = build_model_efficient_sequential(EFFICIENT_TYPE, first_block_to_uf=first_block_to_uf)
        
        attention_layer_vision = tf.keras.layers.Attention()
        output_img_att = attention_layer_vision(x_event, model_img.output)
        
        attention_layer_text = tf.keras.layers.Attention()
        output_text_att = attention_layer_text(model_img.output, x)
        
        # 2 layers :  
        hidden_out_size = 512
        out_fused_scores = tf.concat([output_text_att, output_img_att], axis=1)
        out_fused_scores = tf.keras.layers.Dense(hidden_out_size, activation='sigmoid')(out_fused_scores)
        out = tf.keras.layers.Dense(nb_classes, activation='sigmoid')(out_fused_scores)
        # if the output looks like a logits so, not bounded and not like probability distribution
        # (i.e. from_logits if no sigmoid activation at the last layer???)
        from_logits = False 
        
    ###### Le mieux code (dropout, commentaires, etc...) ######
    elif OUTPUT_NETWORK == 16: # 131 + Attention on image embedding, using the embedding of the first token of the event
        ### Other Network schema --> Here you take a value from the sequence output instead of the pooled output
        x = sequence_output[:, 0, :] # token [CLS] used to get obtain an embedding used for further classification 
        hidden_size_output = 768 # size of text embedding
        # x_event = sequence_output[:, 1, :] # token [CLS] used to get obtain an embedding used for further classification 
        
        ####### Model vision #######
        EFFICIENT_TYPE = int(model_name_vision[-1])
        model_img, input_img = build_model_efficient_sequential(EFFICIENT_TYPE, first_block_to_uf=first_block_to_uf)
        
        ####### Cross-Attention #######
        # vision reduced to same size than text
        output_img = tf.keras.layers.Dense(hidden_size_output, activation='sigmoid')(model_img.output)
        output_img = tf.keras.layers.Dropout(0.1)(output_img)
        # attention on vision
        attention_layer_vision = tf.keras.layers.Attention()
        output_img_att = attention_layer_vision([x, output_img])
        
        ######## output layers #######
        hidden_out_size = 512
        out_fused_scores = tf.concat([x, output_img_att], axis=1, name='concat_MM')
        out_fused_scores = tf.keras.layers.Dropout(0.1)(out_fused_scores)
        out_fused_scores = tf.keras.layers.Dense(hidden_out_size, activation='sigmoid', name='dense_MM')(out_fused_scores)
        out_fused_scores = tf.keras.layers.Dropout(0.1)(out_fused_scores)
        out = tf.keras.layers.Dense(nb_classes, name='dense_output_MM')(out_fused_scores)
        # if the output looks like a logits so, not bounded and not like probability distribution
        # (i.e. from_logits if no sigmoid activation at the last layer???)
        from_logits = True 
        
    elif OUTPUT_NETWORK == 17: # 14 + Attention on text embedding, using the embedding of image
        ### Other Network schema --> Here you take a value from the sequence output instead of the pooled output
        x = sequence_output[:, 0, :] # token [CLS] used to get obtain an embedding used for further classification 
        hidden_size_output = 768 # size of text embedding
        
        # x_event = sequence_output[:, 1, :] # token [CLS] used to get obtain an embedding used for further classification 
        
        EFFICIENT_TYPE = int(model_name_vision[-1])
        model_img, input_img = build_model_efficient_sequential(EFFICIENT_TYPE, first_block_to_uf=first_block_to_uf)
        
        
        # TODO : test --> this looks weird 
        output_img = tf.keras.layers.Dense(hidden_size_output, activation='sigmoid')(model_img.output)
        attention_layer_vision = tf.keras.layers.Attention()
        output_img_att = attention_layer_vision([x, output_img])
        attention_layer_text = tf.keras.layers.Attention()
        output_text_att = attention_layer_text([output_img, x])
        
        # 2 layers :  
        hidden_out_size = 512
        out_fused_scores = tf.concat([output_text_att, output_img_att], axis=1)
        out_fused_scores = tf.keras.layers.Dense(hidden_out_size, activation='sigmoid')(out_fused_scores)
        out = tf.keras.layers.Dense(nb_classes, activation='sigmoid')(out_fused_scores)
        # if the output looks like a logits so, not bounded and not like probability distribution
        # (i.e. from_logits if no sigmoid activation at the last layer???)
        from_logits = False 
        
    # TEST --> This should be the standard feature fusion, using dropout
    elif OUTPUT_NETWORK == 131: # MULTIMODAL : feature fusion and TRAINING THE VISION part 
        ### Other Network schema --> Here you take a value from the sequence output instead of the pooled output
        x = sequence_output[:, 0, :] # token [CLS] used to get obtain an embedding used for further classification 
        hidden_size_output = 768 # size of text embedding
        # x_event = sequence_output[:, 1, :] # token [CLS] used to get obtain an embedding used for further classification 
        
        ####### Model vision #######
        EFFICIENT_TYPE = int(model_name_vision[-1])
        model_img, input_img = build_model_efficient_sequential(EFFICIENT_TYPE, first_block_to_uf=first_block_to_uf)
        
        ######## output layers #######
        hidden_out_size = 512 # --> why this size ? 
        out_fused_scores = tf.concat([x, model_img.output], axis=1, name='concat_MM')
        out_fused_scores = tf.keras.layers.Dropout(0.1)(out_fused_scores)
        out_fused_scores = tf.keras.layers.Dense(hidden_out_size, activation='sigmoid', name='dense_MM')(out_fused_scores)
        out_fused_scores = tf.keras.layers.Dropout(0.1)(out_fused_scores)
        out = tf.keras.layers.Dense(nb_classes, name='dense_output_MM')(out_fused_scores)
        # if the output looks like a logits so, not bounded and not like probability distribution
        # (i.e. from_logits if no sigmoid activation at the last layer???)
        from_logits = True 
        
    elif OUTPUT_NETWORK == 132: # 13 avec dropout  
        ### Other Network schema --> Here you take a value from the sequence output instead of the pooled output
        x = sequence_output[:, 0, :] # token [CLS] used to get obtain an embedding used for further classification 
        
        EFFICIENT_TYPE = int(model_name_vision[-1])
        model_img, input_img = build_model_efficient_sequential(EFFICIENT_TYPE, first_block_to_uf=first_block_to_uf)
        # 2 layers :  
        hidden_out_size = 512
        out_fused_scores = tf.concat([x, model_img.output], axis=1)
        out_fused_scores = tf.keras.layers.Dense(hidden_out_size, activation='sigmoid')(out_fused_scores)
        out_fused_scores = tf.keras.layers.Dropout(0.1)(out_fused_scores) # added here 132
        
        # from_logits=False + sigmoid --> ce qui est chelou 
        out = tf.keras.layers.Dense(nb_classes, activation='sigmoid')(out_fused_scores)
        # if the output looks like a logits so, not bounded and not like probability distribution
        # (i.e. from_logits if no sigmoid activation at the last layer???)
        from_logits = False 
        
    ############### MARCHE PAS BIEN ############### 
    elif OUTPUT_NETWORK == 133: # 132 + from_logits=True + pas d'activation
        ### Other Network schema --> Here you take a value from the sequence output instead of the pooled output
        x = sequence_output[:, 0, :] # token [CLS] used to get obtain an embedding used for further classification 
        
        EFFICIENT_TYPE = int(model_name_vision[-1])
        model_img, input_img = build_model_efficient_sequential(EFFICIENT_TYPE, first_block_to_uf=first_block_to_uf)
        # 2 layers :  
        hidden_out_size = 512
        out_fused_scores = tf.concat([x, model_img.output], axis=1)
        out_fused_scores = tf.keras.layers.Dense(hidden_out_size, activation='sigmoid')(out_fused_scores)
        out_fused_scores = tf.keras.layers.Dropout(0.1)(out_fused_scores)
        out = tf.keras.layers.Dense(nb_classes)(out_fused_scores)
        # if the output looks like a logits so, not bounded and not like probability distribution
        # (i.e. from_logits if no sigmoid activation at the last layer???)
        from_logits = True         
    
    ############### MARCHE LE MIEUX ###############
    ###### Devrait pas bien marcher, comme 133 --> marche le mieux en fait !
    elif OUTPUT_NETWORK == 134: # 133 + tanh
        ### Other Network schema --> Here you take a value from the sequence output instead of the pooled output
        x = sequence_output[:, 0, :] # token [CLS] used to get obtain an embedding used for further classification 
        
        EFFICIENT_TYPE = int(model_name_vision[-1])
        model_img, input_img = build_model_efficient_sequential(EFFICIENT_TYPE, first_block_to_uf=first_block_to_uf)
        # 2 layers :  
        hidden_out_size = 512
        out_fused_scores = tf.concat([x, model_img.output], axis=1)
        out_fused_scores = tf.keras.layers.Dense(hidden_out_size, activation='tanh')(out_fused_scores)
        out_fused_scores = tf.keras.layers.Dropout(0.1)(out_fused_scores)
        out = tf.keras.layers.Dense(nb_classes)(out_fused_scores)
        # if the output looks like a logits so, not bounded and not like probability distribution
        # (i.e. from_logits if no sigmoid activation at the last layer???)
        from_logits = True         

    elif OUTPUT_NETWORK == 135: # 132 + tanh + from_logits=False + sigmo
        ### Other Network schema --> Here you take a value from the sequence output instead of the pooled output
        x = sequence_output[:, 0, :] # token [CLS] used to get obtain an embedding used for further classification 
        
        EFFICIENT_TYPE = int(model_name_vision[-1])
        model_img, input_img = build_model_efficient_sequential(EFFICIENT_TYPE, first_block_to_uf=first_block_to_uf)
        # 2 layers :  
        hidden_out_size = 512
        out_fused_scores = tf.concat([x, model_img.output], axis=1)
        out_fused_scores = tf.keras.layers.Dense(hidden_out_size, activation='tanh')(out_fused_scores)
        out_fused_scores = tf.keras.layers.Dropout(0.1)(out_fused_scores)
        out = tf.keras.layers.Dense(nb_classes, activation='sigmoid')(out_fused_scores)
        # if the output looks like a logits so, not bounded and not like probability distribution
        # (i.e. from_logits if no sigmoid activation at the last layer???)
        from_logits = False   
        
    ### Marche pas du tout ##### 
    elif OUTPUT_NETWORK == 136: # 134(qui marche le mieux) + dropout sur les sorties des unimodaux 
        ### Other Network schema --> Here you take a value from the sequence output instead of the pooled output
        x = sequence_output[:, 0, :] # token [CLS] used to get obtain an embedding used for further classification 
        
        EFFICIENT_TYPE = int(model_name_vision[-1])
        model_img, input_img = build_model_efficient_sequential(EFFICIENT_TYPE, first_block_to_uf=first_block_to_uf)
        # 2 layers :  
        hidden_out_size = 512
        out_fused_scores = tf.concat([x, model_img.output], axis=1)
        out_fused_scores = tf.keras.layers.Dropout(0.1)(out_fused_scores)
        out_fused_scores = tf.keras.layers.Dense(hidden_out_size, activation='tanh')(out_fused_scores)
        out_fused_scores = tf.keras.layers.Dropout(0.1)(out_fused_scores)
        out = tf.keras.layers.Dense(nb_classes)(out_fused_scores)
        # if the output looks like a logits so, not bounded and not like probability distribution
        # (i.e. from_logits if no sigmoid activation at the last layer???)
        from_logits = True   
        
    elif OUTPUT_NETWORK == 137: # 135 + dropout 
        ### Other Network schema --> Here you take a value from the sequence output instead of the pooled output
        x = sequence_output[:, 0, :] # token [CLS] used to get obtain an embedding used for further classification 
        
        EFFICIENT_TYPE = int(model_name_vision[-1])
        model_img, input_img = build_model_efficient_sequential(EFFICIENT_TYPE, first_block_to_uf=first_block_to_uf)
        # 2 layers :  
        hidden_out_size = 512
        out_fused_scores = tf.concat([x, model_img.output], axis=1)
        out_fused_scores = tf.keras.layers.Dropout(0.1)(out_fused_scores)
        out_fused_scores = tf.keras.layers.Dense(hidden_out_size, activation='tanh')(out_fused_scores)
        out_fused_scores = tf.keras.layers.Dropout(0.1)(out_fused_scores)
        out = tf.keras.layers.Dense(nb_classes, activation='sigmoid')(out_fused_scores)
        # if the output looks like a logits so, not bounded and not like probability distribution
        # (i.e. from_logits if no sigmoid activation at the last layer???)
        from_logits = False   
        
        
    # Ici on fait un max pour fusionner 
    elif OUTPUT_NETWORK == 140: # 171 + self attention over MM 
        ### Other Network schema --> Here you take a value from the sequence output instead of the pooled output
        x = sequence_output[:, 0, :] # token [CLS] used to get obtain an embedding used for further classification 
        hidden_size_output = 768 # size of text embedding
        # x_event = sequence_output[:, 1, :] # token [CLS] used to get obtain an embedding used for further classification 
        
        ####### Model vision #######
        EFFICIENT_TYPE = int(model_name_vision[-1])
        model_img, input_img = build_model_efficient_sequential(EFFICIENT_TYPE, first_block_to_uf=first_block_to_uf)
        
        ####### Cross-Attention #######
        # vision reduced to same size than text
        output_img = tf.keras.layers.Dense(hidden_size_output, activation='sigmoid')(model_img.output)
        output_img = tf.keras.layers.Dropout(0.1)(output_img)
        
        # vision reduced to same size than text
        x = tf.keras.layers.Dense(x, activation='sigmoid')(model_img.output)
        x = tf.keras.layers.Dropout(0.1)(x)

        ######## Fusion #######
        out_fused_scores = tf.max(tf.concat([x[None,...], output_img[None,...]], axis=0), axis=0, name='max_MM')
        ######## output layers #######
        hidden_out_size = 512
        out_fused_scores = tf.keras.layers.Dropout(0.1)(out_fused_scores)
        self_attention_layer_MM = tf.keras.layers.Attention()
        out_fused_scores = self_attention_layer_MM([out_fused_scores, out_fused_scores])
        out_fused_scores = tf.keras.layers.Dropout(0.1)(out_fused_scores)
        out_fused_scores = tf.keras.layers.Dense(hidden_out_size, activation='sigmoid', name='dense_MM')(out_fused_scores)
        out_fused_scores = tf.keras.layers.Dropout(0.1)(out_fused_scores)
        out = tf.keras.layers.Dense(nb_classes, name='dense_output_MM')(out_fused_scores)
        # if the output looks like a logits so, not bounded and not like probability distribution
        # (i.e. from_logits if no sigmoid activation at the last layer???)
        from_logits = True 
        
    ###### Le mieux code (dropout, commentaires, etc...) ###### TODO: coder toutes les output comme celui la, pour que ce soit clair
    elif OUTPUT_NETWORK == 161: # 131 + Attention on image embedding, using the embedding of the first token of the event
        ### Other Network schema --> Here you take a value from the sequence output instead of the pooled output
        x = sequence_output[:, 0, :] # token [CLS] used to get obtain an embedding used for further classification 
        hidden_size_output = 768 # size of text embedding
        # x_event = sequence_output[:, 1, :] # token [CLS] used to get obtain an embedding used for further classification 
        
        ####### Model vision #######
        EFFICIENT_TYPE = int(model_name_vision[-1])
        model_img, input_img = build_model_efficient_sequential(EFFICIENT_TYPE, first_block_to_uf=first_block_to_uf)
        
        ####### Cross-Attention #######
        # vision reduced to same size than text
        output_img = tf.keras.layers.Dense(hidden_size_output, activation='sigmoid')(model_img.output)
        output_img = tf.keras.layers.Dropout(0.1)(output_img)
        # attention on vision
        attention_layer_vision = tf.keras.layers.Attention()
        output_img_att = attention_layer_vision([x, output_img])
        
        ######## output layers #######
        hidden_out_size = 512
        out_fused_scores = tf.concat([x, output_img_att], axis=1, name='concat_MM')
        out_fused_scores = tf.keras.layers.Dropout(0.1)(out_fused_scores)
        out_fused_scores = tf.keras.layers.Dense(hidden_out_size, activation='sigmoid', name='dense_MM')(out_fused_scores)
        out_fused_scores = tf.keras.layers.Dropout(0.1)(out_fused_scores)
        out = tf.keras.layers.Dense(nb_classes, name='dense_output_MM')(out_fused_scores)
        # if the output looks like a logits so, not bounded and not like probability distribution
        # (i.e. from_logits if no sigmoid activation at the last layer???)
        from_logits = True 
        
    ###### Le mieux code (dropout, commentaires, etc...) ######
    elif OUTPUT_NETWORK == 171: # 131 + Attention on image embedding, using the embedding of the first token of the event + Attention on text using the image
        ### Other Network schema --> Here you take a value from the sequence output instead of the pooled output
        x = sequence_output[:, 0, :] # token [CLS] used to get obtain an embedding used for further classification 
        hidden_size_output = 768 # size of text embedding
        # x_event = sequence_output[:, 1, :] # token [CLS] used to get obtain an embedding used for further classification 
        
        ####### Model vision #######
        EFFICIENT_TYPE = int(model_name_vision[-1])
        model_img, input_img = build_model_efficient_sequential(EFFICIENT_TYPE, first_block_to_uf=first_block_to_uf)
        
        ####### Cross-Attention #######
        # vision reduced to same size than text
        output_img = tf.keras.layers.Dense(hidden_size_output, activation='sigmoid')(model_img.output)
        output_img = tf.keras.layers.Dropout(0.1)(output_img)
        # attention on vision
        attention_layer_vision = tf.keras.layers.Attention()
        output_img_att = attention_layer_vision([x, output_img])
        # attention on text
        attention_layer_text = tf.keras.layers.Attention()
        output_img_text = attention_layer_text([output_img, x])

        ######## output layers #######
        hidden_out_size = 512
        out_fused_scores = tf.concat([output_img_text, output_img_att], axis=1, name='concat_MM')
        out_fused_scores = tf.keras.layers.Dropout(0.1)(out_fused_scores)
        out_fused_scores = tf.keras.layers.Dense(hidden_out_size, activation='sigmoid', name='dense_MM')(out_fused_scores)
        out_fused_scores = tf.keras.layers.Dropout(0.1)(out_fused_scores)
        out = tf.keras.layers.Dense(nb_classes, name='dense_output_MM')(out_fused_scores)
        # if the output looks like a logits so, not bounded and not like probability distribution
        # (i.e. from_logits if no sigmoid activation at the last layer???)
        from_logits = True         

    ###### Le mieux code (dropout, commentaires, etc...) ######
    elif OUTPUT_NETWORK == 181: # 171 + self attention over MM 
        ### Other Network schema --> Here you take a value from the sequence output instead of the pooled output
        x = sequence_output[:, 0, :] # token [CLS] used to get obtain an embedding used for further classification 
        hidden_size_output = 768 # size of text embedding
        # x_event = sequence_output[:, 1, :] # token [CLS] used to get obtain an embedding used for further classification 
        
        ####### Model vision #######
        EFFICIENT_TYPE = int(model_name_vision[-1])
        model_img, input_img = build_model_efficient_sequential(EFFICIENT_TYPE, first_block_to_uf=first_block_to_uf)
        
        ####### Cross-Attention #######
        # vision reduced to same size than text
        output_img = tf.keras.layers.Dense(hidden_size_output, activation='sigmoid')(model_img.output)
        output_img = tf.keras.layers.Dropout(0.1)(output_img)
        # attention on vision
        attention_layer_vision = tf.keras.layers.Attention()
        output_img_att = attention_layer_vision([x, output_img])
        # attention on text
        attention_layer_text = tf.keras.layers.Attention()
        output_img_text = attention_layer_text([output_img, x])

        ######## output layers #######
        hidden_out_size = 512
        out_fused_scores = tf.concat([output_img_text, output_img_att], axis=1, name='concat_MM')
        out_fused_scores = tf.keras.layers.Dropout(0.1)(out_fused_scores)
        self_attention_layer_MM = tf.keras.layers.Attention()
        out_fused_scores = self_attention_layer_MM([out_fused_scores, out_fused_scores])
        out_fused_scores = tf.keras.layers.Dropout(0.1)(out_fused_scores)
        out_fused_scores = tf.keras.layers.Dense(hidden_out_size, activation='sigmoid', name='dense_MM')(out_fused_scores)
        out_fused_scores = tf.keras.layers.Dropout(0.1)(out_fused_scores)
        out = tf.keras.layers.Dense(nb_classes, name='dense_output_MM')(out_fused_scores)
        # if the output looks like a logits so, not bounded and not like probability distribution
        # (i.e. from_logits if no sigmoid activation at the last layer???)
        from_logits = True 

    elif OUTPUT_NETWORK == 90: # LXMERT cross-modal blocks are used 
        ### Other Network schema --> Here you take a value from the sequence output instead of the pooled output
        x = sequence_output[:, 0, :] # token [CLS] used to get obtain an embedding used for further classification 
        hidden_size_output = 768 # size of text embedding
        # x_event = sequence_output[:, 1, :] # token [CLS] used to get obtain an embedding used for further classification 
        
        ####### Model vision #######
        EFFICIENT_TYPE = int(model_name_vision[-1])
        model_img, input_img = build_model_efficient_sequential(EFFICIENT_TYPE, first_block_to_uf=first_block_to_uf)
        
        ####### Fusion #######
        #lxmert_no_pos = '/home/emmproc/data/Valentin/models/lxmert-base-uncased-without-pos'
        #lxmert_no_pos_size_efficient = '/home/emmproc/data/Valentin/models/lxmert-base-uncased-without-pos-size-efficient'
        lxmert_no_pos = '/eos/jeodpp/data/projects/REFOCUS/' + 'data/Valentin/models/lxmert-base-uncased-without-pos'
        lxmert_no_pos_size_efficient = '/eos/jeodpp/data/projects/REFOCUS/' + 'data/Valentin/models/lxmert-base-uncased-without-pos-size-efficient'
        from transformers import TFLxmertModel, LxmertConfig
        # TODO: change hardcoded 
        PRE_TRAINED_LXMERT = False
        if PRE_TRAINED_LXMERT == 'all':
            # I load a pretrained model 
            # model_fusion = TFLxmertModel.from_pretrained(lxmert_no_pos+'-size-efficient')
            # Or I just load a randomly initialized model, with the size of the visual_feat being model_img.output.shape
            model_fusion = TFLxmertModel.from_pretrained('unc-nlp/lxmert-base-uncased', cache_dir = CACHE_DIR)
            model_fusion.lxmert.encoder.visn_fc.use_visual_pos = False
            model_fusion.lxmert.use_visual_pos = False
            model_fusion.use_already_trained_embs = True
        elif PRE_TRAINED_LXMERT == 'half':
            config_lxm = LxmertConfig.from_json_file(lxmert_no_pos_size_efficient+'/config.json')
            model_fusion = TFLxmertModel(config_lxm)
            model_fusion.built = True
            model_fusion.load_weights(lxmert_no_pos+'/tf_model.h5', by_name=True, skip_mismatch=True)
            model_fusion.use_already_trained_embs = True
        else:
            config_lxm = LxmertConfig.from_json_file(lxmert_no_pos_size_efficient+'/config.json')
            model_fusion = TFLxmertModel(config_lxm)
            model_fusion.built = True
            model_fusion.load_weights(lxmert_no_pos+'/tf_model.h5', by_name=True, skip_mismatch=True)
            model_fusion.use_already_trained_embs = True
            
        # dict_input_lxmert = {'input_ids' : None, 'visual_feats' : model_img.output, 'inputs_embeds' : x}
        dict_input_lxmert = {'visual_feats' : model_img.output, 'inputs_embeds' : x}
        # dict_input_lxmert = {'input_ids' : input_word_ids, 'visual_feats' : model_img.output} # test pour voir si inputs_embeds merde ou alors visual_feats
        out_fused = model_fusion(dict_input_lxmert, return_dict=True)
        cls_emb = out_fused.pooled_output

        ######## output layers ####### --> Attention j'ai mis une tanh ici a la place d'une sigmoid (ca parait plus coherent avec OUTPUT_NETWORK=7)
        hidden_out_size = 512
        cls_emb = tf.keras.layers.Dropout(0.1)(cls_emb)
        cls_emb = tf.keras.layers.Dense(hidden_out_size, activation='tanh', name='dense_MM')(cls_emb)
        cls_emb = tf.keras.layers.Dropout(0.1)(cls_emb)
        out = tf.keras.layers.Dense(nb_classes, name='dense_output_MM')(cls_emb)
        # if the output looks like a logits so, not bounded and not like probability distribution
        # (i.e. from_logits if no sigmoid activation at the last layer???)
        from_logits = True
        
    elif OUTPUT_NETWORK == 900: # LXMERT cross-modal blocks are used ; not pre-trained
        ### Other Network schema --> Here you take a value from the sequence output instead of the pooled output
        # x = sequence_output[:, 0, :] # token [CLS] used to get obtain an embedding used for further classification 
        hidden_size_output = 768 # size of text embedding
        # x_event = sequence_output[:, 1, :] # token [CLS] used to get obtain an embedding used for further classification 
        
        ####### Model vision #######
        EFFICIENT_TYPE = int(model_name_vision[-1])
        model_img, input_img = build_model_efficient_sequential(EFFICIENT_TYPE, first_block_to_uf=first_block_to_uf)
        img_emb = model_img.output
        
        ####### Fusion ####### 
        # lxmert_no_pos = '/home/emmproc/data/Valentin/models/lxmert-base-uncased-without-pos'
        lxmert_no_pos = '/eos/jeodpp/data/projects/REFOCUS/' + 'data/Valentin/models/lxmert-base-uncased-without-pos'

        from transformers import TFLxmertModel
        # TODO: change hardcoded 
        PRE_TRAINED_LXMERT = False
        if PRE_TRAINED_LXMERT:
            # I load a pretrained model 
            model_fusion = TFLxmertModel.from_pretrained(lxmert_no_pos, cache_dir = CACHE_DIR)
            # Or I just load a randomly initialized model, with the size of the visual_feat being model_img.output.shape
        else:
            from transformers import LxmertConfig
            #config_lxm = LxmertConfig.from_json_file(lxmert_no_pos+'/config_size_output_efficient.json')
            #model_fusion = TFLxmertModel(config_lxm)
            #model_fusion.load_weights(lxmert_no_pos+'/tf_model.h5', by_name=True, skip_mismatch=True)
            model_fusion = TFLxmertModel.from_pretrained('unc-nlp/lxmert-base-uncased', cache_dir = CACHE_DIR)
            model_fusion.lxmert.encoder.visn_fc.use_visual_pos = False
            model_fusion.lxmert.use_visual_pos = False
            
            img_emb = tf.keras.layers.Dropout(0.1)(img_emb)
            img_emb = tf.keras.layers.Dense(2048, activation='tanh', name='dense_img_reshape')(img_emb)
            img_emb = tf.expand_dims(img_emb, axis=1)
            # print(img_emb.shape)
        
        # dict_input_lxmert = {'input_ids' : None, 'visual_feats' : model_img.output, 'inputs_embeds' : x}
        # dict_input_lxmert = {'visual_feats' : model_img.output, 'inputs_embeds' : x}
        dict_input_lxmert = {'input_ids' : input_word_ids, 'visual_feats' : img_emb} # test pour voir si inputs_embeds merde ou alors visual_feats
        out_fused = model_fusion(dict_input_lxmert, return_dict=True)
        cls_emb = out_fused.pooled_output

        ######## output layers ####### --> Attention j'ai mis une tanh ici a la place d'une sigmoid (ca parait plus coherent avec OUTPUT_NETWORK=7)
        hidden_out_size = 512
        cls_emb = tf.keras.layers.Dropout(0.1)(cls_emb)
        cls_emb = tf.keras.layers.Dense(hidden_out_size, activation='tanh', name='dense_MM')(cls_emb)
        cls_emb = tf.keras.layers.Dropout(0.1)(cls_emb)
        out = tf.keras.layers.Dense(nb_classes, name='dense_output_MM')(cls_emb)
        # if the output looks like a logits so, not bounded and not like probability distribution
        # (i.e. from_logits if no sigmoid activation at the last layer???)
        from_logits = True 
        
    if not bool_multimodal: 
        if 't5' not in model_name:
            model = tf.keras.models.Model(
                inputs=[input_word_ids, input_mask, segment_ids], 
                outputs=out     
            )
        else: # modele t5, no sep since its seq2seq
            model = tf.keras.models.Model(
                inputs={'input_word_ids': input_word_ids, 'input_mask': input_mask, 'decoder_input_ids': decoder_input_ids}, 
                outputs=out     
            )      
    else:

        model = tf.keras.models.Model(
            inputs=[input_word_ids, input_mask, segment_ids, input_img], 
            outputs=out     
        )
       
    # Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule
    # TODO: Are the values of epsilon and clipnorm importants ?? 
    # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08, clipnorm=1.0)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if nb_classes > 1: # if classification not ordinal 
        if multi_labels:
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)
            # loss = tf.keras.losses.MeanSquaredError() # aussi possible
            # metric = tf.keras.metrics.CategoricalAccuracy('acc')
            # metric = tf.keras.metrics.BinaryCrossentropy('binary_crossentropy', from_logits=from_logits)
            metric = tf.keras.metrics.BinaryAccuracy('acc', threshold = 0)
            # metric = tf.keras.metrics.AUC(name='auc', num_thresholds=200, curve='ROC', summation_method='interpolation', thresholds=None, multi_label=True)

            # metric = tf.keras.metrics.MeanIoU(n_classes=2)
        else:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits) # not as mk defined it before, but should be better since less preproc of labels 
            metric = tf.keras.metrics.SparseCategoricalAccuracy('acc')

        if OUTPUT_NETWORK == 4: # ce qu'avait fait MK de base 
            loss='binary_crossentropy'
            metric = 'accuracy'
            SPARSE_CATEGORICAL = False
    else: # if regression or ordinal classification trained as regression
        loss = tf.keras.losses.MeanSquaredError()
        metric = tf.keras.metrics.RootMeanSquaredError('rmse')
        if task_type == 'ordinal-classification':
            # TODO: create the class OrdinalCategoricalAccuracy for the ordinal-classification
            print('TODO: create the class OrdinalCategoricalAccuracy for the ordinal-classification')
            # raise ValueError('')

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    if verbose:
        model.summary(line_length=150)
    
    return model, tokenizer