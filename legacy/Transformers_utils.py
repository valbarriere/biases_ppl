import pandas as pd 
import os
import argparse

# INIT_PATH = '/home/emmproc/'
INIT_PATH = '/eos/jeodpp/data/projects/REFOCUS/'

def read_csv_val(fn, encoding = 'utf-8'):
	return pd.read_csv(fn, sep='\t', quotechar='"', encoding=encoding, header=0)

def to_csv_val(df, fn, encoding = 'utf-8'):
	print('file "{}" saved'.format(fn))
	return df.to_csv(fn, sep='\t', quotechar='"', encoding=encoding, index=False)

def str2bool(v):
	"""
	Function to hangle boolean in an argparse parameter that is postive by default
	"""
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')


def reduce_size_str_training_id(str_training_id):
	"""
	Sometimes the string is too long...
	"""
	# in order to reduce the file name size and avoid protocol errors 
	str_training_id = str_training_id.replace('sb-10k_train_val.tsv_multilingual-AND-Sentipolc16-trainval_general.csv_val_multilingual-AND-intertass2018-ES-train-tagged.tsv_val_multilingual-AND-TASS2019_country_ES_train.tsv_val_multilingual-AND-DEFT2015-task1-trainval_val.tsv_multilingual-AND-SemEval2018-task1-Valence-c-En-train.txt_val_multilingual_multilingual_devval.tsv', 'all-real-datasets-multilingual')

	str_training_id = str_training_id.replace('SemEval2013-task2-subtaskB-train.tsv_val-AND-SemEval2013-task2-subtaskB-test.tsv_val-AND-SemEval2017-task4-dev.subtask-A.english.INPUT.txt_val-AND-twitter-2016train-A.txt_val-AND-twitter-2016dev-A.txt_val-AND-twitter-2014test-A.txt_val-AND-twitter-2015test-A.txt_val', 'datasets-config2-english')	
	
	str_training_id = str_training_id.replace('SemEval2013-task2-subtaskB-train.tsv_val_multilingual-AND-SemEval2013-task2-subtaskB-test.tsv_val_multilingual-AND-SemEval2017-task4-dev.subtask-A.english.INPUT.txt_val_multilingual-AND-twitter-2016train-A.txt_val_multilingual-AND-twitter-2016dev-A.txt_val_multilingual-AND-twitter-2014test-A.txt_val_multilingual-AND-twitter-2015test-A.txt_val_multilingual', 'datasets-config2-multilingual') 
	
	# on rajoute SemEval2013-task2-subtaskB-dev.tsv_val
	str_training_id = str_training_id.replace('SemEval2013-task2-subtaskB-train.tsv_val-AND-SemEval2013-task2-subtaskB-test.tsv_val-AND-SemEval2013-task2-subtaskB-dev.tsv_val-AND-SemEval2017-task4-dev.subtask-A.english.INPUT.txt_val-AND-twitter-2016train-A.txt_val-AND-twitter-2016dev-A.txt_val-AND-twitter-2014test-A.txt_val-AND-twitter-2015test-A.txt_val', 'datasets-config2-english')	
	
	str_training_id = str_training_id.replace('SemEval2013-task2-subtaskB-train.tsv_val_multilingual-AND-SemEval2013-task2-subtaskB-test.tsv_val_multilingual-AND-SemEval2013-task2-subtaskB-dev.tsv_val_multilingual-AND-SemEval2017-task4-dev.subtask-A.english.INPUT.txt_val_multilingual-AND-twitter-2016train-A.txt_val_multilingual-AND-twitter-2016dev-A.txt_val_multilingual-AND-twitter-2014test-A.txt_val_multilingual-AND-twitter-2015test-A.txt_val_multilingual', 'datasets-config2-multilingual') 
	
	str_training_id = str_training_id.replace('SemEval2013-task2-subtaskB-test.tsv_val_multilingual-27-AND-SemEval2013-task2-subtaskB-train.tsv_val_multilingual-27-AND-SemEval2017-task4-test.subtask-A.english.txt_val_multilingual-27-AND-SemEval2017-task4-dev.subtask-A.english.INPUT.txt_val_multilingual-27-AND-Eurotweets_Hungarian_val_without_line_return.tsv_clean_train-AND-Eurotweets_Polish_val_without_line_return.tsv_clean_train-AND-Eurotweets_Russian_val_without_line_return.tsv_clean_train-AND-Eurotweets_Slovak_val_without_line_return.tsv_clean_train-AND-Eurotweets_Slovenian_val_without_line_return.tsv_clean_train-AND-Eurotweets_Swedish_val_without_line_return.tsv_clean_train-AND-DEFT2015-task1-train_val.tsv-AND-intertass2018-ES-all-tagged.tsv_val-AND-TASS2019_country_ES_train.tsv_val-AND-Sentipolc16-train_general.csv_val-AND-sb-10k_train_val.tsv', 'model_for_prod_26_MTSEM_and_11RAW') 
	
	str_training_id = str_training_id.replace('SemEval2013-task2-subtaskB-test.tsv_val_multilingual-27-AND-SemEval2013-task2-subtaskB-train.tsv_val_multilingual-27-AND-SemEval2017-task4-test.subtask-A.english.txt_val_multilingual-27-AND-SemEval2017-task4-dev.subtask-A.english.INPUT.txt_val_multilingual-27-AND-Eurotweets_Hungarian_val_without_line_return.tsv_clean_train-AND-Eurotweets_Polish_val_without_line_return.tsv_clean_train-AND-Eurotweets_Russian_val_without_line_return.tsv_clean_train-AND-Eurotweets_Slovak_val_without_line_return.tsv_clean_train-AND-Eurotweets_Swedish_val_without_line_return.tsv_clean_train-AND-DEFT2015-task1-train_val.tsv-AND-intertass2018-ES-all-tagged.tsv_val-AND-TASS2019_country_ES_train.tsv_val-AND-Sentipolc16-train_general.csv_val-AND-sb-10k_train_val.tsv', 'model_for_prod_26_MTSEM_and_10RAW') 
	
	str_training_id = str_training_id.replace('SemEval2013-task2-subtaskB-test.tsv_val-AND-SemEval2013-task2-subtaskB-train.tsv_val-AND-SemEval2017-task4-test.subtask-A.english.txt_val-AND-SemEval2017-task4-dev.subtask-A.english.INPUT.txt_val-AND-Eurotweets_Hungarian_val_without_line_return.tsv_clean_train-AND-Eurotweets_Polish_val_without_line_return.tsv_clean_train-AND-Eurotweets_Russian_val_without_line_return.tsv_clean_train-AND-Eurotweets_Slovak_val_without_line_return.tsv_clean_train-AND-Eurotweets_Swedish_val_without_line_return.tsv_clean_train-AND-DEFT2015-task1-train_val.tsv-AND-intertass2018-ES-all-tagged.tsv_val-AND-TASS2019_country_ES_train.tsv_val-AND-Sentipolc16-train_general.csv_val-AND-sb-10k_train_val.tsv', 'model_for_prod_0_MTSEM_and_10RAW') 
	
	str_training_id = str_training_id.replace('SemEval2013-task2-subtaskB-test.tsv_val-AND-SemEval2013-task2-subtaskB-train.tsv_val-AND-SemEval2017-task4-test.subtask-A.english.txt_val-AND-SemEval2017-task4-dev.subtask-A.english.INPUT.txt_val-AND-Eurotweets_Hungarian_val_without_line_return.tsv_clean_train-AND-Eurotweets_Polish_val_without_line_return.tsv_clean_train-AND-Eurotweets_Russian_val_without_line_return.tsv_clean_train-AND-Eurotweets_Slovak_val_without_line_return.tsv_clean_train-AND-Eurotweets_Swedish_val_without_line_return.tsv_clean_train-AND-DEFT2015-task1-trainval_val.tsv-AND-intertass2018-ES-all-tagged.tsv_val-AND-TASS2019_country_ES_train.tsv_val-AND-Sentipolc16-trainval_general.csv_val-AND-sb-10k_train_val.tsv', 'model_for_prod_0_MTSEM_and_10RAW_trainvalDEFTandSPolc') 
	
	str_training_id = str_training_id.replace('SemEval2013-task2-subtaskB-test.tsv_val-AND-SemEval2013-task2-subtaskB-train.tsv_val-AND-SemEval2017-task4-test.subtask-A.english.txt_val-AND-SemEval2017-task4-dev.subtask-A.english.INPUT.txt_val-AND-Eurotweets_Hungarian_val_without_line_return.tsv_clean_train-AND-Eurotweets_Polish_val_without_line_return.tsv_clean_train-AND-Eurotweets_Russian_val_without_line_return.tsv_clean_train-AND-Eurotweets_Slovak_val_without_line_return.tsv_clean_train-AND-Eurotweets_Slovenian_val_without_line_return.tsv_clean_train-AND-Eurotweets_Swedish_val_without_line_return.tsv_clean', 'ZS_COMPARISON_EN_and_10RAW') 
	

	print('...Changed str_training_id to %s'%str_training_id)
	return str_training_id

def return_filename_of_pretrained_model(pre_training_text):
	"""
	Return the filename of the pretraiend model, this is special to my scripts (and super messy)
	"""
	fn = None
	path_ini = INIT_PATH + 'data/Valentin/tweets_alex/dump/'
	
	if pre_training_text:
		# for the multilingual tweets; dev is tweeti-b.dev.dist.Tweets.tsv.utf8.multilingual_clean_val
		if pre_training_text == 'crisis-benchmark-mling':
			fn='fine-tuned_xlm-roberta-base_crisis_consolidated_humanitarian_filtered_lang_train_few_labels.tsv_val_crisis_consolidated_humanitarian_filtered_lang_en_dev.tsv_val_all-epoch_1e-06_output-type7-best-model-checkpoint-val_acc.hdf5'
			path_ini = INIT_PATH + 'data/Valentin/Crisis_Benchmark/dump/'
		elif pre_training_text == 'data-augmentation2tweeti':
			fn='fine-tuned_xlm-roberta-base_datasets-config2-multilingual_tweeti-b.dev.dist.Tweets.tsv.utf8.multilingual_clean_val_all-epoch_1e-06_CW_output-type7-best-model-checkpoint-val_acc.hdf5'
		elif pre_training_text == 'english2tweeti':
			fn='fine-tuned_xlm-roberta-base_datasets-config2-english_tweeti-b.dev.dist.Tweets.tsv.utf8.multilingual_clean_val_all-epoch_1e-06_CW_output-type7-best-model-checkpoint-val_acc.hdf5'
		elif pre_training_text == 'data-augmentation2real':
			fn='fine-tuned_xlm-roberta-base_datasets-config2-multilingual_multilingual_devval.tsv_all-epoch_1e-06_CW_output-type7-best-model-checkpoint-val_acc.hdf5'
		elif pre_training_text == 'english2real':
			fn='fine-tuned_xlm-roberta-base_datasets-config2-english_multilingual_devval.tsv_all-epoch_1e-06_CW_output-type7-best-model-checkpoint-val_acc.hdf5'
		elif pre_training_text == 'english-roberta':
			fn='fine-tuned_roberta-base_datasets-config2-english_twitter-2016devtest-A.txt_val_all-epoch_5e-07_CW_already-ft-on-tweets-continue-training_output-type7-best-model-checkpoint-val_acc.hdf5'
		elif pre_training_text == 'english2conll':
			fn='fine-tuned_xlm-roberta-base_datasets-config2-english_multilingual_devval_conll.tsv_all-epoch_1e-06_CW_output-type7-best-model-checkpoint-val_acc.hdf5'
		elif pre_training_text == 'english2english':
			fn='fine-tuned_xlm-roberta-base_datasets-config2-english_twitter-2016devtest-A.txt_val_all-epoch_5e-07_CW_output-type7-best-model-checkpoint-val_acc.hdf5'
		elif 'data-augmentation_ET_devval' in pre_training_text: # for multilingual + Estonian
			fn='fine-tuned_xlm-roberta-base_SemEval2013-task2-subtaskB-train.tsv_val_multilingual_ET-AND-SemEval2013-task2-subtaskB-test.tsv_val_multilingual_ET-AND-SemEval2017-task4-dev.subtask-A.english.INPUT.txt_val_multilingual_ET-AND-SemEval2017-task4-test.subtask-A.english.txt_val_multilingual_ET_multilingual_devval_conll.tsv_all_all-epoch_1e-06_CW_output-type7-best-model-checkpoint-val_acc.hdf5'
		elif 'data-augmentation_ET_tweeti' in pre_training_text: # for multilingual + Estonian
			fn='fine-tuned_xlm-roberta-base_SemEval2013-task2-subtaskB-train.tsv_val_multilingual_ET-AND-SemEval2013-task2-subtaskB-test.tsv_val_multilingual_ET-AND-SemEval2017-task4-dev.subtask-A.english.INPUT.txt_val_multilingual_ET-AND-SemEval2017-task4-test.subtask-A.english.txt_val_multilingual_ET_tweeti-b.dev.dist.Tweets.tsv.utf8.multilingual_clean_val_all-epoch_1e-06_CW_output-type7-best-model-checkpoint-val_acc.hdf5'
		elif 'data-augmentation' in pre_training_text:
			fn='fine-tuned_xlm-roberta-base_SemEval2017-task4-dev.subtask-A.english.INPUT.txt_val_multilingual-AND-SemEval2017-task4-test.subtask-A.english.txt_val_multilingual-AND-SemEval2013-task2-subtaskB-train.tsv_val_multilingual-AND-SemEval2013-task2-subtaskB-test.tsv_val_multilingual_tweeti-b.dev.dist.Tweets.tsv.utf8.multilingual_clean_val_all-epoch_1e-06_CW_output-type7-best-model-checkpoint-val_acc.hdf5'
		elif 'english' in pre_training_text:
			fn='fine-tuned_xlm-roberta-base_SemEval2017-task4-dev.subtask-A.english.INPUT.txt_val-AND-SemEval2017-task4-test.subtask-A.english.txt_val-AND-SemEval2013-task2-subtaskB-train.tsv_val-AND-SemEval2013-task2-subtaskB-test.tsv_val_tweeti-b.dev.dist.Tweets.tsv.utf8.multilingual_clean_val_all-epoch_1e-06_CW_output-type7-best-model-checkpoint-val_acc.hdf5'
		elif pre_training_text == 'continue-training': # use a model with the same configuration, but that should be already trained
			fn = os.path.basename(trained_modelFilePath[:-5]+'-best-model-checkpoint-{}.hdf5'.format(monitor_value)).replace('_already-ft-on-tweets-continue-training', '')
		elif pre_training_text == 'XStance-XLM': # use a model with the same configuration, but that should be already trained
			path_ini = INIT_PATH + 'data/Valentin/Debating_Europe/dump/'
			fn = 'fine-tuned_xlm-roberta-base_XStance_train.tsv_XStance_validation.tsv_all-epoch_1e-06_CW_output-type7-best-model-checkpoint-val_acc.hdf5'
		elif pre_training_text == "Debating-QA":
			path_ini = INIT_PATH + 'data/Valentin/Debating_Europe/dump/'
			fn = 'fine-tuned_xlm-roberta-base_stance_level_QA_no_context_3_labels.tsv_train_stance_level_QA_no_context_3_labels.tsv_dev_all-epoch_1e-05_CW_output-type7-best-model-checkpoint-val_acc.hdf5'
		elif pre_training_text == "Xstance-XLM-Debating-QA":
			path_ini = INIT_PATH + 'data/Valentin/Debating_Europe/dump/'
			fn = 'fine-tuned_xlm-roberta-base_stance_level_QA_no_context_3_labels.tsv_train_stance_level_QA_no_context_3_labels.tsv_dev_all-epoch_1e-06_CW_already-ft-on-text-XStance-XLM_output-type7-best-model-checkpoint-val_acc.hdf5'		
		elif pre_training_text == 'Debating-QA-XStance-XLM': # normaly tf_stance_ternary???
			path_ini = INIT_PATH + 'data/Valentin/Debating_Europe/dump/'
			fn = 'fine-tuned_xlm-roberta-base_XStance_train.tsv_XStance_validation.tsv_all-epoch_1e-06_CW_output-type7-best-model-checkpoint-val_acc.hdf5'
		elif pre_training_text == 'XStance-CFS-XLM': # normaly tf_stance_ternary???
			path_ini = INIT_PATH + 'data/Valentin/Debating_Europe/dump/'
			fn = 'fine-tuned_xlm-roberta-base_XStance_train.tsv-AND-scrapped_CoFE_nohandlabels_autolabels.tsv_traintest_XS_CFS_validation.tsv_all-epoch_1e-06_CW_output-type7-best-model-checkpoint-val_acc.hdf5'
		elif pre_training_text == 'XStance-CFS-Xdebates-XLM': # normaly tf_stance_ternary???
			path_ini = INIT_PATH + 'data/Valentin/Debating_Europe/dump/'
			fn = 'fine-tuned_xlm-roberta-base_XStance_train.tsv-AND-scrapped_CoFE_nohandlabels_autolabels_Xdebates.tsv_traintest_XS_CFS_Xdebates_validation.tsv_all-epoch_1e-06_CW_output-type7-best-model-checkpoint-val_acc.hdf5'
		elif pre_training_text == 'XStance-CFS-XLM_MDPI': # normaly tf_stance_ternary???
			path_ini = INIT_PATH + 'data/Valentin/Debating_Europe/dump/'
			fn = 'fine-tuned_xlm-roberta-base_XStance_train.tsv-AND-CFS_classifier.tsv_traintest_XS_CFS_validation_MDPI.tsv_all-epoch_1e-06_CW_output-type7-best-model-checkpoint-val_acc.hdf5'
		elif pre_training_text == 'XStance-CFS-Xdebates-XLM_MDPI': # normaly tf_stance_ternary???
			path_ini = INIT_PATH + 'data/Valentin/Debating_Europe/dump/'
			fn = 'fine-tuned_xlm-roberta-base_XStance_train.tsv-AND-CFS_classifier_Xdebates.tsv_traintest_XS_CFS_Xdebates_validation_MDPI.tsv_all-epoch_1e-06_CW_output-type7-best-model-checkpoint-val_acc.hdf5'
		elif pre_training_text == "TSA_tweets":
			path_ini = INIT_PATH + 'data/Valentin/IJCAI2019_data/dump/'
			fn = 'fine-tuned_roberta-base_targeted_twitter2015-train.tsv_val-AND-targeted_twitter2017-train.tsv_val_targeted_twitter2015-dev.tsv_val_all-epoch_1e-05_output-type7-best-model-checkpoint-val_acc.hdf5'		
		elif pre_training_text == "XLM_TSA_tweets":
			path_ini = INIT_PATH + 'data/Valentin/IJCAI2019_data/dump/'
			fn = 'fine-tuned_xlm-roberta-base_targeted_twitter2015-train.tsv_val-AND-targeted_twitter2017-train.tsv_val_targeted_twitter2015-dev.tsv_val_all-epoch_1e-05_output-type7-best-model-checkpoint-val_acc.hdf5'	
		elif pre_training_text == "XLM_TSA_tweets_transfo4":
			path_ini = INIT_PATH + 'data/Valentin/IJCAI2019_data/dump/'
			fn = 'fine-tuned_xlm-roberta-base_targeted_twitter2015-train.tsv_val-AND-targeted_twitter2017-train.tsv_val_targeted_twitter2015-dev.tsv_val_all-epoch_1e-05_output-type7_TEST-test_transfo4-best-model-checkpoint-val_acc.hdf5'
		elif pre_training_text == "tf-roberta-goemotions-6emotions":
			path_ini = INIT_PATH + 'data/Valentin/models/'
			fn = 'tf-roberta-goemotions-6emotions.h5'
		elif pre_training_text == "tf-roberta-goemotions2-6emotions":
			path_ini = INIT_PATH + 'data/Valentin/models/'
			fn = 'tf-roberta-goemotions2-6emotions.h5'
	else:
		path_ini = ''
		fn = ''
		
	return path_ini + fn

def return_dict_labels_for_training(bool_labels_not_from_first_train, inputTsvFile, str_corpus, else_class, multi_labels):
	"""
	"""
	if bool_labels_not_from_first_train:
		# TODO: change these hard-coded values
		if 'ACL' in str_corpus:
			if 'finegrained' in inputTsvFile:
				list_labels = ['ABDUCT_DISSAP', 'AGREEMENT', 'AIR_STRIKE', 'ARMED_CLASH', 'ARREST', 'ATTACK', 'CHANGE_TO_GROUP_ACT', 'CHEM_WEAP', 'DISR_WEAP', 'FORCE_AGAINST_PROTEST', 'GOV_REGAINS_TERIT', 'GRENADE', 'HQ_ESTABLISHED', 'PROPERTY_DISTRUCT', 'MOB_VIOL', 'NON_STATE_ACTOR_OVERTAKES_TER', 'NON_VIOL_TERRIT_TRANSFER', 'OTHER', 'PEACE_PROTEST', 'PROTEST_WITH_INTER', 'REM_EXPLOS', 'SEX_VIOL', 'ART_MISS_ATTACK', 'SUIC_BOMB', 'VIOL_DEMONSTR']
			else:   
				list_labels = ['PRO', 'RIO', 'BAT', 'EXP', 'SDEV', 'VAC']
		elif 'Valence-oc' in inputTsvFile: # TODO: and this one is the worst.... 
			list_labels = ['very negative', 'negative', 'slightly negative', 'positive', 'slightly positive', 'very positive']
		elif 'Emotion' in inputTsvFile: # TODO: and this one is the worst.... 
			list_labels = ['anger', 'disgust', 'fear', 'sad', 'surprise', 'joy']
		elif 'tweet' in str_corpus:
			list_labels = ['positive', 'negative']
		elif ('Crisis_Benchmark' in str_corpus):
			list_labels = ['affected_individual', 'caution_and_advice',
			   'displaced_and_evacuations', 'donation_and_volunteering',
			   'infrastructure_and_utilities_damage', 'injured_or_dead_people',
			   'missing_and_found_people', 'not_humanitarian',
			   'requests_or_needs', 'response_efforts', 'sympathy_and_support']
			else_class = False
		else:
			raise ValueError("Error putting dict_lab")
		# not useful if we take the labels of the first train
		if else_class:
			list_labels.append('else')
	# take the labels that are present in the first train file 
	else:
		df = read_csv_val(inputTsvFile)
		
		if not multi_labels:
			list_labels = list(df.label.unique())
			# in order to keep the same order, we sort by alphabetical order
			list_labels.sort()
		# Impossible to use df.label if multi-label ? 
		else:
			# TODO : Can be used for GoEMotions only... should change. multi_labels = path_to_labels?
			if 'GoEmotions' in inputTsvFile:
				# path_to_labels = '/home/emmproc/data/Valentin/GoEmotions/emotions.txt'
				path_to_labels = '/eos/jeodpp/data/projects/REFOCUS/' + 'data/Valentin/GoEmotions2/emotions.txt'
				
				dfe = pd.read_csv(path_to_labels, header = None)
				list_labels = dfe[0].values
			else:
				list_labels = []
				for labs in df.label.map(lambda x: x.split(',')):
					for lab in labs:
						if lab not in list_labels:
							list_labels.append(lab)
				list_labels.sort()
	 
	dict_lab = {lab: idx for idx, lab in enumerate(list_labels)}
	
	# On fout tout en lower... 
	dict_lab = {k.lower():v for k,v in dict_lab.items()}
		
	return dict_lab

# Regarding the platform we are using 
IS_COLAB = False # in colab, other paths 
IS_LOCAL = False # local
IS_JHUB = True

def not_XOR_function(A: bool, B: bool, C: bool):
	"""
	Return True if there is 0, 2, or 3 activations at the same time.  
	"""
	res = (((not A) or B or C) and 
	((not B) or A or C) and 
	((not C) or A or B)
		  )
	
	return res
	
def return_str_paths(IS_COLAB, IS_LOCAL, IS_JHUB, str_corpus):
	"""
	return the path of the corpus regarding the platform we use
	"""
	# XOR function, we need only one value  
	if not_XOR_function(IS_COLAB, IS_LOCAL, IS_JHUB):
		raise ValueError('What is the platform? Jhub, Colab, Local?')

	if IS_COLAB:
		if not os.path.isdir('/content/drive'): 
			from google.colab import drive
			drive.mount('/content/drive')

			path_corpus = '/content/drive/My Drive/DATASETS/'+str_corpus
			assert os.path.isdir(path_dump)

		if os.path.isdir('/content/drive'):
			path_corpus = '/content/drive/My Drive/DATASETS/'+str_corpus

		path_dump = path_corpus

	elif IS_LOCAL:
		INIT_PATH = '/Users/Valou/' 
		print("INIT_PATH: " + INIT_PATH)
		path_dump =  INIT_PATH + 'Google_Drive/DATASETS/'+str_corpus

	elif IS_JHUB:
		# INIT_PATH = '/home/emmproc/'
		INIT_PATH = '/eos/jeodpp/data/projects/REFOCUS/'
		print("INIT_PATH: " + INIT_PATH)
		path_corpus = INIT_PATH + 'data/Valentin/'+str_corpus
		path_dump = path_corpus + 'dump/'
	
	return path_corpus, path_dump
