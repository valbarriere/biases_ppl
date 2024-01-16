import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support, classification_report
import tensorflow as tf

class ReturnBestEarlyStopping(tf.keras.callbacks.EarlyStopping):
    def __init__(self, **kwargs):
        super(ReturnBestEarlyStopping, self).__init__(**kwargs)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            if self.verbose > 0:
                print(f'\nEpoch {self.stopped_epoch + 1}: early stopping')
        elif self.restore_best_weights:
            if self.verbose > 0:
                print('Restoring model weights from the end of the best epoch.')
            self.model.set_weights(self.best_weights)   

def create_y_multilabels(predictions, threshold_multi_labels, nb_lab=None):
	"""
	If multi labels then use a threshold to select
	nb_lab is the number of labels to select 
	"""
	y_classes = predictions > threshold_multi_labels
	# nb_lab = 1
	for idx_y, (y_class, labels_sorted) in enumerate(zip(y_classes, predictions.argsort(axis=-1))):
		# taking the nb_lab most probable labels IFF they are abose the threshold
		if nb_lab: y_classes[idx_y, labels_sorted[:-nb_lab]] = False
		# in order to take care of the case where no prediction is > 0 
		y_classes[idx_y, labels_sorted[-1]] = True	
		
	return y_classes

class Metrics_F1(tf.keras.callbacks.Callback):
	"""
	Homemade class to obtain other metrics than Accuracy on the validation data
	This is not useful on the train data since F1 is meaningul on the whole dataset and it's too costly to compute it on train
	Only used during the training. 
	I changed the tensorflow.python.keras.engine.training_v2._process_training_inputs() function to get the validation data with this : 
if validation_in_fit:
	# if Metrics_F1 in callbacks, it has avg_type as attr, then gives its index
	idx_F1 = [i for i, x in enumerate([hasattr(c, 'avg_type') for c in callbacks]) if x]
	if len(idx_F1) == 1: 
		idx_mycallback = idx_F1[0]
		if not hasattr(callbacks[idx_mycallback], 'validation_data'):
			callbacks[idx_mycallback].validation_data = (inputs, targets)
			print('Valentin: Using validation data') # happens only when you don't specify validation_data 
	"""
	
	def __init__(self, dict_lab, validation_data=None, avg_type='macro', output_fn=None, multi_labels=False, threshold_multi_labels=0, remove_labels=[], use_tf_dataset=False):
		super().__init__()
		
		# A remplacer par un elif type(validation_data) is tf.Dataset: ou qqchose comme ca...
		# self.use_tf_dataset = use_tf_dataset
		self.use_tf_dataset = hasattr(validation_data, 'element_spec')
		
		if type(validation_data) is tuple:
			self.validation_data = validation_data
			# TODO: sparse_categorical a verifier
			self.sparse_categorical = True if len(validation_data[1].shape) == 1 else False
			self.y_true = validation_data[1]
			
		elif self.use_tf_dataset:
			self.validation_data = validation_data
			self.sparse_categorical = True if len(validation_data.element_spec[-1].shape) == 1 else False # == 2 is categorical labels 
			self.y_true = np.concatenate([x for x in validation_data.map(lambda feats, lab: lab)], axis=0)
			
		if remove_labels:
			for remove_label in remove_labels:
				del dict_lab[remove_label]	
				
		self.dict_lab = dict_lab
				
		self.labels = list(self.dict_lab.values())
		
		self.avg_type = avg_type
		
		# file name to output the metrics 
		self.output_fn = output_fn
		
		self.multi_labels = multi_labels
		# proba threshold regarding whether the model predict or not the class (0 by default since output comes from logits (-inf to +inf))
		self.threshold_multi_labels = threshold_multi_labels
		
	def on_train_begin(self, logs={}):
		self.val_f1s = []
		self.val_recalls = []
		self.val_precisions = []
		
	def _sparse_categorical(self):
		"""
		In the case there is no validation data, but we split the train
		"""
		if self.validation_data[1].shape[1] == 1:
			self.sparse_categorical = True
			
			#self.validation_data[1] = self.validation_data[1].flatten()
			self.validation_data = (self.validation_data[0], np.array([x[0] for x in self.validation_data[1]]))
		
		
	def transform_x_to_labels(self):
		"""
		Ongoing work... never called right now
		TODO: change this one, to adapt to cumsum_labels
		"""
		if self.sparse_categorical:
			y_predict = self.model.predict(self.validation_data[0]).argmax(axis=-1)
			y_true = self.validation_data[1]
		else: # TODO: test the time. Use model.set_params({'validation_data' : dev_data}) outside then self.model.validation_data here 
			# to see if the dataset is in the gpu memory so that it is faster
			y_predict = self.model.predict(self.validation_data[0]).argmax(axis=-1)
			y_true = self.validation_data[1].argmax(axis=-1)
		
		return y_true, y_predict
	
	def on_test_end(self, epoch, logs={}): 
		"""
		The 'test' is also for validation when used during training 
		
		use_tf_dataset ne marche que pour sparse_categorical pour l'instant 
		"""
		if not hasattr(self, 'sparse_categorical'): self._sparse_categorical()
		
		if self.sparse_categorical:
			# val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round() # No
			if not self.use_tf_dataset:
				y_predict = self.model.predict(self.validation_data[0]).argmax(axis=-1)
				y_true = self.y_true
			else:
				y_predict = self.model.predict(self.validation_data).argmax(axis=-1)
				y_true = self.y_true
		else: # TODO: test the time. Use model.set_params({'validation_data' : dev_data}) outside then self.model.validation_data here 
			# to see if the dataset is in the gpu memory so that it is faster
			
			if self.use_tf_dataset:
				raise ValueError('Need to code this feature: not sparse_categorical and use_tf_dataset')
				
			y_predict = self.model.predict(self.validation_data[0])
			y_true = self.y_true
			# if multi labels then use a threshold to select 
			if self.multi_labels:
				y_classes = create_y_multilabels(y_predict, self.threshold_multi_labels, nb_lab=None)
				y_predict = y_classes
			else:
				# if single label
				y_predict = y_predict.argmax(axis=-1)
				y_true = y_true.argmax(axis=-1)
		# print(y_true[:3], y_predict[:3], self.sparse_categorical)
		_val_f1_all = f1_score(y_true, y_predict, average=None, labels = self.labels)
		_val_recall_all = recall_score(y_true, y_predict, average=None, labels = self.labels)
		_val_precision_all = precision_score(y_true, y_predict, average=None, labels = self.labels)
		self.val_f1s.append(_val_f1_all)
		self.val_recalls.append(_val_recall_all)
		self.val_precisions.append(_val_precision_all)
		
		f_score_avg = 100*f1_score(y_true, y_predict, average=self.avg_type, labels = self.labels)
		prec_score_avg = 100*precision_score(y_true, y_predict, average=self.avg_type, labels = self.labels)
		rec_score_avg = 100*recall_score(y_true, y_predict, average=self.avg_type, labels = self.labels)
		
		str_to_print = "\nTest on the Dev Set: Accuracy {:.2f}".format(100*np.mean(y_true == y_predict))
		str_to_print += "\nClass '{}' — val_f1: {:.2f} — val_precision: {:.2f} — val_recall {:.2f}".format(self.avg_type, f_score_avg, prec_score_avg, rec_score_avg)
		
		for num_lab, lab in enumerate(self.dict_lab.keys()):
			str_to_print += "\nClass '{}' — val_f1: {:.2f} — val_precision: {:.2f} — val_recall {:.2f}".format(lab, 100*_val_f1_all[num_lab], 
							100*_val_precision_all[num_lab], 100*_val_recall_all[num_lab])
		
		print(str_to_print+'\n')
		
		# Printing that into a txt file
		# TODO: integrate this into TBoard... 
		if self.output_fn:
			with open(self.output_fn, 'a') as f:
				f.write(str_to_print+'\n')
			
		return 


def metrics_for_regression(y_reg, y_gold_reg):
	
	# TODO: test with normal classification
	# prediction or classes
	# y_pred = y_classes
	# always classes for V-oc
	# y_gold_pred = y_gold_classes
	from sklearn.metrics import cohen_kappa_score, r2_score, mean_squared_error
	from scipy.stats import pearsonr
	
	# TODO: test if it's working
	pearson, _ = pearsonr(y_gold_reg, y_reg)
	
	# if it's a prediction in a singleclass configuration, then you need to transform the value into an ordinal class 
	# works only for integer values and not continuous values
	if type(y_gold_reg[0]) == int:
		kappa = cohen_kappa_score(y_gold_reg, y_reg)
	else:
		kappa=0
	
	r2 = r2_score(y_gold_reg, y_reg)
	mse = mean_squared_error(y_gold_reg, y_reg)
	
	#str_acc += ', Pearson corr: {:.2f}, Kappa: {:.2f}'.format(pearson, kappa)
	return pearson, kappa, r2, mse

def calculate_and_print_metrics(y_gold_classes, y_classes, y_reg, dict_lab, path_dump_pred, str_id_training_and_testing, 
								task_type, ordinal_data, dict_args, save_pred = False, input_data_File = ''):
	"""
	If regression, can still calculate the F1 over classes if it's a regression over ordinal classes
	y_gold_classes if just y_gold, it's not necessarily classes 
	"""
	
	str_acc = "{}".format(input_data_File)
	str_classification_report = ''
	dict_args['test_set'] = input_data_File
	
	# all but pure regression : classification, ordinal classification trained as classif, ordinal classification trained as regression
	if 'classification' in task_type: 
		# TODO: validate that labels = dict_lab.values()
		# prec, rec, f1, samp = precision_recall_fscore_support(y_gold_classes, y_classes, average=None, labels = range(np.max(y_gold_classes)+1))
		prec, rec, f1, samp = precision_recall_fscore_support(y_gold_classes, y_classes, average=None, labels = list(dict_lab.values()))
		for lab, idx_lab in dict_lab.items():
			dict_args['Precision-%s'%lab] = prec[idx_lab]
			dict_args['Recall-%s'%lab] = rec[idx_lab]
			dict_args['F1-%s'%lab] = f1[idx_lab]
			dict_args['n_samples-%s'%lab] = samp[idx_lab] # useful to calculate weighted metrics 
		
		avgrec = np.mean(rec)
		dict_args['Recall-macro'] = avgrec
		f1mac = np.mean(f1)
		dict_args['F1-macro'] = f1mac
		f1w = np.sum(f1*samp/np.sum(samp))
		dict_args['F1-weighted'] = f1w
		acc = np.mean(y_classes==y_gold_classes)
		dict_args['Accuracy'] = acc
		
		
		str_acc += ", Accuracy: {:.2f}, Macro-F1: {:.2f}, weighted-F1: {:.2f}".format(
																100*acc,
															   100*f1mac,
																100*f1w)
		# TODO: validate that labels = dict_lab.values()
		#str_classification_report = classification_report(y_gold_classes, y_classes, target_names=list(dict_lab.keys()), 
														 # digits=3, labels = range(np.max(y_gold_classes)+1))
		
		list_labels = list(dict_lab.values())
		target_names = list(dict_lab.keys())
		# if we dont care about neutral --> Dailydialog
		# if "remove_labels" in dict_args.keys():
		for remove_label_name in dict_args['remove_labels']:
			list_labels.remove(dict_lab[remove_label_name])
			target_names.remove(remove_label_name)
			
		# remove the labels not in the test set 
		# if multi-labels
		if len(y_gold_classes.shape) > 1:
			# then it's a sparse vector of labels like [0,0,0,1]
			# find the all indexes where it's equal to one 
			y_labels= np.unique(np.where(y_gold_classes == 1)[1])
		else:
			y_labels = np.unique(y_gold_classes)
		labels_not_in_test_set = [k for k in list_labels if k not in y_labels]
		if len(labels_not_in_test_set):
			print('Remove the %d labels from train set that are not in the test set:'%len(labels_not_in_test_set),labels_not_in_test_set[:10], 'and more...'*(len(labels_not_in_test_set) > 10))
			inv_dict_lab={v:k for k,v in dict_lab.items()}
			for remove_label_idx in labels_not_in_test_set:
				list_labels.remove(remove_label_idx)
				target_names.remove(inv_dict_lab[remove_label_idx])	
				
		str_classification_report = classification_report(y_gold_classes, y_classes, target_names=target_names, 
														 digits=4, labels=list_labels)
		if 'F1-positive' in dict_args.keys():
			if 'F1-negative' in dict_args.keys():
				str_classification_report += '\nLatex (Avg_rec, F1mac, F1PN): %.1f  &   %.1f  &   %.1f\n\n'%(100*dict_args['Recall-macro'], 
																									 100*dict_args['F1-macro'], 
																									 50*(dict_args['F1-positive']+dict_args['F1-negative']))
		
		# if not multilabels (or not sparse)
		if len(y_gold_classes.shape) <= 1:
			print('Creating confusion matrix...')
			df_y_gold_classes = pd.Series(y_gold_classes, name='Actual')
			df_y_classes = pd.Series(y_classes, name='Predicted')
			inv_dict_lab={v:k for k,v in dict_lab.items()}

			df_confusion = pd.crosstab(df_y_gold_classes.map(inv_dict_lab), df_y_classes.map(inv_dict_lab))
			## save
			if save_pred: df_confusion.to_html(path_dump_pred + "confmtrx" + str_id_training_and_testing + ".html")
			else: print(df_confusion)
				
	# regression or ordinal classification, we can also have ordinal classification when training using pure classification
	if ordinal_data:
		# if not None
		if type(y_classes) is np.ndarray:
			pearson, kappa, r2, mse = metrics_for_regression(y_classes, y_gold_classes)
			str_acc += ', with classes: Pearson corr: {:.2f}, Kappa: {:.2f}, R2: {:.2f}, RMSE: {:.2f}'.format(pearson, kappa, r2, np.sqrt(mse))
			dict_args['Pearson-classes'] = pearson
			dict_args['kappa-classes'] = kappa
		# if not None
		if type(y_reg) is np.ndarray:
			pearson, kappa, r2, mse = metrics_for_regression(y_reg, y_gold_classes)
			str_acc += ', with classes: Pearson corr: {:.2f}, Kappa: {:.2f}, R2: {:.2f}, RMSE: {:.2f}'.format(pearson, kappa, r2, np.sqrt(mse))
			dict_args['Pearson-reg'] = pearson
			dict_args['kappa-reg'] = kappa
			
	print(str_acc)
	print(str_classification_report)
	
	inv_dict_lab = {v:k for k,v in dict_lab.items()}
	
	#save predicted classes
	# if exists
	if (type(y_classes) is np.ndarray) and save_pred:
		if len(y_classes.shape) == 1:
			y_classes_dump = np.array([inv_dict_lab[k] for k in y_classes])
			np.save(path_dump_pred + 'pred' + str_id_training_and_testing + '.npy', y_classes_dump)
	else:
		print('not dumping predictions y_classes')
	# if exists
	if (type(y_reg) is np.ndarray) and save_pred:
		if len(y_reg.shape) == 1:
			y_reg_dump = np.array([inv_dict_lab[k] for k in y_reg])
			np.save(path_dump_pred + 'predreg' + str_id_training_and_testing + '.npy', y_reg_dump)	
	else:
		print('not dumping predictions y_reg')
		
	return str_acc + '\n' + str_classification_report + '\n', dict_args
