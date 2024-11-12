"""
Script transforming the data into perturbed data with names typical from other countries, in order to test the bias of the model
Just return the examples with entities inside
Only work for english for now since it uses en_core_web_sm for the NER

Author: Anonymous_Submission 01/24
"""
import spacy
import itertools
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
from checklist.perturb import Perturb

class HFDoc():
	def __init__(self, text, ents):
		self.text = text
		self.ents = ents

class NER():
	def __init__(self, ner_type, ner_name):
		self.ner_type = ner_type
		if self.ner_type == "spacy":
			self.nlp = spacy.load(ner_name)
		elif self.ner_type == "hf":
			tokenizer = AutoTokenizer.from_pretrained(ner_name)
			model = AutoModelForTokenClassification.from_pretrained(ner_name)
			self.nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
		else:
			raise Exception(f"{ner_type} type not supported. Choose")
		
	def pipe(self, texts):
		if self.ner_type == "spacy":
			docs = list(self.nlp.pipe(texts))
		else: # self.ner_type == "hf"
			ner_entities = self.nlp(list(texts))
			docs = [HFDoc(text, ents) for text, ents in zip(texts, ner_entities)]
		return docs
	
class PerturbedExamples():
	def __init__(self, 
				 list_countries : list = ['Austria',
										 'Belgium',
										 'Czech_Republic',
										 'Denmark',
										 'United_Kingdom',
										 'Estonia',
										 'Finland',
										 'France',
										 'Germany',
										 'Hungary',
										 'Iceland',
										 'Ireland',
										 'Italy',
										 'Lithuania',
										 'Malta',
										 'the_Netherlands',
										 'Norway',
										 'Poland',
										 'Portugal',
										 'Slovenia',
										 'Spain',
										 'Sweden',
										 'Switzerland',
										 'Albania',
										 'Belarus',
										 'Bosnia_and_Herzegovina',
										 'Bulgaria',
										 'Croatia',
										 'Luxembourg',
										 'Serbia',
										 'Greece',
										 'Romania',
										 'Slovakia',
										 'Ukraine',
										 'Morocco', 
                                         'Russia',
                                         'Turkey',
                                         'India',
										 ], 
				 use_female : bool = True,
				 ner_type: str = "spacy",
				 ner_name: str = "xx_ent_wiki_sm"
				):
		self.list_countries = list_countries
		self.use_female = use_female
		self.ner = NER(ner_type, ner_name) 
		
	def all_countries(self, texts, labels, n=2):
		"""
		Create the new data in the form of a dictionnary
		"""
		
		list_transformed={country : {} for country in self.list_countries}
		list_ex_to_tag=[]
		list_ex_to_tag_ini = []

		pdata = self.ner.pipe(texts)

		for country, gender in itertools.product(self.list_countries, ['male'] + ['female']*self.use_female):
			ret = Perturb.perturb(pdata, Perturb.change_names_country_specific(country, gender), n=n)

			list_ex_to_tag=[]
			list_ex_to_tag_ini = []

			for list_transf, lab in zip(ret.data, labels):
				len_ex_transformed = len(list_transf[1:])

				# if there are transformed data because the text contains an entity 
				if len_ex_transformed > 1:
					list_ex_to_tag.append((list_transf[1:], [lab]*len_ex_transformed ))
					list_ex_to_tag_ini.append((list_transf[0], lab))

			list_transformed[country][gender] = ([kk for k in list_ex_to_tag for kk in k[0]], [kk for k in list_ex_to_tag for kk in k[1]])

			# on sauve les tweeets d'origine 
			if 'Original' not in list_transformed.keys():
				list_transformed['Original'] = ([k[0] for k in list_ex_to_tag_ini], [k[1] for k in list_ex_to_tag_ini])
		
		return list_transformed
	
if __name__ == '__main__':
	
	data = ['John is a very smart person, he lives in Ireland.','Luke Smith has 3 sisters.', 'Luke is awful', 'I do not like salmon', 'Mary came home yesterday']
	
	perturber = PerturbedExamples(list_countries=['France', 'Spain', 'Sweden'])
	
	print(perturber.all_countries(data, ['positive', 'positive', 'negative', 'negative', 'neutral']))