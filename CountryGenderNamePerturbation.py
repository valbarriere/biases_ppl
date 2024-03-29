"""
Script transforming the data into perturbed data with names typical from other countries, in order to test the bias of the model
Just return the examples with entities inside
Only work for english for now since it uses en_core_web_sm for the NER

Author: Anonymous_Submission 01/24
"""
import spacy
# Change en_core_web_sm for xx_ent_wiki_sm (+ efficiency) or xx_sent_ud_sm (+ accuracy) in order to go multilingual ; also I use xx_ent_wiki_sm just for entities
# see: https://spacy.io/models
# nlp = spacy.load('en_core_web_sm')
nlp = spacy.load('xx_ent_wiki_sm')

from checklist.perturb import Perturb
import itertools
from tqdm import tqdm

texts = ['John is a very smart person, he lives in Ireland.','Luke Smith has 3 sisters.', 'Luke is an ass']

labels = ['positive', 'positive', 'negative']

list_countries = ['France', 'Spain', 'Sweden']


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
				):
		self.list_countries = list_countries
		self.use_female = use_female
		
	def all_countries(self, texts, labels, n=2):
		"""
		Create the new data in the form of a dictionnary
		"""
		
		list_transformed={country : {} for country in self.list_countries}
		list_ex_to_tag=[]
		list_ex_to_tag_ini = []

		pdata = list(nlp.pipe(texts))

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