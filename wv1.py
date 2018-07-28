import configparser
import gensim 
from gensim.models import Word2Vec
import logging
from os import listdir

class MySentences(object):
	def __init__(self, dirname):
		self.dirname = 'C:\\Users\\elze\\Documents\\MyProjects\\Python\\SkillClustersProcessedJobPostings\\SCP3\\20180619\\0001_0200'
 
	def __iter__(self):
		for jobFileName in os.listdir(self.dirname):
			jobFileExtendedName = importFileDirectory + "\\" + jobFileName
			jobFile = open(jobFileExtendedName, "r")
			jobDescription = jobFile.read()

			lines = jobDescription.split("\n")
			jobDescriptionOneLine = ';'.join(lines);
			#oneLineJobDescriptions.append(jobDescriptionOneLine)
			yield gensim.utils.simple_preprocess(jobDescriptionOneLine)


config = configparser.RawConfigParser()

importFileDirectory = 'C:\\Users\\elze\\Documents\\MyProjects\\Python\\SkillClustersProcessedJobPostings\\SCP3\\20180619\\0001_0200'

def read_input():

	oneLineJobDescriptions = []
	for jobFileName in listdir(importFileDirectory):
		jobFileExtendedName = importFileDirectory + "\\" + jobFileName
		jobFile = open(jobFileExtendedName, "r")
		jobDescription = jobFile.read()

		lines = jobDescription.split("\n")
		jobDescriptionOneLine = ';'.join(lines);
		oneLineJobDescriptions.append(jobDescriptionOneLine)
	return oneLineJobDescriptions


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = Word2Vec(read_input(), iter=10, min_count=10, size=300, workers=4)

vocab_size = len(model.wv.vocab)
print ("vocab_size = " + str(vocab_size))
#print(model.wv.index2word[vocab_size - 1], model.wv.index2word[vocab_size - 2], model.wv.index2word[vocab_size - 3])

for i in range(vocab_size - 1):
	print(model.wv.index2word[i])




