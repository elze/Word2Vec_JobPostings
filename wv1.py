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

def read_input_old():

	oneLineJobDescriptions = []
	for jobFileName in listdir(importFileDirectory):
		jobFileExtendedName = importFileDirectory + "\\" + jobFileName
		jobFile = open(jobFileExtendedName, "r")
		jobDescription = jobFile.read()

		lines = jobDescription.split("\n")
		jobDescriptionOneLine = ';'.join(lines);
		wordsInASentence = gensim.utils.simple_preprocess(jobDescriptionOneLine, deacc=False, min_len=2, max_len=30)
		
		oneLineJobDescriptions.append(jobDescriptionOneLine)
	return oneLineJobDescriptions


def read_input():

	jobTokenLists = []
	for jobFileName in listdir(importFileDirectory):
		jobFileExtendedName = importFileDirectory + "\\" + jobFileName
		jobFile = open(jobFileExtendedName, "r")
		jobDescription = jobFile.read()

		lines = jobDescription.split("\n")
		jobDescriptionOneLine = ';'.join(lines);
		wordsInASentence = gensim.utils.simple_preprocess(jobDescriptionOneLine, deacc=False, min_len=2, max_len=30)
		jobTokenLists.append(wordsInASentence)
		
	return jobTokenLists


#sampleSentences = [['Experience', 'in', 'ASP.NET MVC', 'and', 'WebAPI'], ['Strong', 'JavaScript', 'background'], ['Proficiency', 'in', 'SQL Server'], ['Knowledge', 'working', 'in', 'AWS', 'Azure', 'or', 'other', 'cloud' 'services']]
# train word2vec on the two sentences
#model = gensim.models.Word2Vec(sentences, min_count=1)

jobPostingTokenLists = read_input()

model = gensim.models.Word2Vec(jobPostingTokenLists, iter=10, min_count=1, size=300, workers=4)
vocab_size = len(model.wv.vocab)
print ("vocab_size = " + str(vocab_size))

for i in range(vocab_size - 1):
	print(model.wv.index2word[i])

#for jtl in jobPostingTokenLists:
#	print(jtl)