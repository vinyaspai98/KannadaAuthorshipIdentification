# Basic libraries
import io
import os
import sys
import argparse
import numpy as np
from os import walk
from sklearn import svm
#import feature_extract as test
# Scikit learn stuff
from sklearn import linear_model
from sklearn.metrics import *
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from firebase import firebase
firebase_db = firebase.FirebaseApplication('https://hackathon-ab821.firebaseio.com/', None)
import pyrebase
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

config = {
  "apiKey": "AIzaSyB_6OR0rAtv3xgCzQC45A0mjdzTW_KF2cw",
  "authDomain": "hackathon-ab821.firebaseapp.com",
  "databaseURL": "https://hackathon-ab821.firebaseio.com",
  "projectId": "hackathon-ab821",
  "storageBucket": "hackathon-ab821.appspot.com",
  "messagingSenderId": "407886884054",
  "appId": "1:407886884054:web:9b0c1709e25124bb"
}

firebase = pyrebase.initialize_app(config)
db = firebase.database()
Train_status = {
	"status" : "null",
	"text": "null"	
	}
firebase_db.put('','Train',Train_status)
msg = {
	"status" : "null"	
	}
firebase_db.put('','msg',msg)
test_ = {
    "author" : "null",
    "Prediction array": "null",	
    "Prediction Probability": "null",	
	}
firebase_db.put('','Test',test_)


def stream_handler(message):
	global best_classifier
    # print(message["event"]) # put
    # print(message["path"]) # /-K7yGTTEp7O549EzTYtI
	# print(message["data"]) # {'title': 'Pyrebase', "body": "etc..."}
	print(message)
	status = message["data"]['status']# {'title': 'Pyrebase', "body": "etc..."}
	text = message["data"]['text']# {'title': 'Pyrebase', "body": "etc..."}
	print(status)
	if (status == 'start'):
		best_classifier,best_accuracy = Train()
		Train_status = {
			"status" : "trained"+'_'+str(best_accuracy),
			"text": "null"	
			}
		firebase_db.put('','Train',Train_status)
	
	
	if (status.__contains__('trained') and text != "null"):
		Test(best_classifier)
	

print('Initialising...')
my_stream = db.child("Train").stream(stream_handler)


def calculateTop5Accuracy(labels, predictionProbs):
	"""
	Takes as input labels and prediction probabilities and calculates the top-5 accuracy of the model
	"""
	acc = []
	for i in range(0, len(predictionProbs)):
		predProbs = predictionProbs[i]
		predProbsIndices = np.argsort(-predProbs)[:5]
		if labels[i] in predProbsIndices:
			acc.append(1)
		else:
			acc.append(0)

	return round(((acc.count(1) * 100) / len(acc)), 2)


# Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument("--articles_per_author", type=int, default = 12, help="Number of articles to use for each author. If an author has fewer number of articles, we ignore it.")
parser.add_argument("--authors_to_keep", type=int, default = 3, help="Number of authors to use for training and testing purposes")
parser.add_argument("--data_folder", type=str, default = "data", help="Folder where author data is kept. Each author should have a separate folder where each article should be a separate file inside the author's folder.")
parser.add_argument("--model", type=str, default = "RandomForest", help="Training model")
parser.add_argument("--doc", type=str, default = "test.txt", help="Testing input")
parser.add_argument("--id", type=int, default = 5, help="id")


args = parser.parse_args()
ARTICLES_PER_AUTHOR = args.articles_per_author
AUTHORS_TO_KEEP = args.authors_to_keep
DATA_FOLDER = args.data_folder
MODEL = args.model
DOC = args.doc
ID = args.id


def Train():
	# Load raw data from the folder
	print("Data loading...")
	msg_status = {
			"status" : "Data loading...",
				
			}
	firebase_db.put('','msg',msg_status)
	Train_status = {
			"status" : "training",
			"text": "null"	
			}
	firebase_db.put('','Train',Train_status)
	folders = []
	vector=[]
	for(_,dirs,_) in walk(DATA_FOLDER):
		folders.extend(dirs)

	authorArticles = []
	labels = []
	authorId = 0
	for author in folders:
		authorFiles = []
		for(_,_,f) in walk(DATA_FOLDER + "/" + author):
			authorFiles.extend(f)

		if len(authorFiles) < ARTICLES_PER_AUTHOR:
			continue

		authorFiles = authorFiles[:ARTICLES_PER_AUTHOR]
		print("Loading %d files from %s" % (len(authorFiles), author))
		msg_status = {
			"status" : "Loading "+str(len(authorFiles)) +" files from "+str(author.replace('data_','')),
				
			}
		firebase_db.put('','msg',msg_status)
		temp_vector=[]
		for file in authorFiles:
			data = open(DATA_FOLDER + "/" + author + "/" + file, "r",encoding="utf8").readlines()
			data = ''.join(str(line) for line in data)
			temp_vector=FeatureExtration(data,15,4) 
			authorArticles.append(data)
			vector += temp_vector
			for K in range(len(temp_vector)):
				labels.append(authorId)

		# Stop when we have stored data for AUTHORS_TO_KEEP
		authorId = authorId + 1
		if authorId == AUTHORS_TO_KEEP:
			break


	from sklearn.utils import shuffle
	vector = np.array(vector)
	labels = np.array(labels)

	vector,labels = shuffle(vector,labels)


	print("\nTraining and testing...")
	print('ML model :',MODEL.upper(),'\n\n')
	# Train and get results
	accuracies, precisions, recalls, fscores, top5accuracies = [], [], [], [], []
	temp_accuracy = 0
	for i in range(10): # Train and test 10 different times and average the results
		# Split data into training and testing
		trainData, testData, trainLabels, testLabels = train_test_split(vector, labels, test_size=0.2)

		# Convert raw corpus into tfidf scores
		# vectorizer = TfidfVectorizer(min_df = 10)
		# vectorizer.fit(trainData)
		# trainData = vectorizer.transform(trainData).toarray()
		# testData = vectorizer.transform(testData).toarray()
		
		# Create a classifier instance
		if MODEL == 'RandomForest':
			classifier = RandomForestClassifier(n_estimators = 120)
		if MODEL == 'svm':
			classifier =  svm.SVC(kernel='linear',probability=True)
		if MODEL == 'NaiveBayes':
			classifier =  GaussianNB()
		if MODEL == 'knn':
			classifier = KNeighborsClassifier(n_neighbors= 3)
		

		# classifier =  svm.SVC(kernel='linear',probability=True)
		# classifier = KNeighborsClassifier(n_neighbors= 3)
		# Train classifier
		classifier.fit(trainData, trainLabels)
		
		# Get test predictions
		testPredictions = classifier.predict(testData)
		
		testPredictionsProbs = classifier.predict_proba(testData)
		testTopFiveAccuracy = calculateTop5Accuracy(testLabels, testPredictionsProbs)

		# Calculate metrics
		accuracy = round(accuracy_score(testLabels, testPredictions) * 100, 2)
		precision = round(precision_score(testLabels, testPredictions, average = 'macro') * 100, 2)
		recall = round(recall_score(testLabels, testPredictions, average = 'macro') * 100, 2)
		fscore = round(f1_score(testLabels, testPredictions, average = 'macro',labels=np.unique(testPredictions)) * 100, 2)

		# Store metrics in lists
		if(temp_accuracy< accuracy):
			temp_classifier=classifier
			temp_accuracy=accuracy
			# print('best score,', temp_accuracy)
		accuracies.append(accuracy)
		# print('Accuracy:', accuracy)
		precisions.append(precision) 
		recalls.append(recall) 
		fscores.append(fscore) 
		top5accuracies.append(testTopFiveAccuracy)
	
	best_accuracy = round(max(np.mean(accuracies), np.mean(precisions), np.mean(recalls), np.mean(fscores)),2)
	print("Accuracy: ",best_accuracy)
	return temp_classifier,best_accuracy

count=1
def Test(classifier):
	temp_vector = []
	test_vector = []
	test_labels = []
	data = firebase_db.get('/Train',None)['text']
	data = ''.join(str(line) for line in data)
	# print(data)
	temp_vector=FeatureExtration(data,15,4)
	test_vector += temp_vector
	# for K in range(len(temp_vector)):
	# 	test_labels.append(ID)
	testPredictions = classifier.predict(test_vector)
	testPredictionsProbs = classifier.predict_proba(test_vector)
	print(testPredictions)
	arr = testPredictions.tolist()
	if(max(arr,key=arr.count) == 0):
		name = 'Somashekar'
	if(max(arr,key=arr.count) == 1):
		name = 'Hrudayashiva'
	if(max(arr,key=arr.count) == 2):
		name = 'Ravi belegere'
	global count
	author_name=name+'_author'
	for k in range(count):
		author_name+='_'
	Test_status = {
			"author" : author_name	
			}
	count+=1
	# Test_status = {
	# 		"author" : name+'_author',
	# 		"Prediction array": str(arr),	
	# 		"Prediction Probability": str(testPredictionsProbs),	
	# 		}
	firebase_db.put('','Test',Test_status)
	print('Author Name: ', name)
	print(testPredictionsProbs)



############################Feature_extract###############
import collections as coll
import math
import pickle
import string
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



def slidingWindow(sequence, winSize, step=1):
    try:
        it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence length.")

    sequence = sent_tokenize(sequence)

    # Pre-compute number of chunks to omit
    numOfChunks = int(((len(sequence) - winSize) / step) + 1)

    l = []
    # Do the work
    for i in range(0, numOfChunks * step, step):
        l.append(" ".join(sequence[i:i + winSize]))

    return l




# removing stop words plus punctuation.
def Avg_wordLength(str):
    str.translate(string.punctuation)
    tokens = word_tokenize(str, language='english')
    st = [",", ".", "'", "!", '"', "#", "$", "%", "&", "(", ")", "*", "+", "-", ".", "/", ":", ";", "<", "=", '>', "?",
          "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']
    stop = stopwords.words('english') + st
    words = [word for word in tokens if word not in stop]
    return np.average([len(word) for word in words])


# ----------------------------------------------------------------------------


# returns avg number of characters in a sentence
def Avg_SentLenghtByCh(text):
    tokens = sent_tokenize(text)
    return np.average([len(token) for token in tokens])


# ----------------------------------------------------------------------------

# returns avg number of words in a sentence
def Avg_SentLenghtByWord(text):
    tokens = sent_tokenize(text)
    return np.average([len(token.split()) for token in tokens])



# -----------------------------------------------------------------------------

# COUNTS SPECIAL CHARACTERS NORMALIZED OVER LENGTH OF CHUNK
def CountSpecialCharacter(text):
    st = ["#", "$", "%", "&", "(", ")", "*", "+", "-", "/", "<", "=", '>',
          "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']
    count = 0
    for i in text:
        if (i in st):
            count = count + 1
    return count / len(text)


# ----------------------------------------------------------------------------

def CountPuncuation(text):
    st = [",", ".", "'", "!", '"', ";", "?", ":", ";"]
    count = 0
    for i in text:
        if (i in st):
            count = count + 1
    return float(count) / float(len(text))


# ----------------------------------------------------------------------------


# also returns Honore Measure R
def hapaxLegemena(text):
    words = RemoveSpecialCHs(text)
    V1 = 0
    # dictionary comprehension . har word kay against value 0 kardi
    freqs = {key: 0 for key in words}
    for word in words:
        freqs[word] += 1
    for word in freqs:
        if freqs[word] == 1:
            V1 += 1
    N = len(words)
    V = float(len(set(words)))
    R = 100 * math.log(N) / max(1, (1 - (V1 / V)))
    h = V1 / N
    return R, h


# ---------------------------------------------------------------------------

def hapaxDisLegemena(text):
    words = RemoveSpecialCHs(text)
    count = 0
    # Collections as coll Counter takes an iterable collapse duplicate and counts as
    # a dictionary how many equivelant items has been entered
    freqs = coll.Counter()
    freqs.update(words)
    for word in freqs:
        if freqs[word] == 2:
            count += 1

    h = count / float(len(words))
    S = count / float(len(set(words)))
    return S, h


# ---------------------------------------------------------------------------

# c(w)  = ceil (log2 (f(w*)/f(w))) f(w*) frequency of most commonly used words f(w) frequency of word w
# measure of vocabulary richness and connected to zipfs law, f(w*) const rak kay zips law say rank nikal rahay hein
def AvgWordFrequencyClass(text):
    words = RemoveSpecialCHs(text)
    # dictionary comprehension . har word kay against value 0 kardi
    freqs = {key: 0 for key in words}
    for word in words:
        freqs[word] += 1
    maximum = float(max(list(freqs.values())))
    return np.average([math.floor(math.log((maximum + 1) / (freqs[word]) + 1, 2)) for word in words])


# --------------------------------------------------------------------------
# TYPE TOKEN RATIO NO OF DIFFERENT WORDS / NO OF WORDS
def typeTokenRatio(text):
    words = word_tokenize(text)
    return len(set(words)) / len(words)


# --------------------------------------------------------------------------
# logW = V-a/log(N)
# N = total words , V = vocabulary richness (unique words) ,  a=0.17
# we can convert into log because we are only comparing different texts
def BrunetsMeasureW(text):
    words = RemoveSpecialCHs(text)
    a = 0.17
    V = float(len(set(words)))
    N = len(words)
    B = (V - a) / (math.log(N))
    return B


# ------------------------------------------------------------------------
def RemoveSpecialCHs(text):
    text = word_tokenize(text)
    st = [",", ".", "'", "!", '"', "#", "$", "%", "&", "(", ")", "*", "+", "-", ".", "/", ":", ";", "<", "=", '>', "?",
          "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']

    words = [word for word in text if word not in st]
    return words


# -------------------------------------------------------------------------
# K  10,000 * (M - N) / N**2
# , where M  Sigma i**2 * Vi.
def YulesCharacteristicK(text):
    words = RemoveSpecialCHs(text)
    N = len(words)
    freqs = coll.Counter()
    freqs.update(words)
    vi = coll.Counter()
    vi.update(freqs.values())
    M = sum([(value * value) * vi[value] for key, value in freqs.items()])
    K = 10000 * (M - N) / math.pow(N, 2)
    return K


# -------------------------------------------------------------------------


# -1*sigma(pi*lnpi)
# Shannon and sympsons index are basically diversity indices for any community
def ShannonEntropy(text):
    words = RemoveSpecialCHs(text)
    lenght = len(words)
    freqs = coll.Counter()
    freqs.update(words)
    arr = np.array(list(freqs.values()))
    distribution = 1. * arr
    distribution /= max(1, lenght)
    import scipy as sc
    H = sc.stats.entropy(distribution, base=2)
    # H = sum([(i/lenght)*math.log(i/lenght,math.e) for i in freqs.values()])
    return H


# ------------------------------------------------------------------
# 1 - (sigma(n(n - 1))/N(N-1)
# N is total number of words
# n is the number of each type of word
def SimpsonsIndex(text):
    words = RemoveSpecialCHs(text)
    freqs = coll.Counter()
    freqs.update(words)
    N = len(words)
    n = sum([1.0 * i * (i - 1) for i in freqs.values()])
    D = 1 - (n / (N * (N - 1)))
    return D



# -----------------------------------------------------------------
def dale_chall_readability_formula(text, NoOfSectences):
    words = RemoveSpecialCHs(text)
    difficult = 0
    adjusted = 0
    NoOfWords = len(words)
    with open('dale-chall.pkl', 'rb',encoding="utf8") as f:
        fimiliarWords = pickle.load(f)
    for word in words:
        if word not in fimiliarWords:
            difficult += 1
    percent = (difficult / NoOfWords) * 100
    if (percent > 5):
        adjusted = 3.6365
    D = 0.1579 * (percent) + 0.0496 * (NoOfWords / NoOfSectences) + adjusted
    return D




def PrepareData(text1, text2, Winsize):
    chunks1 = slidingWindow(text1, Winsize, Winsize)
    chunks2 = slidingWindow(text2, Winsize, Winsize)
    return " ".join(str(chunk1) + str(chunk2) for chunk1, chunk2 in zip(chunks1, chunks2))


# ------------------------------------------------------------------

# returns a feature vector of text
def FeatureExtration(text, winSize, step):
    # cmu dictionary for syllables

    chunks = slidingWindow(text, winSize, step)
    vector = []
    for chunk in chunks:
        feature = []
        # LEXICAL FEATURES
        meanwl = (Avg_wordLength(chunk))
        feature.append(meanwl)

        meansl = (Avg_SentLenghtByCh(chunk))
        feature.append(meansl)

        mean = (Avg_SentLenghtByWord(chunk))
        feature.append(mean)

        means = CountSpecialCharacter(chunk)
        feature.append(means)

        p = CountPuncuation(chunk)
        feature.append(p)
    

        TTratio = typeTokenRatio(chunk)
        feature.append(TTratio)

        HonoreMeasureR, hapax = hapaxLegemena(chunk)
        feature.append(hapax)

        SichelesMeasureS, dihapax = hapaxDisLegemena(chunk)
        feature.append(dihapax)

        YuleK = YulesCharacteristicK(chunk)
        feature.append(YuleK)

        S = SimpsonsIndex(chunk)
        feature.append(S)

        B = BrunetsMeasureW(chunk)
        feature.append(B)

        Shannon = ShannonEntropy(text)
        feature.append(Shannon)

        vector.append(feature)
    return vector




	
