import argparse
import json
import nltk
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import numpy
import os

from flask import Flask, jsonify, request

MODEL_FILE = './data/trained_model.pkl'
FEAT_VECTOR_FILE = './data/model_feat_vector.pkl'
MIN_SAMPLE_SIZE = 5
MIN_CLASSIFY_SIZE = 4


# model cache
LOADED_MODEL = None
LOADED_FEAT = None

def initializeClassifier():
  return SVC(kernel='linear', probability=True)

def runArgParser():
  '''parse commandline arguments'''
  parser = argparse.ArgumentParser(description='Train/Test a telegram people classifier.')
  # create subparser
  subparsers = parser.add_subparsers(dest='command', metavar='CMD', help='Train a model, test a model, or classify something.')
  subparsers.required = True
  # subparser for training
  parser_train = subparsers.add_parser('train', help='Train a model with the given training data.')
  parser_train.add_argument('file', metavar='FILE', help='File containing training data.')
  # subparser for testing
  parser_test = subparsers.add_parser('test', help='Test a model with the given test data. Trained model must exist.')
  parser_test.add_argument('file', metavar='FILE', help='File containing test data.')
  # classify 
  parser_class = subparsers.add_parser('class', help='Classify a single message using the trained model.')
  parser_class.add_argument('message', metavar='MSG', help='The message to classify')

  # training data metrics 
  parser_metrics = subparsers.add_parser('metrics', help='Print metrics about the given training or test data.')
  parser_metrics.add_argument('file', metavar='FILE', help='File containing training or test data.')

  parser_metrics = subparsers.add_parser('http', help='Expose a simple restful API for classifying messages via a Flask HTTP server.')

  return parser.parse_args();


def preprocessText(textString):
  '''given a telegram message, pre-process it and return a dictionary
    {
      "raw": "The Original Message",
      "tokens": ['tokenized', 'array', 'of', 'words'],
      "pos_tags": [('tag','V'),('words','NN')],
      "token_set": {'bag', 'of', 'unique', 'words'},
      "bigrams": [('two', 'words')],
      "trigrams": [('now', 'three', 'words')],
    }
  '''
  textData = {}
  textData['raw'] = textString
  textData['tokens'] = nltk.tokenize.word_tokenize(textString)
  textData['tokens_no_stops'] = [word for word in textData['tokens'] if word not in nltk.corpus.stopwords.words('english')]
  textData['pos_tags'] = nltk.pos_tag(textData['tokens'])
  textData['token_set'] = { x.lower() for x in textData['tokens'] }
  textData['bigrams'] = [tuple(textData['tokens'][i:i+2]) for i in range(len(textData['tokens'])-2+1)]
  textData['trigrams'] = [tuple(textData['tokens'][i:i+3]) for i in range(len(textData['tokens'])-3+1)]
  textData['pos_bigrams'] = [tuple(textData['pos_tags'][i:i+2][1]) for i in range(len(textData['pos_tags'])-2+1)]

  return textData

def extractFeatures(textData):
  '''extract features from the text data and return a dictionary of features'''
  features = defaultdict(int)
  # iterate pos_tags
  for i in range(len(textData['pos_tags'])):
    curr_word = textData['pos_tags'][i]
    # features['word+' + str(i)] = curr_word[0].lower()
    # features['pos+' + str(i)] = curr_word[1]
    # features['word-' + str(i - len(textData['pos_tags'])-1)] = curr_word[0].lower()
    # features['pos-' + str(i - len(textData['pos_tags'])-1)] = curr_word[1]
    # ensure at least 1 word in
    # if i >= 1:
      # features['word_' + curr_word[0].lower() + '-1'] = textData['pos_tags'][i-1][0].lower()
      # features['pos_' + curr_word[1] + '-1'] = textData['pos_tags'][i-1][1]
    # ensure 1 word left
    # if i < len(textData['pos_tags']) - 1:
    #   features['word_' + curr_word[0].lower() + '+1'] = textData['pos_tags'][i+1][0].lower()
    #   features['pos_' + curr_word[1] + '+1'] = textData['pos_tags'][i+1][1]
    # ensure at least 2 words in
    if i >= 2:
      features['word_' + curr_word[0].lower() + '-2'] = textData['pos_tags'][i-2][0].lower()
      features['pos_' + curr_word[1] + '-2'] = textData['pos_tags'][i-2][1]
    # ensure 2 words left
    if i < len(textData['pos_tags']) - 2:
      features['word_' + curr_word[0].lower() + '+2'] = textData['pos_tags'][i+2][0].lower()
      features['pos_' + curr_word[1] + '+2'] = textData['pos_tags'][i+2][1]
  # iterate POS bigrams
  for t in textData['pos_bigrams']:
    features['has_pos_bigram_' + str(t)] = True
  # iterate bigrams
  for t in textData['bigrams']:
    features['has_bigram_' + str(t)] = True
  # iterate trigrams
  for t in textData['trigrams']:
    features['has_trigram_' + str(t)] = True
  # itertate set of tokens
  for t in textData['token_set']:
    # contains token at all
    features['has_word_' + t] = True
  # iterate all tokens
  for t in textData['tokens']:
    features['count_' + t.lower()] += 1


  return features


def trainModel(filename):
  
  featureList = []
  label_list = []

  with open(filename) as f:
    # process each message
    for line in f:
      messageLine = json.loads(line)
      # only process if text message
      if 'text' in messageLine:
        lineData = {
          'class': messageLine['from']['print_name']
        }
        textData = preprocessText(messageLine['text'])
        # check if its worth keeping
        if len(textData['tokens']) < MIN_SAMPLE_SIZE:
          continue
        features = extractFeatures(textData)
        lineData['data'] = textData
        lineData['features'] = features
        label_list.append(lineData['class'])
        featureList.append(lineData['features'])
  # return
  vectorizer = DictVectorizer()

  feat_vector_unrestricted = vectorizer.fit_transform(featureList)
  # X_unrestricted = feat_
  support = SelectKBest(chi2, k=30).fit(feat_vector_unrestricted, label_list)
  vectorizer.restrict(support.get_support())
  feat_vector = vectorizer.fit(featureList)
  X_train = feat_vector.transform(featureList)
  classifier =  initializeClassifier().fit(X_train, label_list)

  # save the trained model to disk
  joblib.dump(classifier, MODEL_FILE)
  # save the fitted feature vector
  joblib.dump(feat_vector, FEAT_VECTOR_FILE) 

def testModel(filename):
  if not os.path.isfile(MODEL_FILE) or not os.path.isfile(FEAT_VECTOR_FILE):
    print('Model "' + MODEL_FILE +'" not found.')
    print('be sure to train model first!')
    return 

  featureList = []
  label_list = []

  with open(filename) as f:
    # process each message
    for line in f:
      messageLine = json.loads(line)
      # only process if text message
      if 'text' in messageLine:
        lineData = {
          'class': messageLine['from']['print_name']
        }
        textData = preprocessText(messageLine['text'])
        if len(textData['tokens']) < MIN_SAMPLE_SIZE:
          continue
        features = extractFeatures(textData)
        lineData['data'] = textData
        lineData['features'] = features
        label_list.append(lineData['class'])
        featureList.append(lineData['features'])

  feat_vector = joblib.load(FEAT_VECTOR_FILE)
  X_test = feat_vector.transform(featureList)

  classifier = joblib.load(MODEL_FILE)

  test_pred = classifier.predict(X_test)
  # print(len(test_pred))
  # print(len(label_list))
  acc = numpy.mean(test_pred == label_list) * 100
  print('Accuracy: ' + str(acc))


def loadModelIfNot():
  global LOADED_MODEL
  global LOADED_FEAT
  if not(LOADED_MODEL and LOADED_FEAT):
    LOADED_MODEL = joblib.load(MODEL_FILE)
    LOADED_FEAT = joblib.load(FEAT_VECTOR_FILE)
  return LOADED_MODEL, LOADED_FEAT


def classifyMessage(message, as_json=False):
  textData = preprocessText(message)

  if len(textData['tokens']) < MIN_CLASSIFY_SIZE:
    print("Message too short to accurately classify!")
    return
  features = extractFeatures(textData)

  classifier, feat_vector = loadModelIfNot()
  
  X_test = feat_vector.transform([features])

  probs = classifier.predict_proba(X_test)
  results = []
  for i, v in enumerate(probs[0]):
    results.append((classifier.classes_[i], v))

  results.sort(key=lambda r: r[1], reverse=True)

  if as_json:
    json_res = {}
    json_res['message'] = message
    json_res['classes'] = [ {'label': { 'print_name': r[0] }, 'probability': r[1] } for r in results ]
    return json_res
  else:
    for r in results:
      print(r[0].ljust(20), r[1]*100)

def runHttpServer():
  app = Flask(__name__)
  loadModelIfNot()

  @app.route('/')
  def index():
    return 'Classify messages at "/classify"'

  @app.route('/classify')
  def classify():
    message = request.args.get('message')
    if not(message):
      err = {'error': "Message required"}
      return jsonify(err), 400
    if len(nltk.tokenize.word_tokenize(message)) < MIN_CLASSIFY_SIZE:
      err = {'error': 'Message too short. Must be at least ' + str(MIN_CLASSIFY_SIZE) + ' words.'}
      return jsonify(err), 400

    return jsonify(classifyMessage(message, as_json=True))

  app.run()

def calcTrainingMetrics(filename):
  wordCounts = defaultdict(int)
  userCounts = defaultdict(int)
  with open(filename) as f:
    # process each message
    for line in f:
      messageLine = json.loads(line)
      # only process if text message
      if 'text' in messageLine:
        lineData = {
          'class': messageLine['from']['print_name']
        }
        textData = preprocessText(messageLine['text'])
        # check if its worth keeping
        if len(textData['tokens']) < MIN_SAMPLE_SIZE:
          continue
        for t in textData['tokens_no_stops']:
          wordCounts[t.lower()] += 1
        userCounts[messageLine['from']['print_name']] += 1
  
  print("User Sampling")
  userCounts = [ (k,v) for k, v in userCounts.items() ]
  userCounts.sort(key=lambda r: r[1], reverse=True)
  print('Rank'.ljust(7) + 'Person'.ljust(25) + 'Sample Count')
  print('-'*50)
  for i, v in enumerate(userCounts):
    print((str(i+1) + '.').ljust(7) + str(v[0]).ljust(25) + str(v[1]))
  

def main():
  '''main program execution'''
  args = runArgParser()
  if args.command == 'train':
    trainModel(args.file)
  elif args.command == 'test':
    testModel(args.file)
  elif args.command == 'class':
    classifyMessage(args.message)
  elif args.command == 'metrics':
    calcTrainingMetrics(args.file)
  elif args.command == 'http':
    runHttpServer()



if __name__ == '__main__':
  main()