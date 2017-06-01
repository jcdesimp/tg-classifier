import argparse
import json
import nltk
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import numpy
import os

MODEL_FILE = './data/trained_model.pkl'
FEAT_VECTOR_FILE = './data/model_feat_vector.pkl'

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

  return parser.parse_args();

def preprocessText(textString):
  '''given a telegram message, preprocess it and return a dictionary
    {
      "raw": "The Original Message",
      "tokens": ['tokenized', 'array', 'of', 'words'],
      "pos_tags": [('tag','V'),('words','NN')],
      "token_set": {'bag', 'of', 'unique', 'words'},
    }
  '''
  textData = {}
  textData['raw'] = textString
  textData['tokens'] = nltk.tokenize.word_tokenize(textString)
  textData['pos_tags'] = nltk.pos_tag(textData['tokens'])
  textData['token_set'] = { x.lower() for x in textData['tokens'] }

  return textData

def extractFeatures(textData):
  '''extract features from the text data and return a dictionary of features'''
  features = defaultdict(int)

  # contains word
  for t in textData['token_set']:
    features[t] = True

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
        features = extractFeatures(textData)
        lineData['data'] = textData
        lineData['features'] = features
        label_list.append(lineData['class'])
        featureList.append(lineData['features'])
  # return
  feat_vector = DictVectorizer().fit(featureList)
  X_train = feat_vector.transform(featureList)
  classifier = LinearSVC().fit(X_train, label_list)

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

def classifyMessage():
  pass

def main():
  '''main program execution'''
  args = runArgParser()
  if args.command == 'train':
    trainModel(args.file)
  elif args.command == 'test':
    testModel(args.file)
  elif args.command == 'class':
    classifyMessage()



if __name__ == '__main__':
  main()