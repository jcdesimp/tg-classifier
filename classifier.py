import argparse
import json
import nltk
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

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
  parser_test.add_argument('message', metavar='MSG', help='The message to classify')

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
  with open(filename) as f:
    # process each message
    featureList = []
    labelList = []
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
        labelList.append(lineData['class'])
        featureList.append(lineData['features'])
    # return
    feat_vector = DictVectorizer().fit(featureList)
    X_train = feat_vector.transform(featureList)
    classifier = LinearSVC().fit(X_train, labelList)

    # save the trained model to disk
    joblib.dump(classifier, MODEL_FILE)
    # save the fitted feature vector
    joblib.dump(feat_vector, FEAT_VECTOR_FILE) 

def testModel():
  pass

def classifyMessage():
  pass

def main():
  '''main program execution'''
  args = runArgParser()
  if args.command == 'train':
    trainModel(args.file)
  elif args.command == 'test':
    testMOdel()
  elif args.command == 'class':
    classifyMessage()



if __name__ == '__main__':
  main()