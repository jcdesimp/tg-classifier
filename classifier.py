import argparse


def runArgParser():
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

def trainModel():
  pass

def testModel():
  pass

def classifyMessage():
  pass

def main():
  args = runArgParser()
  if args.command == 'train':
    trainModel()
  elif args.command == 'test':
    testMOdel()
  elif args.command == 'class':
    classifyMessage()



if __name__ == '__main__':
  main()