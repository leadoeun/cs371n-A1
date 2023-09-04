# models.py

from sentiment_data import *
from sentiment_data import List
from utils import *
import numpy as np
import random

from collections import Counter

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        # raise Exception("Must be implemented")
    def get_indexer(self):
        return self.indexer
    
    def initialization(self, train_exs):
        count = Counter()
        for sentence in train_exs:
            count += self.extract_features(sentence.words, True)
        return count


    def extract_features(self, sentence: List[str], add_to_indexer: bool=False)->Counter:
        count = Counter()
        for word in sentence:
            if (not self.indexer.contains(word)) and add_to_indexer:
                self.indexer.add_and_get_index(word, True)
                count[word] = 1
            elif self.indexer.contains(word):
                count[word] += 1
        return count

    


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weight_vector, featurizer):
        self.weight = weight_vector
        self.featurizer = featurizer
        
        # raise Exception("Must be implemented")
    def predict(self, sentence: List[str]) -> int:
        # use feature extractor to build feature vec (Counter)
        fvec = self.featurizer.extract_features(sentence, False)
        dot_product = 0.0
        for word in sentence:
            dot_product += self.weight[self.featurizer.get_indexer().index_of(word)] * fvec[word]
        return 1 if dot_product > 0 else 0


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self):
        raise Exception("Must be implemented")


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    random.shuffle(train_exs)
    # initialize weight = 0 with length of indexer (len of feature vector)
    count = feat_extractor.initialization(train_exs)
    weight = np.zeros(len(feat_extractor.get_indexer()))
    epochs = 15
    step_size = 1.3
    # print(len(weight), len(feat_extractor.get_indexer()))
    for t in range (0, epochs):
        random.shuffle(train_exs)
        # epoch 15, step size 1.3
        if t % 4 == 1:
            step_size -= 0.3
        #     print(t + 1)
        # if t % 4 == 1:
        #     step_size -= 1 / (t + 1)
            # print(t+1)
        # for a single sentence
        for sentence in train_exs:
            # bag of words Counter
            # print("cp1")
            count = feat_extractor.extract_features(sentence.words, False)
            ypred = PerceptronClassifier(weight, feat_extractor)
            if ypred.predict(sentence.words) == sentence.label:
                continue
            elif sentence.label == 1:
                for word in sentence.words:
                    # get index of weights that need to be updated
                    idx = feat_extractor.get_indexer().index_of(word)
                    weight[idx] += step_size * count[word]
            elif sentence.label == 0:
                for word in sentence.words:
                    # get index of weights that need to be updated
                    idx = feat_extractor.get_indexer().index_of(word)
                    weight[idx] -= step_size * count[word]

    n_highest = np.argpartition(weight, -10)[-10:]
    for n in n_highest:
        print(feat_extractor.get_indexer().get_object(n))
    n_lowest = np.argpartition(weight, 10)[:10]
    for n in n_lowest:
        print(feat_extractor.get_indexer().get_object(n))
    return PerceptronClassifier(weight, feat_extractor)
    raise Exception("Must be implemented")


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    raise Exception("Must be implemented")


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model