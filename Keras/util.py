import argparse
import sys
import tempfile
import tensorflow as tf
import numpy as np

FILE = '../data/sample.csv'

def str2int(s):
    return int(s)

def read_csv(filename):
    features = []
    policy_labels = []
    value_labels = []
    with open(filename) as inf:
        next(inf)
        for line in inf:
            currentLine = line.strip().split(",")
            currentLine = list(map(str2int, currentLine))
            features.append(currentLine[0:192])
            policy_labels.append(currentLine[192:193])
            value_labels.append(currentLine[193:194])
    return np.array(features[:-60]), \
           np.array(policy_labels[:-60]), \
           np.array(value_labels[:-60]), \
           np.array(features[-60:]), \
           np.array(policy_labels[-60:]), \
           np.array(value_labels[-60:])

def get_policy(predictions):
    return np.fliplr(np.argsort(predictions[0]))

def get_value(predictions):
    return predictions[1]
