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
            features.append(currentLine[0:144])
            policy_labels.append(currentLine[144:145])
            value_labels.append(currentLine[145:146])
    return np.array(features[:-600]), \
           np.array(policy_labels[:-600]), \
           np.array(value_labels[:-600]), \
           np.array(features[-600:]), \
           np.array(policy_labels[-600:]), \
           np.array(value_labels[-600:])

