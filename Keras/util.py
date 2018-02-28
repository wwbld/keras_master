import tensorflow as tf
import numpy as np
import copy

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
    return np.fliplr(np.argsort(predictions[0]))[0]

def get_value(predictions):
    return predictions[1][0]

# given a board and next move, build a new board and 
# return the most possible policy result
def give_exact_prediction(model, board, move, player, human, ai):
    board_copy = copy.deepcopy(board)
    board_copy.update(player, move)
    width = board_copy.width
    height = board_copy.height
    state = []
    for i in range(height-1,-1,-1):
        for j in range(width):
            loc = i * width + j
            if board_copy.states[loc] == human:
                state.append(1)
                state.append(0)
                state.append(0)
            elif board_copy.states[loc] == ai:
                state.append(0)
                state.append(1)
                state.append(0)
            else:
                state.append(0)
                state.append(0)
                state.append(0)
    predictions = model.predict(np.array([state]))
    return get_policy(predictions)[0], get_value(predictions)[player-1]    
