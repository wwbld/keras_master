"""
author: arrti
github: https://github.com/arrti
blog:   http://www.cnblogs.com/xmwd

This implement of MC and UCT was not completely right,
but it can work, please see README.
Welcome to discuss with me.
""" 
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.models import load_model
import numpy as np
import util


import copy
import time
import csv
from random import choice, shuffle
from math import log, sqrt

N_TIMES = 5
SAMPLE = '../data/sample_2.csv'
MODEL = '../../model/model_1'

class Board(object):
    """
    board for game
    """

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        self.states = {} # board states, key:move, value: piece type
        self.n_in_row = int(kwargs.get('n_in_row', 5)) # need how many pieces in a row to win

    def init_board(self):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not less than %d' % self.n_in_row)

        self.availables = list(range(self.width * self.height)) # available moves

        for m in self.availables:
            self.states[m] = -1

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move  // self.width
        w = move  %  self.width
        return [h, w]

    def location_to_move(self, location):
        if(len(location) != 2):
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if(move not in range(self.width * self.height)):
            return -1
        return move


    def update(self, player, move):
        self.states[move] = player
        self.availables.remove(move)


class MCTS(object):
    """
    AI player, use Monte Carlo Tree Search with UCB
    """

    def __init__(self, board, model, 
                 play_turn, n_in_row=5, time=5, max_actions=1000):
        self.board = board
        self.model = model
        self.human = None
        self.ai = None
        self.play_turn = play_turn
        self.calculation_time = float(time)
        self.max_actions = max_actions
        self.n_in_row = n_in_row

        self.player = play_turn[0] # AI is first at now
        self.confident = 1.96
        self.max_depth = 1

    def get_action(self):
        if len(self.board.availables) == 1:
            return self.board.availables[0]

        self.plays = {} # key:(player, move), value:visited times
        self.wins = {} # key:(player, move), value:win times
        simulations = 0
        begin = time.time()
        while time.time() - begin < self.calculation_time:
            board_copy = copy.deepcopy(self.board)  # simulation will change board's states,
            play_turn_copy = copy.deepcopy(self.play_turn) # and play turn
            self.run_simulation(board_copy, play_turn_copy)
            simulations += 1

        #print("total simulations=", simulations)

        move = self.select_one_move()
        location = self.board.move_to_location(move)
        #print('Maximum depth searched:', self.max_depth)

        #print("AI move: %d,%d\n" % (location[0], location[1]))

        return move

    def run_simulation(self, board, play_turn):
        """
        MCTS main process
        """

        plays = self.plays
        wins = self.wins
        availables = board.availables

        player = self.get_player(play_turn)
        visited_states = set()
        winner = -1
        expand = True
        # Simulation
        for t in range(1, self.max_actions + 1):
            # Selection
            # if all moves have statistics info, choose one that have max UCB value
            if all(plays.get((player, move)) for move in availables):
                log_total = log(
                    sum(plays[(player, move)] for move in availables))
                value, move = max(
                    ((wins[(player, move)] / plays[(player, move)]) +
                     (util.give_exact_prediction(self.model, board, 
                      move, player, self.human, self.ai)[0]) / (1 + plays[(player, move)]), move)
                    for move in availables)   # AlphaGo
            else:
                # choose the action has the max (policy + value)
                value, move = max(
                    (util.give_exact_prediction(self.model, board, move, player, self.human, self.ai)[0] + 
                     util.give_exact_prediction(self.model, board, move, player, self.human, self.ai)[1], move)
                    for move in availables)   
            board.update(player, move)

            print('move is {0:2d}, player is {1}'.format(move, player))

            # Expand
            # add only one new child node each time
            if expand and (player, move) not in plays:
                expand = False
                plays[(player, move)] = 0
                wins[(player, move)] = 0
                if t > self.max_depth:
                    self.max_depth = t

            visited_states.add((player, move))

            is_full = not len(availables)
            win, winner = self.has_a_winner(board)
            if is_full or win:
                break

            player = self.get_player(play_turn)

        # Back-propagation
        for player, move in visited_states:
            if (player, move) not in plays:
                continue
            plays[(player, move)] += 1 # all visited moves
            if player == winner:
                wins[(player, move)] += 1 # only winner's moves
                print("winner is, ", player)

    def get_player(self, players):
        p = players.pop(0)
        players.append(p)
        return p

    def select_one_move(self):
        percent_wins, move = max(
            (self.wins.get((self.player, move), 0) /
             self.plays.get((self.player, move), 1),
             move)
            for move in self.board.availables)

        return move

    def adjacent_moves(self, board, player, plays):
        """
        adjacent moves without statistics info
        """
        moved = list(set(range(board.width * board.height)) - set(board.availables))
        adjacents = set()
        width = board.width
        height = board.height

        for m in moved:
            h = m // width
            w = m % width
            if w < width - 1:
                adjacents.add(m + 1) # right
            if w > 0:
                adjacents.add(m - 1) # left
            if h < height - 1:
                adjacents.add(m + width) # upper
            if h > 0:
                adjacents.add(m - width) # lower
            if w < width - 1 and h < height - 1:
                adjacents.add(m + width + 1) # upper right
            if w > 0 and h < height - 1:
                adjacents.add(m + width - 1) # upper left
            if w < width - 1 and h > 0:
                adjacents.add(m - width + 1) # lower right
            if w > 0 and h > 0:
                adjacents.add(m - width - 1) # lower left

        adjacents = list(set(adjacents) - set(moved))
        for move in adjacents:
            if plays.get((player, move)):
                adjacents.remove(move)
        return adjacents

    def has_a_winner(self, board):
        moved = list(set(range(board.width * board.height)) - set(board.availables))
        if(len(moved) < self.n_in_row + 2):
            return False, -1

        width = board.width
        height = board.height
        states = board.states
        n = self.n_in_row
        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                len(set(states[i] for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                len(set(states[i] for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                len(set(states[i] for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                len(set(states[i] for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def __str__(self):
        return "AI"


class Human(object):
    """
    human player
    """

    def __init__(self, board, player):
        self.board = board
        self.player = player

    def get_action(self):
        try:
            location = [int(n, 10) for n in input("Your move: ").split(",")]
            move = self.board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in self.board.availables:
            print("invalid move")
            move = self.get_action()
        return move

    def __str__(self):
        return "Human"


class Game(object):
    """
    game server
    """

    def __init__(self, board, model, **kwargs):
        self.board = board
        self.model = model
        self.player = [1, 2] # player1 and player2
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.time = float(kwargs.get('time', 5))
        self.max_actions = int(kwargs.get('max_actions', 1000))
        self.count = int(kwargs.get('count', -1))

    def start(self):
        p1, p2 = self.init_player()
        self.board.init_board()

        human = MCTS(self.board, self.model, 
                     [p1, p2], self.n_in_row, self.time, self.max_actions)
        ai = MCTS(self.board, self.model,
                  [p2,p1], self.n_in_row, self.time, self.max_actions)
        human.human = human.player
        human.ai = ai.player
        ai.human = human.player
        ai.ai = ai.player
        players = {}
        players[p1] = ai
        players[p2] = human
        turn = [p1, p2]
        shuffle(turn)
        self.graphic(self.board, human, ai)
        globalStates = []
        moves = []
        while(1):
            p = turn.pop(0)
            turn.append(p)
            player_in_turn = players[p]
            move = player_in_turn.get_action()
             
            tempState = self.printBoard(self.board, p, human, ai)
            globalStates.append(tempState)
            moves.append(move)
     
            self.board.update(p, move)
            self.graphic(self.board, human, ai)
            end, winner = self.game_end(ai)
            if end:
                if winner != -1:
                    print("Game end. Winner is", winner)
                    self.print_move_win(globalStates, moves, winner-1)
                break

    # format 144 input features, 1 move, 1 winner
    def print_move_win(self, globalStates, moves, winner):
        myfile = open(SAMPLE, 'a+')
        with myfile:
            writer = csv.writer(myfile)
            for i in range(len(globalStates)):
                temp = globalStates[i]
                temp.append(moves[i])
                temp.append(winner)
                writer.writerow(temp)
        print("{0} rows in file, count {1}".\
              format(sum(1 for row in csv.reader(open(SAMPLE))), self.count))

    def init_player(self):
        plist = list(range(len(self.player)))
        index1 = choice(plist)
        plist.remove(index1)
        index2 = choice(plist)

        return self.player[index1], self.player[index2]

    def game_end(self, ai):
        win, winner = ai.has_a_winner(self.board)
        if win:
            return True, winner
        elif not len(self.board.availables):
            print("Game end. Tie")
            return True, -1
        return False, -1

    def graphic(self, board, human, ai):
        """
        Draw the board and show game info
        """
        width = board.width
        height = board.height

        print("Human Player", human.player, "with X".rjust(3))
        print("AI    Player", ai.player, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                if board.states[loc] == human.player:
                    print('X'.center(8), end='')
                elif board.states[loc] == ai.player:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')
    
    # format 8x8*3 = 192 features
    def printBoard(self, board, p, human, ai):
        width = board.width
        height = board.height
        state = []
        for i in range(height-1, -1, -1):
            for j in range(width):
                loc = i * width + j
                if board.states[loc] == human.player:
                    state.append(1)
                    state.append(0)
                    state.append(0)
                elif board.states[loc] == ai.player:
                    state.append(0)
                    state.append(1)
                    state.append(0)
                else:
                    state.append(0)
                    state.append(0)
                    state.append(0)
        return state

def run():
    n = 5
    model = load_model(MODEL)
    for i in range(N_TIMES):
        try:
            board = Board(width=8, height=8, n_in_row=n)
            game = Game(board, model, n_in_row=n, time=1, count=i)
            game.start()
        except KeyboardInterrupt:
            print('\n\rquit')

if __name__ == '__main__':
    run()
