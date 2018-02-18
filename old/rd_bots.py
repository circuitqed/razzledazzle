import numpy as np
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class RazzleDazzleBot:
    def __init__(self, name):
        self.name = name

    def get_move(self, position):
        pass


class MiniMaxBot(RazzleDazzleBot):
    def __init__(self, name="MiniMaxBot"):
        self.name = name

        # implement





class IdiotBot(RazzleDazzleBot):
    def __init__(self, name="Idiot"):
        self.name = name

    def get_move(self, position):
        moves = list(position.generate_moves())
        return moves[np.random.randint(len(moves))]


class TurkBot(RazzleDazzleBot):
    def __init__(self, name="TurkBot"):
        self.name = name

    def get_move(self, position):
        print (position)
        legal = False
        while True:
            try:
                m = position.parse_move(input("%s move: " % self.name))
            except:
                print ("Illegal move!")
            print (m)
            if position.is_legal(m):
                break
            else:
                print("Illegal move!")
        return m


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(10 * (56 + 2), 10 * (56 + 2))  # an affine operation: y = Wx + b
        self.fc2 = nn.Linear(10 * (56 + 2), 5 * 56)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)


class OldPositionNet(nn.Module):
    def __init__(self):
        super(PositionNet, self).__init__()
        self.fc1 = nn.Linear(10 * (56 + 2), 10 * (56 + 2))  # an affine operation: y = Wx + b
        self.fc2 = nn.Linear(10 * (56 + 2), 5 * (56 + 2))
        self.fc3 = nn.Linear(5 * (56 + 2), 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        F.dropout(x, training=self.training)
        x = F.relu(self.fc4(x))
        F.dropout(x, training=self.training)
        return F.tanh(self.fc5(x))

class PositionNet(nn.Module):
    def __init__(self):
        super(PositionNet, self).__init__()
        #self.fc1 = nn.Linear(10 * (56 + 2), 10 * (56 + 2))  # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(10 * (56 + 2), 5 * (56 + 2))
        self.fc2 = nn.Linear(5 * (56 + 2), 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        F.dropout(x, training=self.training)
        #x = F.relu(self.fc3(x))
        #F.dropout(x, training=self.training)
        return F.tanh(self.fc3(x))

class PositionBot(RazzleDazzleBot):
    """Position bot does a 1-ply minimax using its PositionNet to evaluate the board position
       It generates a list of legal moves,
       and then returns the worst one from perspective of the opposing player"""

    def __init__(self, name="Position", path=''):
        self.name = name
        self.path = path
        self.selectivity = 2

        try:
            if name[-4:] != ".bot":
                name+=".bot"
            self.posnet = PositionNet()
            self.posnet.load_state_dict(torch.load(os.path.join(self.path, name)))
        except:
            print("Failed to load: " + os.path.join(self.path, name))
            print("Loading new PositionNet model")
            self.posnet = PositionNet()

    def get_move(self, position):
        moves = list(position.generate_moves())
        positions = [position.move(m).rotate() for m in moves]
        weights = (1.-self.eval_positions(positions))/2.
        weights = weights/np.sum(weights)
        if np.random.random() > 0.90:
            m = np.random.choice(list(range(len(moves))),p=weights)
        else:
            m = np.argmax(weights)

        return moves[m]

    def eval_position(self, position):
        eval = self.posnet(Variable(torch.from_numpy(np.array([position.state.flatten()])).float()))
        return eval.data.numpy()[0][0]

    def eval_positions(self, positions):
        states = np.array([p.state.flatten() for p in positions])
        eval = self.posnet(Variable(torch.from_numpy(states).float()))
        return eval.data.numpy().flatten()

    def save(self):
        torch.save(self.posnet.state_dict(), os.path.join(self.path, self.name + ".bot"))
