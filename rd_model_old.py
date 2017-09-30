import numpy as np
from collections import OrderedDict
from copy import deepcopy
from itertools import count


class RazzleDazzleModel:
    def __init__(self, position=None):
        self.position = position


def initial_board():
    player = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0,  # 0 - 8
             0, 0, 0, 0, 0, 0, 0, 0, 0,  # 9 - 17
             0, 1, 1, 1, 1, 1, 1, 1, 0,  # 18 - 26
             0, 0, 0, 0, 0, 0, 0, 0, 0,  # 27 - 35
             0, 0, 0, 0, 0, 0, 0, 0, 0,  # 36 - 44
             0, 0, 0, 0, 0, 0, 0, 0, 0,  # 45 - 53
             0, 0, 0, 0, 0, 0, 0, 0, 0,  # 54 - 62
             0, 0, 0, 0, 0, 0, 0, 0, 0,  # 63 - 71
             0, 0, 0, 0, 0, 0, 0, 0, 0,  # 72 - 80
             0, 0, 0, 0, 0, 0, 0, 0, 0,  # 81 - 89
             0, 0, 0, 0, 0, 0, 0, 0, 0,  # 90 - 98
             0, 0, 0, 0, 0, 0, 0, 0, 0  # 99 - 107
             ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 1, 1, 1, 1, 1, 1, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0
             ]
        ]
    )

    eligible = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 1, 1, 0, 1, 1, 1, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0
             ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 1, 1, 0, 1, 1, 1, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0
             ]])

    ball = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 1, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0
         ],
        [0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 1, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0
         ]])

    return Position(player, ball, eligible)


class Position2:
    N, E, S, W = 7, 1, -7, -1
    PDIRS = np.array(((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)), dtype=int)
    BDIRS = np.array(((-1, 0), (1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)), dtype=int)

    SQUARES = 56
    PIECES = 7
    ELIGIBLE = 1
    BALL = 1

    def __init__(self, state=None, score=0):
        if state is None:
            state = np.zeros((Position2.PIECES * 2, Position2.SQUARES + Position2.ELIGIBLE + Position2.BALL))
            for ii in xrange(Position2.PIECES):
                state[ii][ii] = 1.0  # White players on first row
                state[-1 - ii][Position2.SQUARES - ii - 1] = 1.0  # Black players on last row
                if ii != 3:
                    state[ii][-2] = 1.0  # All players eligible
                    state[-1 - ii][-2] = 1.0
                else:
                    state[ii][-1] = 1.0  # Middle players get the ball
                    state[-1 - ii][-1] = 1.0
            score = 0

        self.state = state
        self.score = score

    def rotate(self):
        state = self.state[::-1][:].copy().T
        state[:Position2.SQUARES] = state[Position2.SQUARES - 1::-1]
        return Position2(state.T, -self.score)

    def board_array(self):
        board = np.zeros(Position2.SQUARES)

        for ii in xrange(Position2.PIECES):
            board += self.state[ii][:Position2.SQUARES] * (ii + 1)
            board += self.state[-1 - ii][:Position2.SQUARES] * (Position2.PIECES + ii + 1)
        return board.astype(int)

    def is_legal(self, move):
        return move in self.generate_moves()

    def on_board(self, loc):
        return (loc[0] > -1) and (loc[0] < 7) and (loc[1] > -1) and (loc[1] < 8)

    def generate_passes(self, board, eligible, m):
        el2 = eligible.copy()
        if m[-1]==15:
            pass
        if len(m) > 1:
            yield m
        for d in Position2.BDIRS:
            el2[board[m[-1]]-1] = 0  # Set this piece as ineligible receiver for future passes
            loc = np.array([m[-1] % 7, m[-1] // 7], dtype=int)
            while True:
                loc += d
                if not self.on_board(loc): break
                b = board[loc[1] * 7 + loc[0]]
                if b > 0:  # If piece
                    if (b < 8) and (eligible[b - 1] > 0.5):  # If white and eligible
                        for move in self.generate_passes(board, el2, m + [loc[1] * 7 + loc[0]]):
                            yield move
                    break

    def generate_moves(self):
        board = self.board_array()
        for ii, p in enumerate(self.state[:Position2.PIECES]):
            ploc = np.argmax(p[:Position2.SQUARES])
            px, py = ploc % 7, ploc // 7
            if p[-1] < 0.5:  # doesn't have ball
                new_ps = np.array([px, py]) + Position2.PDIRS

                for newp in new_ps:
                    if self.on_board(newp):
                        if abs(board[newp[1] * 7 + newp[0]]) < .5:
                            yield (ploc, newp[1] * 7 + newp[0])
            else:  # has the ball
                eligible = self.state.T[-2][:Position2.PIECES]
                for move in self.generate_passes(board, eligible, [ploc]):
                    yield move

    def move(self, m):
        # type: (object) -> object
        f = m[0]
        t = m[-1]

        p = np.argmax(self.state[:, f])
        state = self.state.copy()

        if state[p, -1] > 0.5:  # If piece has the ball its a pass
            state[p, -1] = 0
            for ii, mi in enumerate(m):
                p2 = np.argmax(state[:, mi])
                state[p2, -2] = 0
            state[p2][-1] = 1
        else:
            state[p][t] = state[p][f]
            state[p][f] = 0
            state[p][-2] = 1
        return Position2(state)

    def __repr__(self):
        board = self.board_array()

        ans = ""
        rep = {0: "+",
               1: "x", 2: "X", 3: "W",
               -1: "o", -2: "O", -3: "B"}

        board = board.reshape((8, 7))
        for ii, row in enumerate(board[::-1]):
            ans += "%d  " % (board.shape[0] - ii)
            for s in row:
                # print -np.sign(s - Position2.PIECES) ,(1 + self.state[s][-2]),  (1+ 2*self.state[s][-1])
                ans += rep[-np.sign(s - Position2.PIECES - 0.5) * np.sign(s) * (1 + self.state[s - 1][-2]) * (
                    1 + 2 * self.state[s - 1][-1])] + " "
            ans += "\n"
        ans += ('\n   a b c d e f g \n\n')
        return ans


class Position:
    N, E, S, W = -9, 1, 9, -1
    Pdirections = (N + N + E, E + N + E, E + S + E, S + S + E, S + S + W, W + S + W, W + N + W, N + N + W)
    Bdirections = (N, S, E, W, N + E, N + W, S + E, S + W)

    def __init__(self, player, ball, eligible, score=0):
        self.player = player
        self.ball = ball
        self.eligible = eligible
        self.score = score

    def rotate(self):
        return Position(self.player[[1, 0], ::-1].copy(), self.ball[[1, 0], ::-1].copy(),
                        self.eligible[[1, 0], ::-1].copy(),
                        -self.score)

    def move(self, move):
        # type: (object) -> object
        np = deepcopy(self)

        p = move[0]
        q = move[1]

        if np.player[0][p] == 1:
            jj = 0
        elif np.player[1][p] == 1:
            jj = 1
        else:
            raise Exception('Illegal move')

        if (np.ball[jj][p] == 1):
            # if throwing the ball
            for ii in move:  # render players along path ineligible
                np.eligible[jj][ii] = 0

            np.ball[jj][move[-1]] = np.ball[jj][p]  # move the ball
            np.player[jj][p] = np.ball[jj][p]
            np.ball[jj][p] = 0

        else:
            # If its player moving (not the ball)
            np.player[jj][q] = np.player[jj][p]
            np.eligible[jj][q] = np.player[jj][q]
            np.player[jj][p] = 0
            np.eligible[jj][p] = 0
        return np

    def valid(self, ind):
        return ind >= 17 and ind <= 90 and ind % 9 != 0 and ind % 9 != 8

    def eligible_receivers(self, p, eligible):
        for d in Position.Bdirections:
            for kk in count(p + d, d):
                if self.player[1][kk] == 1 or eligible[kk] == 0 or not self.valid(kk): break
                if eligible[kk] == 1:
                    yield kk
                    break

    def generate_passes(self, move, eligible):
        for r in self.eligible_receivers(move[-1], eligible):
            el2 = eligible.copy()
            el2[r] = 0
            yield move + [r]
            for m in self.generate_passes(move + [r], el2):
                yield m

    def generate_moves(self):
        for ii in xrange(2, 8 + 4 - 2):
            for jj in xrange(1, 7 + 2 - 1):
                ind = ii * (7 + 2) + jj
                if self.player[0][ind] != 1: break
                if self.ball[0][ind] == 1:
                    for move in self.generate_passes([ind], self.eligible[0]):
                        yield move
                else:
                    for d in Position.Pdirections:
                        if self.player[0][ind + d] != 1 and self.player[1][ind + d] != 1:
                            if self.valid(ind + d):
                                yield [ind, ind + d]

    def is_legal(self, move):
        return move in self.generate_moves()

    def __repr__(self):
        ans = ""
        for ii in xrange(8 + 4 - 3, 1, -1):
            ans += "%d  " % (ii - 1)
            for jj in xrange(1, 7 + 2 - 1):
                ind = ii * (7 + 2) + jj
                if self.ball[0][ind] == 1:
                    ans += "W "
                elif self.ball[1][ind] == 1:
                    ans += "B "
                elif self.player[0][ind] == 1:
                    if self.eligible[0][ind] == 1:
                        ans += "X "
                    elif self.eligible[1][ind] == 0:
                        ans += "x "
                elif self.player[1][ind] == 1:
                    if self.eligible[1][ind] == 1:
                        ans += "O "
                    elif self.eligible[1][ind] == 0:
                        ans += "o "
                else:
                    ans += "+ "
            ans += "\n"
        ans += ('\n   a b c d e f g \n\n')
        return ans


class IdiotBot:
    def __init__(self):
        pass


# The normal OrderedDict doesn't update the position of a key in the list,
# when the value is changed.
class LRUCache:
    '''Store items in the order the keys were last added'''

    def __init__(self, size):
        self.od = OrderedDict()
        self.size = size

    def get(self, key, default=None):
        try:
            self.od.move_to_end(key)
        except KeyError:
            return default
        return self.od[key]

    def __setitem__(self, key, value):
        try:
            del self.od[key]
        except KeyError:
            if len(self.od) == self.size:
                self.od.popitem(last=False)
        self.od[key] = value


if __name__ == "__main__":
    # pos = initial_board()
    # print pos
    #
    # for ii, move in enumerate(pos.generate_moves()):
    #     print ii, move
    #     print pos.move(move)
    #     if ii == 7: my_move = move
    #
    # print pos.move(my_move)
    # print pos.move(my_move).rotate()
    pos = Position2()
    # print pos.state
    print pos

    m1=(2,15)
    m2=(4,17)

    pos=pos.move(m1).move(m2)


    for ii, m in enumerate(pos.generate_moves()):
        print ii, ": ", m
        print pos.move(m)
