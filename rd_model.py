import numpy as np
from collections import OrderedDict

class Position:
    N, E, S, W = 7, 1, -7, -1
    PDIRS = np.array(((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)), dtype=int)
    BDIRS = np.array(((-1, 0), (1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)), dtype=int)

    SQUARES = 56
    PIECES = 5
    ELIGIBLE = 1
    BALL = 1

    def __init__(self, state=None, score=0):
        if state is None:
            state = np.zeros((Position.PIECES * 2, Position.SQUARES + Position.ELIGIBLE + Position.BALL))
            for ii in xrange(Position.PIECES):
                state[ii][ii+1] = 1.0  # White players on first row
                state[-1 - ii][Position.SQUARES - ii - 2] = 1.0  # Black players on last row
                if ii != 2:
                    state[ii][-2] = 1.0  # All players eligible
                    state[-1 - ii][-2] = 1.0
                else:
                    state[ii][-1] = 1.0  # Middle players get the ball
                    state[-1 - ii][-1] = 1.0
            score = 0

        self.state = state
        self.score = score
        self.next_move = None

    def rotate(self):
        state = np.roll(self.state, Position.PIECES, axis=0)  # Swap black and white pieces
        state[:, :Position.SQUARES] = np.fliplr(state[:, :Position.SQUARES])
        return Position(state, -self.score)

    def board_array(self):
        board = np.zeros(Position.SQUARES)

        for ii in xrange(len(self.state)):
            board += self.state[ii][:Position.SQUARES] * (ii + 1)
        return board.astype(int)

    def is_legal(self, move):
        try:
            list(self.generate_moves()).index(move)
            return True
        except:
            return False

    def on_board(self, loc):
        return (loc[0] > -1) and (loc[0] < 7) and (loc[1] > -1) and (loc[1] < 8)

    def generate_passes(self, board, eligible, m):
        el2 = eligible.copy()
        if m[-1] == 15:
            pass
        if len(m) > 1:
            yield m
        for d in Position.BDIRS:
            el2[board[m[-1]] - 1] = 0  # Set this piece as ineligible receiver for future passes
            loc = np.array([m[-1] % 7, m[-1] // 7], dtype=int)
            while True:
                loc += d
                if not self.on_board(loc): break
                b = board[loc[1] * 7 + loc[0]]
                if b > 0:  # If piece
                    if (b < 6) and (eligible[b - 1] > 0.5):  # If white and eligible
                        for move in self.generate_passes(board, el2, m + [loc[1] * 7 + loc[0]]):
                            yield move
                    break

    def generate_moves(self):
        board = self.board_array()
        for ii, p in enumerate(self.state[:Position.PIECES]):
            ploc = np.argmax(p[:Position.SQUARES])
            px, py = ploc % 7, ploc // 7
            if p[-1] < 0.5:  # doesn't have ball
                new_ps = np.array([px, py]) + Position.PDIRS

                for newp in new_ps:
                    if self.on_board(newp):
                        if abs(board[newp[1] * 7 + newp[0]]) < .5:
                            yield [ploc, newp[1] * 7 + newp[0]]
            else:  # has the ball
                eligible = self.state.T[-2][:Position.PIECES]
                for move in self.generate_passes(board, eligible, [ploc]):
                    yield move

    def value(self):
        b1, b2 = self.find_balls()
        if self.find_piece(b1) > Position.SQUARES - 7:
            return 1
        elif self.find_piece(b2) < 7:
            return -1
        else:
            return 0

    def get_piece_on_square(self, loc):
        return np.argmax(self.state[:, loc])

    def find_piece(self, p):
        return np.argmax(self.state[p, :Position.SQUARES])

    def find_balls(self):
        return np.argmax(self.state[:Position.PIECES, -1]), Position.PIECES + np.argmax(
            self.state[Position.PIECES:, -1])

    def move(self, m):
        # type: (object) -> object
        f = m[0]
        t = m[-1]
        self.next_move = m

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
        return Position(state)

    def rotate_move(self, m):
        return [7 * 8 - n for n in m]

    def move_str(self, m):
        return '-'.join([chr((mi % 7) + ord('a')) + chr((mi // 7) + ord('1')) for mi in m])

    def get_square(self, c):
        return (ord(c[0].upper()) - ord('A')) + (int(c[1]) - 1) * 7

    def parse_move(self, move_string):
        return [self.get_square(c) for c in move_string.split('-')]

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
                ans += rep[-np.sign(s - Position.PIECES - 0.5) * np.sign(s) * (1 + self.state[s - 1][-2]) * (
                    1 + 2 * self.state[s - 1][-1])] + " "
            ans += "\n"
        ans += ('\n   a b c d e f g \n\n')
        return ans

class RazzleDazzleGame:
    def __init__(self, player1, player2, position=None):
        if position is None: position = Position()
        self.position = position
        self.player1 = player1
        self.player2 = player2
        self.moves = []

    def get_position(self):
        return self.position

    def play(self):
        turn = -1
        while abs(self.position.value()) < 0.5:
            turn *= -1
            if turn == 1:
                m = self.player1.get_move(self.position)
                self.position = self.position.move(m)
                self.moves.append(self.position.move_str(m))
            else:
                position = self.position.rotate()
                m = self.player2.get_move(position)
                self.position = position.move(m).rotate()
                self.moves.append(self.position.move_str(self.position.rotate_move(m)))
        self.winner = turn

    def __repr__(self):
        return ','.join(self.moves)

    def __len__(self):
        return len(self.moves)


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
    pos = Position()
    print pos
    a = list(pos.generate_moves())
    print len(a)
    for m in a:
        print pos.move(m)

    #m1 = [1, 16]
    #m2 = [3, 2, 16]

    #pos = pos.move(m1).move(m2)
