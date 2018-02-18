import numpy as np
import random


class GameState:
    """ A state of the game, i.e. the game board. These are the only functions which are
        absolutely necessary to implement UCT in any 2-player complete information deterministic
        zero-sum game, although they can be enhanced and made quicker, for example by using a
        GetRandomMove() function to generate a random move during rollout.
        By convention the players are numbered 1 and 2.
    """

    def __init__(self):
        self.playerJustMoved = 2  # At the root pretend the player just moved is player 2 - player 1 has the first move

    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = GameState()
        st.playerJustMoved = self.playerJustMoved
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerJustMoved.
        """
        self.playerJustMoved = 3 - self.playerJustMoved

    def GetMoves(self):
        """ Get all possible moves from this state.
        """

    def GetRandomMove(self):
        moves = self.GetMoves()
        if moves == []:
            return None
        else:
            return random.choice(moves)

    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm.
        """

    def __repr__(self):
        """ Don't need this - but good style.
        """
        pass


class RazzleDazzleState:
    """ A state of the game, i.e. the game board. These are the only functions which are
        absolutely necessary to implement UCT in any 2-player complete information deterministic
        zero-sum game, although they can be enhanced and made quicker, for example by using a
        GetRandomMove() function to generate a random move during rollout.
        By convention the players are numbered 1 and 2.
    """

    BDIRS = np.array(((-1, 0), (1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)), dtype=int)

    def __init__(self, board=None, num_moves=0, pieces=None, just_moved=2, rows=8, cols=7):
        self.playerJustMoved = just_moved  # At the root pretend the player just moved is player 2 - player 1 has the first move
        self.num_moves = num_moves
        if board is None:
            self.rows = rows
            self.cols = cols
            self.board = np.zeros((self.rows, self.cols), dtype=int)
            self.board[0, 1:self.cols - 1] = 1
            self.board[-1, 1:self.cols - 1] = -1
            self.board[0, self.cols // 2] = 2
            self.board[-1, self.cols // 2] = -2
            self.pieces = [np.transpose([np.zeros(cols - 1, dtype=int), np.arange(1, self.cols, dtype=int)]),
                           np.transpose(
                               [(self.rows - 1) * np.ones(cols - 1, dtype=int), np.arange(1, self.cols, dtype=int)])]
        else:
            self.board = board.copy()
            self.rows, self.cols = self.board.shape
            # if pieces is not None:
            #     self.pieces = pieces.copy()
            # else:
            #     self.pieces = self.find_pieces()

    def find_pieces(self):
        pieces = [[], []]
        for row in range(self.rows):
            for col in range(self.cols):
                if self.board[row, col] > 0.5:
                    pieces[0].append([row, col])
                elif self.board[row, col] < -0.5:
                    pieces[1].append([row, col])
        return pieces

    def Clone(self):
        """ Create a deep clone of this game state.
        """

        return RazzleDazzleState(board=self.board, num_moves=self.num_moves, pieces=self.pieces,
                                 just_moved=self.playerJustMoved)

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerJustMoved.
        """
        self.playerJustMoved = 3 - self.playerJustMoved
        self.num_moves += 1
        # p=self.playerJustMoved-1
        # print ("index for player in domove %d / %s" % (p, str(move)))
        # if abs(self.board[move[0]]) < 1.5:
        #     idx = -1
        #     for ii in range(len(self.pieces[p])):
        #         if move[0] == (self.pieces[p][ii][0], self.pieces[p][ii][1]):
        #             idx = ii
        #     if not idx > -1:
        #         print (p)
        #         print (self.pieces)
        #         print (move)
        #         print (self)
        #         print (self.move_str(move))
        #         assert False
        #     self.pieces[p][idx] = move[-1]
        ##self.pieces=self.find_pieces()

        old = self.board[move[-1]]
        self.board[move[-1]] = self.board[move[0]]
        self.board[move[0]] = old

    def on_board(self, pos):
        return (pos[0] >= 0) and (pos[0] < self.rows) and (pos[1] >= 0) and (pos[1] < self.cols)

    def move_str(self, m):
        return '-'.join([chr(mi[1] + ord('a')) + chr(mi[0] + ord('1')) for mi in m])

    def get_square(self, c):
        return (int(c[1]) - 1, ord(c[0].upper()) - ord('A'))

    def parse_move(self, move_string):
        return [self.get_square(c) for c in move_string.split('-')]

    def generate_passes(self, m):
        moves = []
        p = self.PlayerSign()
        for dir in self.BDIRS:
            loc = m[-1]
            while True:
                loc = (loc[0] + dir[0], loc[1] + dir[1])
                if not self.on_board(loc): break
                b = self.board[loc]
                if b != 0:
                    if (b == p) and (loc not in m):
                        moves.append(m + [loc])
                        moves += self.generate_passes(m + [loc])
                    break
        return moves

    def GameOver(self):
        return self.num_moves > 50 or (2 in self.board[-1]) or (-2 in self.board[0])

    def PlayerSign(self):
        return int(2 * (self.playerJustMoved - 1.5))

    def PieceMoves(self, pos):
        moves = []
        dests = [(pos[0] + 2, pos[1] + 1), (pos[0] + 2, pos[1] - 1), (pos[0] + 1, pos[1] - 2), (pos[0] + 1, pos[1] + 2),
                 (pos[0] - 1, pos[1] - 2), (pos[0] - 1, pos[1] + 2), (pos[0] - 2, pos[1] - 1), (pos[0] - 2, pos[1] + 1)]
        for d in dests:
            if self.on_board(d) and self.board[d] == 0: moves.append([pos, d])
        return moves

    def GetMoves(self, include_moves=True, include_passes=True):
        """ Get all possible moves from this state.
        """
        if self.GameOver(): return []
        moves = []
        p = self.PlayerSign()
        for row in range(self.rows):
            for col in range(self.cols):
                if include_passes and (self.board[row, col] == 2 * p):
                    moves += self.generate_passes([(row, col)])
                elif include_moves and (self.board[row, col] == p):
                    moves += self.PieceMoves((row, col))
        return moves

    def GetRandomMove(self):
        pass_move = random.random() > 0.5
        moves = self.GetMoves(include_passes=pass_move, include_moves=not pass_move)
        if moves == []:
            moves = self.GetMoves(include_passes=not pass_move, include_moves=pass_move)
        if moves == []:
            return None
        else:
            return random.choice(moves)

        # player = 2 - self.playerJustMoved
        # piece = random.randint(0,self.cols - 3)
        # #print (self.pieces,player,piece)
        # pos = (self.pieces[player][piece][0],self.pieces[player][piece][1])
        # #print ("index for player in GetRandomMove %d / %s" % (player, str(pos)))
        # if abs(self.board[pos[0], pos[1]]) > 1.5:  # if piece has the ball
        #     moves = self.generate_passes([(pos[0], pos[1])])
        # else:
        #     moves = self.PieceMoves(pos)
        # if moves == []:
        #     moves = self.GetMoves()
        #     #print ("Reverting to getmoves")
        # if moves == []:
        #     return None
        # else:
        #     return random.choice(moves)

    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm.
        """
        if self.num_moves > 50:
            return 0.5
        p = int(2 * (playerjm - 1.5))
        # if not self.GameOver():
        #     print (self)
        #     assert False
        if 2 in self.board[-1]:
            r = -1.0
        elif -2 in self.board[0]:
            r = 1.0
        else:
            r = 0.0
        # else:
        #     print("Error no winner: ")
        #     print(self)
        #     print(p)
        #     print(self.board[0])
        #     print(self.board[-1])
        #     print("")

        if p * r > 0:
            return 1.0
        elif p * r < 0:
            return 0.0
        else:
            return 0.5

    def __repr__(self):
        """ Don't need this - but good style.
        """
        rep = {0: "+", 1: "x", 2: "X", -1: "o", -2: "O"}

        ans = ""
        for row in range(self.rows - 1, -1, -1):
            ans += str(row + 1) + " "
            for col in range(self.cols):
                ans += rep[self.board[row, col]] + " "
            ans += "\n"
        ans += "  " + " ".join([chr(ord('a') + ii) for ii in range(self.cols)])
        ans += "\n"
        return ans


class NimState:
    """ A state of the game Nim. In Nim, players alternately take 1,2 or 3 chips with the
        winner being the player to take the last chip.
        In Nim any initial state of the form 4n+k for k = 1,2,3 is a win for player 1
        (by choosing k) chips.
        Any initial state of the form 4n is a win for player 2.
    """

    def __init__(self, ch):
        self.playerJustMoved = 2  # At the root pretend the player just moved is p2 - p1 has the first move
        self.chips = ch

    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = NimState(self.chips)
        st.playerJustMoved = self.playerJustMoved
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerJustMoved.
        """
        assert move >= 1 and move <= 3 and move == int(move)
        self.chips -= move
        self.playerJustMoved = 3 - self.playerJustMoved

    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        return range(1, min([4, self.chips + 1]))

    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm.
        """
        assert self.chips == 0
        if self.playerJustMoved == playerjm:
            return 1.0  # playerjm took the last chip and has won
        else:
            return 0.0  # playerjm's opponent took the last chip and has won

    def __repr__(self):
        s = "Chips:" + str(self.chips) + " JustPlayed:" + str(self.playerJustMoved)
        return s


class OXOState:
    """ A state of the game, i.e. the game board.
        Squares in the board are in this arrangement
        012
        345
        678
        where 0 = empty, 1 = player 1 (X), 2 = player 2 (O)
    """

    def __init__(self):
        self.playerJustMoved = 2  # At the root pretend the player just moved is p2 - p1 has the first move
        self.board = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # 0 = empty, 1 = player 1, 2 = player 2

    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = OXOState()
        st.playerJustMoved = self.playerJustMoved
        st.board = self.board[:]
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerToMove.
        """
        assert move >= 0 and move <= 8 and move == int(move) and self.board[move] == 0
        self.playerJustMoved = 3 - self.playerJustMoved
        self.board[move] = self.playerJustMoved

    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        return [i for i in range(9) if self.board[i] == 0]

    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm.
        """
        for (x, y, z) in [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]:
            if self.board[x] == self.board[y] == self.board[z]:
                if self.board[x] == playerjm:
                    return 1.0
                else:
                    return 0.0
        if self.GetMoves() == []: return 0.5  # draw
        assert False  # Should not be possible to get here

    def __repr__(self):
        s = ""
        for i in range(9):
            s += ".XO"[self.board[i]]
            if i % 3 == 2: s += "\n"
        return s


class OthelloState:
    """ A state of the game of Othello, i.e. the game board.
        The board is a 2D array where 0 = empty (.), 1 = player 1 (X), 2 = player 2 (O).
        In Othello players alternately place pieces on a square board - each piece played
        has to sandwich opponent pieces between the piece played and pieces already on the
        board. Sandwiched pieces are flipped.
        This implementation modifies the rules to allow variable sized square boards and
        terminates the game as soon as the player about to move cannot make a move (whereas
        the standard game allows for a pass move).
    """

    def __init__(self, sz=8):
        self.playerJustMoved = 2  # At the root pretend the player just moved is p2 - p1 has the first move
        self.board = []  # 0 = empty, 1 = player 1, 2 = player 2
        self.size = sz
        assert sz == int(sz) and sz % 2 == 0  # size must be integral and even
        for y in range(sz):
            self.board.append([0] * sz)
        self.board[sz // 2][sz // 2] = self.board[sz // 2 - 1][sz // 2 - 1] = 1
        self.board[sz // 2][sz // 2 - 1] = self.board[sz // 2 - 1][sz // 2] = 2

    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = OthelloState()
        st.playerJustMoved = self.playerJustMoved
        st.board = [self.board[i][:] for i in range(self.size)]
        st.size = self.size
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerToMove.
        """
        (x, y) = (move[0], move[1])
        assert x == int(x) and y == int(y) and self.IsOnBoard(x, y) and self.board[x][y] == 0
        m = self.GetAllSandwichedCounters(x, y)
        self.playerJustMoved = 3 - self.playerJustMoved
        self.board[x][y] = self.playerJustMoved
        for (a, b) in m:
            self.board[a][b] = self.playerJustMoved

    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        return [(x, y) for x in range(self.size) for y in range(self.size) if
                self.board[x][y] == 0 and self.ExistsSandwichedCounter(x, y)]

    def AdjacentToEnemy(self, x, y):
        """ Speeds up GetMoves by only considering squares which are adjacent to an enemy-occupied square.
        """
        for (dx, dy) in [(0, +1), (+1, +1), (+1, 0), (+1, -1), (0, -1), (-1, -1), (-1, 0), (-1, +1)]:
            if self.IsOnBoard(x + dx, y + dy) and self.board[x + dx][y + dy] == self.playerJustMoved:
                return True
        return False

    def AdjacentEnemyDirections(self, x, y):
        """ Speeds up GetMoves by only considering squares which are adjacent to an enemy-occupied square.
        """
        es = []
        for (dx, dy) in [(0, +1), (+1, +1), (+1, 0), (+1, -1), (0, -1), (-1, -1), (-1, 0), (-1, +1)]:
            if self.IsOnBoard(x + dx, y + dy) and self.board[x + dx][y + dy] == self.playerJustMoved:
                es.append((dx, dy))
        return es

    def ExistsSandwichedCounter(self, x, y):
        """ Does there exist at least one counter which would be flipped if my counter was placed at (x,y)?
        """
        for (dx, dy) in self.AdjacentEnemyDirections(x, y):
            if len(self.SandwichedCounters(x, y, dx, dy)) > 0:
                return True
        return False

    def GetAllSandwichedCounters(self, x, y):
        """ Is (x,y) a possible move (i.e. opponent counters are sandwiched between (x,y) and my counter in some direction)?
        """
        sandwiched = []
        for (dx, dy) in self.AdjacentEnemyDirections(x, y):
            sandwiched.extend(self.SandwichedCounters(x, y, dx, dy))
        return sandwiched

    def SandwichedCounters(self, x, y, dx, dy):
        """ Return the coordinates of all opponent counters sandwiched between (x,y) and my counter.
        """
        x += dx
        y += dy
        sandwiched = []
        while self.IsOnBoard(x, y) and self.board[x][y] == self.playerJustMoved:
            sandwiched.append((x, y))
            x += dx
            y += dy
        if self.IsOnBoard(x, y) and self.board[x][y] == 3 - self.playerJustMoved:
            return sandwiched
        else:
            return []  # nothing sandwiched

    def IsOnBoard(self, x, y):
        return x >= 0 and x < self.size and y >= 0 and y < self.size

    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm.
        """
        jmcount = len([(x, y) for x in range(self.size) for y in range(self.size) if self.board[x][y] == playerjm])
        notjmcount = len(
            [(x, y) for x in range(self.size) for y in range(self.size) if self.board[x][y] == 3 - playerjm])
        if jmcount > notjmcount:
            return 1.0
        elif notjmcount > jmcount:
            return 0.0
        else:
            return 0.5  # draw

    def __repr__(self):
        s = ""
        for y in range(self.size - 1, -1, -1):
            for x in range(self.size):
                s += ".XO"[self.board[x][y]]
            s += "\n"
        return s
