# This is a very simple implementation of the UCT Monte Carlo Tree Search algorithm in Python 2.7.
# The function UCT(rootstate, itermax, verbose = False) is towards the bottom of the code.
# It aims to have the clearest and simplest possible code, and for the sake of clarity, the code
# is orders of magnitude less efficient than it could be made, particularly by using a
# state.GetRandomMove() or state.DoRandomRollout() function.
#
# Example GameState classes for Nim, OXO and Othello are included to give some idea of how you
# can write your own GameState use UCT in your 2-player game. Change the game to be played in
# the UCTPlayGame() function at the bottom of the code.
#
# Written by Peter Cowling, Ed Powley, Daniel Whitehouse (University of York, UK) September 2012.
#
# Licence is granted to freely use and distribute for any sensible/legal purpose so long as this comment
# remains in any distributed code.
#
# For more information about Monte Carlo Tree Search check out our web site at www.mcts.ai

from math import *
import random
from game_state import *
import time


class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
        Crashes if state not specified.
    """
    def __init__(self, move = None, parent = None, state = None):
        self.move = move # the move that got us to this node - "None" for the root node
        self.parentNode = parent # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untriedMoves = state.GetMoves() # future child nodes
        self.playerJustMoved = state.playerJustMoved # the only part of the state that the Node needs later

    def UCTSelectChild(self):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        s = sorted(self.childNodes, key = lambda c: c.wins/c.visits + sqrt(2*log(self.visits)/c.visits))[-1]
        return s

    def AddChild(self, m, s):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = Node(move = m, parent = self, state = s)
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n

    def Update(self, result):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.wins += result

    def __repr__(self):
        return "[M:%s W/V/+-/dev: %d / %d / %0.2f / %0.2f U: %s]" % (str(self.move), self.wins, self.visits, self.wins/self.visits-0.5, sqrt(self.visits)/self.visits,str(self.untriedMoves))
            #"[M:" + str(self.move) + " W/V/%/dev:" + str(self.wins) + "/" + str(self.visits) +"/"+ str(self.wins/self.visits) + " U:" + str(self.untriedMoves) + "]"

    def TreeToString(self, indent):
        s = self.IndentString(indent) + str(self)
        for c in self.childNodes:
             s += c.TreeToString(indent+1)
        return s

    def IndentString(self,indent):
        s = "\n"
        for i in range (1,indent+1):
            s += "| "
        return s

    def ChildrenToString(self):
        s = ""
        for c in self.childNodes:
             s += str(c) + "\n"
        return s


def UCT(rootstate, itermax, verbose = False):
    """ Conduct a UCT search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
        Assumes 2 alternating players (player 1 starts), with game results in the range [0.0, 1.0]."""

    rootnode = Node(state = rootstate)
    total_moves=0
    start_time = time.time()
    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        # Select
        while node.untriedMoves == [] and node.childNodes != []: # node is fully expanded and non-terminal
            node = node.UCTSelectChild()
            state.DoMove(node.move)
            total_moves+=1

        # Expand
        if node.untriedMoves != []: # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.untriedMoves)
            state.DoMove(m)
            total_moves += 1
            node = node.AddChild(m,state) # add child and descend tree

        # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
        # moves =state.GetMoves()
        # while moves != []: # while state is non-terminal
        #     state.DoMove(random.choice(moves))
        #     moves = state.GetMoves()
        #     total_moves += 1

        # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
        move = state.GetRandomMove()
        while move is not None:
            state.DoMove(move)
            total_moves+=1
            move=state.GetRandomMove()

        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            node.Update(state.GetResult(node.playerJustMoved)) # state is terminal. Update node with result from POV of node.playerJustMoved
            node = node.parentNode
    stop_time=time.time()
    # Output some information about the tree - can be omitted
    if (verbose): print(rootnode.TreeToString(0))
    else: print(rootnode.ChildrenToString())
    print ("Moves searched: %d\t Time taken: %f s\t %f moves/sec" %(total_moves,stop_time-start_time,total_moves/(stop_time-start_time)) )

    return sorted(rootnode.childNodes, key = lambda c: c.visits)[-1].move # return the move that was most visited

def UCTPlayGame():
    """ Play a sample game between two UCT players where each player gets a different number
        of UCT iterations (= simulations = tree nodes).
    """
    #state = OthelloState(8) # uncomment to play Othello on a square board of the given size
    # state = OXOState() # uncomment to play OXO
    #state = NimState(15) # uncomment to play Nim with the given number of starting chips
    state = RazzleDazzleState(rows=8,cols=7)
    move_list = []
    while (state.GetMoves() != []):
        print(str(state))

        if state.playerJustMoved == 1:
            m = UCT(rootstate = state, itermax = 100000, verbose = False) # play with values for itermax and verbose = True
            #m= random.choice(state.GetMoves())
        else:
            #m = UCT(rootstate = state, itermax = 1000, verbose = False)

            #m= random.choice(state.GetMoves())

            while True:
                try:
                    s=input("Input move: ")
                    m = state.parse_move(s)
                    if m in state.GetMoves():
                        break
                except:
                    pass
        state.DoMove(m)
        move_list.append(m)
        print("Move %d\t Player %d\t Best Move: %s\n" % (len(move_list),state.playerJustMoved, state.move_str(m) ))
    if state.GetResult(state.playerJustMoved) == 1.0:
        print("Player " + str(state.playerJustMoved) + " wins!")
    elif state.GetResult(state.playerJustMoved) == 0.0:
        print("Player " + str(3 - state.playerJustMoved) + " wins!")
    else: print("Nobody wins!")

    print (str(state))
    #print ("%d Xs\t %d Os" % (str(state).count("X"),str(state).count("O")))
    # for m in move_list:
    #     #print(state.move_str(m))
    #     print (str(m))

if __name__ == "__main__":
    """ Play a single game to the end using UCT for both players.
    """
    UCTPlayGame()





