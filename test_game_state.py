from game_state import *

gs= RazzleDazzleState()

print ( int(-2 * (gs.playerJustMoved - 1.5)))

for move in gs.GetMoves():
    print (move)
    gsp=gs.Clone()
    gsp.DoMove(move)
    print(gsp)