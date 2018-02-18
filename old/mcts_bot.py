from rd_model import Position
from rd_bots import RazzleDazzleBot
import datetime
from random import choice


class MCTS(RazzleDazzleBot):
    def __init__(self, name="MCTSBot", **kwargs ):
        RazzleDazzleBot.__init__(name)
        self.sates=[]
        seconds = kwargs.get('time', 30)
        self.calculation_time = datetime.timedelta(seconds=seconds)
        self.max_moves = kwargs.get('max_moves', 100)

        self.wins = {}
        self.plays = {}

    def update(self, position):
        # Takes a game state, and appends it to the history.
        self.state.append(position.state)

    def get_move(self, position):
        # Causes the AI to calculate the best move from the
        # current game state and return it.
        begin = datetime.datetime.utcnow()
        while datetime.datetime.utcnow() - begin < self.calculation_time:
            self.run_simulation()

    def run_simulation(self):
        # Plays out a "random" game from the current position,
        # then updates the statistics tables with the result.
        visited_states = set()
        states_copy = self.states[:]
        state = states_copy[-1]
        player = 1

        expand = True
        for t in range(self.max_moves):
            legal = Position(state).generate_moves()

            play = choice(legal)
            pos = pos.move(play)
            states_copy.append(pos)

            if expand and (player, pos) not in self.plays:
                expand = False
                self.plays[(player, pos)] = 0
                self.wins[(player, pos)] = 0

            visited_states.add((player, pos))

            winner = pos.value() == 1
            if winner:
                break