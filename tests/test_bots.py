from rd_model import Position
from rd_bots import PositionBot


def test_PositionBot():
    pb = PositionBot('test')
    p = Position()

    moves = list(p.generate_moves())
    positions = [p.move(m).rotate() for m in moves]

    print moves
    print pb.eval_positions(positions)
    print pb.get_move(p)

if __name__ == "__main__":
    test_PositionBot()
