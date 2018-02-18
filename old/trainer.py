import tqdm
import os.path

from rd_model import RazzleDazzleGame, Position
from rd_bots import PositionBot, PositionNet, IdiotBot
from slab import get_current_filename, get_next_filename
from slab.datamanagement import SlabFile

# from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random
import numpy as np


# Need to fix this, its a terrible algorithm
def itershuffle(iterable, bufsize=1000):
    """Shuffle an iterator. This works by holding `bufsize` items back
    and yielding them sometime later. This is NOT 100% random, proved or anything."""
    iterable = iter(iterable)
    buf = []
    try:
        while True:
            for i in range(random.randint(1, bufsize - len(buf))):
                buf.append(next(iterable))
            random.shuffle(buf)
            for i in range(random.randint(1, bufsize)):
                if buf:
                    yield buf.pop()
                else:
                    break
    except StopIteration:
        random.shuffle(buf)
        while buf:
            yield buf.pop()
        raise StopIteration


def generate_games(bot1, bot2, games, fname=None):
    """Generate lots of random games and save to text file"""
    for ii in tqdm.tqdm(range(games)):
        game = RazzleDazzleGame(bot1, bot2)

        game.play()
        if fname is not None:
            with open(fname, 'a') as f:
                f.write(str(game) + '\n')
        yield str(game)


def get_game_data(game_str):
    if game_str[-1] == '8':  # get value
        v = 1
    else:
        v = -1
    pos = Position(score=v)
    white_move = True
    for ms in game_str.split(','):
        m = pos.parse_move(ms)
        if not white_move:
            m = pos.rotate_move(m)

        yield [pos.state.flatten(), pos.get_piece_on_square(m[0]), m[-1], v]
        pos.move(m)
        pos.rotate()


def train_loader(bot1, bot2, games, training_fraction=1.0, fname=None):
    dataset = []
    for game_str in generate_games(bot1, bot2, games):
        for gd in get_game_data(game_str):
            dataset.append(gd)
    random.shuffle(dataset)
    dataset = np.transpose(dataset[:int(len(dataset) * training_fraction)])

    return dataset


def train_PositionBot(name="test", datapath="data/", epochs=2):
    args = {}
    args['batch_size'] = 64
    args['cuda'] = False
    args['epochs'] = 10
    args['lr'] = 0.01
    args['momentum'] = 0.5
    args['log_interval'] = 10

    fname = get_current_filename(datapath=datapath, prefix=name, suffix='.bot')

    bot1 = PositionBot(fname)
    bot2 = PositionBot(fname)
    model = bot1.posnet

    if args['cuda']:
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])

    for epoch in range(epochs):
        fname = get_current_filename(datapath=datapath, prefix=name, suffix='.bot')
        bot1 = PositionBot(fname)
        bot2 = PositionBot(fname)
        (pstates, start, stop, scores) = train_loader(bot1, bot2, games=3, training_fraction=0.5)
        model.train()
        if args['cuda']:
            pass
            # pstates, scores = data.cuda(), value.cuda()
        print(np.array(pstates).shape)
        data, value = Variable(torch.from_numpy(np.array(pstates,dtype=float)).float()), Variable(scores)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, value)
        loss.backward()
        optimizer.step()

        fname = get_next_filename(path="./data/", prefix=name, suffix='.bot')
        torch.save(model.state_dict(), fname)

        print("Train Epoch: %d/%d: %d positions\tLoss: %.6f" % (epoch,epochs,len(scores),loss.data[0]))


#
#
# def train(epoch):
#     model.train()
#     for batch_idx, (data, move, value) in enumerate(itershuffle(train_loader(args['batch_size']))):
#         if args['cuda']:
#             data, value = data.cuda(), value.cuda()
#         data, value = Variable(data), Variable(value)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, value)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % args['log_interval'] == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                        100. * batch_idx / len(train_loader), loss.data[0]))

train_PositionBot('test')
