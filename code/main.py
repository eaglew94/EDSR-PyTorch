import torch

import utils
from option import args   #set the args in option.py
from data import data
from trainer import Trainer


if __name__ == '__main__':
    # set the random seed, so the later rand func can return reproducible results
    torch.manual_seed(args.seed)
    checkpoint = utils.checkpoint(args) # log related

    if checkpoint.ok:
        my_loader = data(args).get_loader() # init DataLoader
        t = Trainer(my_loader, checkpoint, args)
        while not t.terminate():
            t.train()
            t.test()

        checkpoint.done()

