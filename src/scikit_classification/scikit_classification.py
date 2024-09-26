import argparse
import torch

from classes.Datasets import DatasetforVirtualData
from classes.Dataloaders import DataLoader, data_spliter
from classes.Models import NN
from classes.Trainer import Trainer
from classes.Utils import get_device, str2loss

def get_argparse():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--loss', default='dims', type=str2loss, help="Enter the loss functions : mse, ce, mae, dims, adims, dima, adima.")
    parser.add_argument('--label', default=10, type=int, help='Type the number of labels to seperate.')
    parser.add_argument('--alpha', default=2, type=float, help="Enter the alpha hyper-parameter of distance loss functions.")
    parser.add_argument('--ratio', default=0.8, type=float, help='Type the ratio of training data')
    parser.add_argument('--seed', default=0, type=int, help='Type any intenger without 0 when you set identical condition for training, model. Or type 0 for random state.')
    return parser.parse_args()

def main(args=None):
    if args.seed:
        torch.manual_seed(seed=args.seed)
    device = get_device()
    dataset = DatasetforVirtualData(size=10000,label=args.label,seed=args.seed)
    train_data, test_data = data_spliter(dataset, ratio=args.ratio)
    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True) # You can edt batch_size(intenger).
    test_dataloader = DataLoader(test_data, batch_size=len(test_data),shuffle=False)
    model = NN(inp=99,linear=500,dropout=0.1,labels=args.label).to(device=device) # You can edit linear(intenger), dropout(float f, 0<= f <=1).
    trainer = Trainer(train_data=train_dataloader, test_data=test_dataloader, model=model,lr=1e-3,eps=1e-8,epochs=50, loss=args.loss, alpha = args.alpha) # You can edit lr(learning rate), eps(epsilons that prevent gradient vanishing problem), and epochs(intenger)
    trainer.train()
    acc = trainer.get_performance()
    print('Result accuracy : ', acc)


if __name__=='__main__':
    args = get_argparse()
    main(args)
    