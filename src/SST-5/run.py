#!/usr/bin/env python3
import argparse
def main(args):
    """Train BERT sentiment classifier."""
    from bert_sentiment.train import train
    train(bert=args.bert,loss=args.loss, seed=args.seed,lr=args.lr,alpha=args.alpha)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--loss', default='dims', type=str, help="Choose your loss function")
    parser.add_argument('--alpha', default=2, type=float, help="Type parameter of loss. Choose your loss function")
    parser.add_argument('--seed', default=0, type=int, help="Type seed")
    parser.add_argument('--bert', default='roberta', type=str, help="Type roberta when you want to use roberta-arge. Or type bert when you want to use bert-large")
    parser.add_argument('--save', default=True, type=bool, help="Type True when you want to save model. Or type False.")
    parser.add_argument('--lr', default=1e-6, type=float, help="Type learning rate")
    main(parser.parse_args())
