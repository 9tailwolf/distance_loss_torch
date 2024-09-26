"""This module defines a configurable SSTDataset class."""

import pytreebank
import torch
from loguru import logger
from transformers import RobertaTokenizer, BertTokenizer
from torch.utils.data import Dataset


class SSTDataset(Dataset):
    def __init__(self, split="train", bert="bert"):
        logger.info(f"Loading SST {split} set")
        sst = pytreebank.load_sst()
        self.sst = sst[split]

        logger.info("Tokenizing")
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

        self.data = [
            (
                self.tokenizer("[CLS] " + tree.to_lines()[0] + " [SEP]", padding="max_length",max_length=512),
                self.get_classification_preds(tree.label)
            )
            for tree in self.sst
        ]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X, y = self.data[index]
        return {
            'input_ids':torch.tensor(X['input_ids']),
            'attention_mask':torch.tensor(X['attention_mask']),
            'labels':torch.LongTensor(y)
        }

    def get_classification_preds(self, i):
        temp = [0 for _ in range(5)]
        temp[i]=1
        return temp