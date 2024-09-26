"""This module defines a configurable SSTDataset class."""

import torch
import pandas as pd
from loguru import logger
from transformers import AutoTokenizer
from torch.utils.data import Dataset


class ESRCDataset(Dataset):
    def __init__(self, split="train", bert="bert"):
        logger.info(f"Loading {split} set")
        data = pd.read_csv("./data/"+split +".csv",encoding='cp949')
        logger.info("Tokenizing")
        if bert=="bert":
            self.tokenizer = AutoTokenizer.from_pretrained('beomi/kcbert-large')
        elif bert=="roberta":
            self.tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
        elif bert=="electra":
            self.tokenizer = AutoTokenizer.from_pretrained('beomi/KcELECTRA-base-v2022')
        self.data = [
            (
                self.tokenizer("[CLS] " + data['신고내용'][i] + " [SEP]", padding="max_length",max_length=256),
                self.get_classification_preds(int(data['긴급코드'][i][1])),
            )
            for i in range(data.shape[0])
        ]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X, y = self.data[index][0],self.data[index][1]

        return {
            'input_ids':torch.tensor(X['input_ids'][:256]),
            'attention_mask':torch.tensor(X['attention_mask'][:256]),
            'labels':torch.LongTensor(y)
        }

    def get_classification_preds(self, i):
        temp = [0 for _ in range(4)]
        temp[i]=1
        return temp