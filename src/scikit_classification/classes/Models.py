import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class NN(nn.Module):
    def __init__(self, inp, linear, dropout, labels):
        super(NN,self).__init__()
        self.Linear = nn.Linear(inp, linear)
        self.BN = nn.BatchNorm1d(linear)
        self.Dropout = nn.Dropout(dropout)
        self.Relu = nn.ReLU()
        self.Classification = nn.Linear(linear, labels)
        self.Softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.Linear(x)
        x = self.BN(x)
        x = self.Relu(x)
        x = self.Dropout(x)
        outputs = self.Softmax(self.Classification(x))
        return outputs

class BERT_NN(nn.Module):
    def __init__(self, linear, dropout, labels):
        super(BERT_NN,self).__init__()
        self.linear, self.dropout, self.labels = linear, dropout, labels
        self.bert = AutoModel.from_pretrained("beomi/KcELECTRA-base-v2022")
        self.Linear = nn.Linear(768, linear)
        self.Dropout = nn.Dropout(dropout)
        self.Relu = nn.ReLU()
        self.Classification = nn.Linear(linear, labels)
        self.Softmax = nn.Softmax(dim = 1)
            
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0][:,0]
        x = self.Linear(x)
        x = self.Relu(x)
        x = self.Dropout(x)
        outputs = self.Softmax(self.Classification(x))
        return outputs
