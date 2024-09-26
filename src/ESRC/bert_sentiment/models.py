import torch.nn as nn
from transformers import AutoModel

class RoBERTa(nn.Module):
    def __init__(self, labels):
        super(RoBERTa,self).__init__()
        self.bert = AutoModel.from_pretrained('klue/roberta-large')
        self.Linear = nn.Linear(1024, labels)
        self.Dropout = nn.Dropout(0.1)
        self.Softmax = nn.Softmax(dim = 1)
            
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).pooler_output
        x = self.Dropout(x)
        x = self.Linear(x)
        outputs = self.Softmax(x)
        return outputs

class BERT(nn.Module):
    def __init__(self, labels):
        super(BERT,self).__init__()
        self.bert = AutoModel.from_pretrained('beomi/kcbert-large')
        self.Linear = nn.Linear(1024, labels)
        self.Dropout = nn.Dropout(0.1)
        self.Softmax = nn.Softmax(dim = 1)
            
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).pooler_output
        x = self.Dropout(x)
        x = self.Linear(x)
        outputs = self.Softmax(x)
        return outputs 