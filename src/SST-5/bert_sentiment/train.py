import os
import torch
import torch.nn as nn

from .loss import *
from loguru import logger
from transformers import BertConfig, BertModel, RobertaModel, RobertaConfig
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup

from .data import SSTDataset

import scipy.stats as stats
import numpy as np
import json

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save(seed, name, acc):
    with open('data.json') as f:
        data = json.load(f)
    if str(name)+'_'+str(seed) in data.keys():
        data[str(name)+'_'+str(seed)] = max(acc, data[str(name)+'_'+str(seed)])
    else:
        data[str(name)+'_'+str(seed)] = acc

    with open('data.json', 'w') as f:
        json.dump(data, f, indent=4)



def calculate_median_label(labels, probabilities):
    cumulative_probabilities = np.cumsum(probabilities)
    median_index = np.searchsorted(cumulative_probabilities, 0.5)
    return labels[median_index]

def eval_dims_acc(pred, label):
    if type(pred) == torch.Tensor:
        pred = pred.cpu().detach().numpy()
    if type(label) == torch.Tensor:
        label = label.cpu().detach().numpy()
    num_classes = pred.shape[1]
    labels = np.arange(num_classes)
    median_labels = np.array([
        calculate_median_label(labels, pred[i])
        for i in range(pred.shape[0])
    ])
    accuracy = np.sum(median_labels == np.argmax(label, axis=1))
    return accuracy.item()


def eval_acc(pred, label):
    if type(pred) == torch.Tensor:
        pred = pred.cpu().detach().numpy()
    if type(label) == torch.Tensor:
        label = label.cpu().detach().numpy()

    return np.sum(np.argmax(pred, axis=1)==np.argmax(label, axis=1))

def train_one_epoch(model, lossfn, optimizer, scheduler, dataset, batch_size=16):
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model.train()
    train_loss, train_acc_t1, train_acc_t2 = 0.0, 0.0, 0.0
    for batch in tqdm(generator):
        x_ids = batch['input_ids'].to(device)
        x_mask =  batch['attention_mask'].to(device)
        labels = torch.LongTensor(batch['labels']).to(device)
        optimizer.zero_grad()
        outputs = model(x_ids, attention_mask=x_mask)
        loss = lossfn(outputs.float(), labels.float())
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        train_acc_t1 += eval_acc(outputs, labels)
        train_acc_t2 += eval_dims_acc(outputs, labels)
    train_loss /= len(dataset)
    train_acc_t1 /= len(dataset)
    train_acc_t2 /= len(dataset)
    return train_loss, train_acc_t1, train_acc_t2


def evaluate_one_epoch(model, lossfn, optimizer, dataset, batch_size=16):
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model.eval()
    total_loss, acc_t1, acc_t2 = 0.0, 0.0, 0.0
    with torch.no_grad():
        for batch in tqdm(generator):
            x_ids = batch['input_ids'].to(device)
            x_mask =  batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(x_ids, attention_mask=x_mask)
            loss = lossfn(outputs.float(), labels.float())

            total_loss += loss.item()
            acc_t1 += eval_acc(outputs, labels)
            acc_t2 += eval_dims_acc(outputs, labels)
    total_loss /= len(dataset)
    acc_t1 /= len(dataset)
    acc_t2 /= len(dataset)
    return total_loss, acc_t1, acc_t2

class RoBERTa(nn.Module):
    def __init__(self, config, labels):
        super(RoBERTa,self).__init__()
        self.bert = RobertaModel.from_pretrained('roberta-large',config = config)
        self.Linear = nn.Linear(1024, labels)
        self.Dropout = nn.Dropout(0.3)
        self.Softmax = nn.Softmax(dim = 1)
            
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).pooler_output
        x = self.Dropout(self.Linear(x))
        outputs = self.Softmax(x)
        return outputs 

class RoBERTaR(nn.Module):
    def __init__(self, config, hidden):
        super(RoBERTaR,self).__init__()
        self.bert = RobertaModel.from_pretrained('roberta-large',config = config)
        self.Linear = nn.Linear(1024, hidden)
        self.Dropout = nn.Dropout(0.3)
        self.regressor = nn.Linear(hidden, 1)
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).pooler_output
        x = self.Dropout(self.Linear(x))
        outputs = self.regressor(x)
        return outputs 

    



def train(
    bert="roberta",
    loss="dims",
    alpha=2,
    epochs=10,
    batch_size=4,
    lr = 1e-6,
    seed = 42
):
    torch.manual_seed(seed)
    trainset = SSTDataset("train",bert=bert)
    devset = SSTDataset("dev",bert=bert)
    testset = SSTDataset("test",bert=bert)


    config = RobertaConfig.from_pretrained('roberta-large')
    model = RoBERTa(config=config, labels = 5).to(device)
    if loss=='dims':
        lossfn = DiMSLoss(alpha).to(device)
    if loss=='oll':
        lossfn = OLLoss(alpha).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(trainset) / batch_size, num_training_steps= 10 * len(trainset) / batch_size)

    for epoch in range(1, epochs+1):
        train_loss, train_acc_t1, train_acc_t2 = train_one_epoch(
            model, lossfn, optimizer, scheduler, trainset, batch_size=batch_size
        )
        val_loss, val_acc_t1, val_acc_t2 = evaluate_one_epoch(
            model, lossfn, optimizer, devset, batch_size=batch_size
        )
        test_loss, test_acc_t1, test_acc_t2 = evaluate_one_epoch(
            model, lossfn, optimizer, testset, batch_size=batch_size
        )
        logger.info(f"epoch={epoch}")
        logger.info(
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, test_loss={test_loss:.4f}"
        )
        logger.info(
            f"train_acc={train_acc_t1:.4f}, val_acc={val_acc_t1:.4f}, test_acc={test_acc_t1:.4f}, {test_acc_t2:.4f}"
        )
        save(seed,str(loss)+str(alpha), test_acc_t1)

    logger.success("Done!")
