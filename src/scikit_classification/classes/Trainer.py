import time
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from transformers import get_linear_schedule_with_warmup
from torch.nn.utils import clip_grad_norm_
from loguru import logger
import json
import scipy.stats as stats

from classes.Utils import get_device
from classes.Loss import *

class Trainer:
    def __init__(self,train_data,test_data,model,lr,eps,epochs,loss, alpha):
        self.device = get_device()
        self.model = model
        self.dataloader_train = train_data
        self.dataloader_test = test_data
        self.epochs = epochs
        self.performance = 0

        if loss=='MSE':
            self.loss_fn = MSELoss()
        elif loss=='DiMS':
            self.loss_fn = DiMSLoss(alpha=alpha)
        elif loss=='CE':
            self.loss_fn = CrossEntropyLoss()
        elif loss=='MAE':
            self.loss_fn = L1Loss()
        elif loss=='OLL':
            self.loss_fn = OLLoss(alpha=alpha)

    
        self.optimizer = AdamW(self.model.parameters(), lr=lr, eps=eps)
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)

    def train(self):
        self.model.train()
        for epoch in tqdm(range(self.epochs)):
            total_loss = 0
            self.current_time = time.time()
            #print('epochs : ', epoch)
            for d in self.dataloader_train:
                self.model.zero_grad()
                X,Y = d['X'].to(self.device),d['Y'].to(self.device)
                outputs = self.model(X)
                loss = self.loss_fn(outputs, Y)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            self.eval(epoch, total_loss)
            
    def eval(self,e,l):
        correct,correct2 = 0,0
        count = 0
        self.model.eval()
        for d in self.dataloader_test:
            X = d['X'].to(self.device)
            Y = d['Y']

            with torch.no_grad():
                outputs = self.model(X)
            outputs = outputs.detach().cpu().numpy()
            correct += self.eval_dims_acc(outputs, Y)
            correct2 += self.eval_accuracy(outputs, Y)
            count += len(outputs)
        accuracy = correct / count
        acc2 = correct2 / count
        self.performance = max(accuracy, self.performance)
        self.model.train()
        return round(accuracy,5), round(acc2,5)

    
    def eval_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = np.array(np.argmax(labels, axis=1)).flatten()
        res =  np.sum(pred_flat == labels_flat)
        return res

    def get_performance(self):
        return self.performance
    
    def eval_dims_acc(self, pred, label):
        if type(pred) == torch.Tensor:
            pred = pred.numpy()
        if type(label) == torch.Tensor:
            label = label.numpy()
        num_classes = pred.shape[1]
        labels = np.arange(num_classes)
        median_labels = np.array([
            self.calculate_median_label(labels, pred[i])
            for i in range(pred.shape[0])
        ])
        accuracy = np.sum(median_labels == np.argmax(label, axis=1))
        return accuracy.item()
    
    def calculate_median_label(self, labels, probabilities):
        cumulative_probabilities = np.cumsum(probabilities)
        median_index = np.searchsorted(cumulative_probabilities, 0.5)
        return labels[median_index]
