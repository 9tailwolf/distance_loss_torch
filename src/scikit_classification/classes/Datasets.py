import torch
from sklearn.datasets import make_regression
from torch.utils.data import Dataset

class DatasetforVirtualData(Dataset):
    def __init__(self,size,label,seed):
        self.label = label
        self.preX,self.preY = make_regression(n_samples=size,random_state=seed)
        self.data = [self.preX[i] + [self.preY[i]] for i in range(size)]
        self.make_labels(sep=label)
        self.preprocessing()
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return {
             'X':self.X[index],
             'Y':self.Y[index]
        }
    
    def preprocessing(self):
        self.X = [torch.Tensor(i[:-1]) for i in self.data]
        self.Y = [torch.Tensor(self.get_classification_preds(int(i[-1]))) for i in self.data]

    def make_labels(self,sep):
        self.data.sort(key=lambda x:x[-1])
        sep_point = (max(self.preY)*1.01 - min(self.preY)) / sep
        min_vaule = min(self.preY)
        keep = []
        for i in range(len(self.data)):
            label = int((self.data[i][-1] - min_vaule) // sep_point)
            self.data[i][-1] = label
            keep.append(label)

    def get_classification_preds(self, i):
        temp = [0 for _ in range(self.label)]
        temp[i] = 1
        return temp