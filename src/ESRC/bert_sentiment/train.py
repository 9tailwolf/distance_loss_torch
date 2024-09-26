import os
import torch
import torch.nn as nn
import json

from .loss import *
from torch.nn import CrossEntropyLoss
from loguru import logger
from transformers import AutoModel
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

from .data import SSTDataset
from .models import *
from transformers import get_linear_schedule_with_warmup

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, lossfn, optimizer, scheduler, dataset, batch_size=16):
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model.train()
    train_loss, train_acc = 0.0, 0.0
    for batch in tqdm(generator):
        x_ids = batch['input_ids'].to(device)
        x_mask =  batch['attention_mask'].to(device)
        labels = torch.LongTensor(batch['labels']).to(device)
        optimizer.zero_grad()
        outputs = model(x_ids, attention_mask=x_mask)
        loss = lossfn(outputs.float(), labels.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        train_loss += loss.item()
        pred_labels = torch.argmax(outputs, axis=1)
        train_acc += (pred_labels == torch.argmax(labels, axis=1)).sum().item()
    scheduler.step()
    train_loss /= len(dataset)
    train_acc /= len(dataset)
    return train_loss, train_acc


def evaluate_one_epoch(model, lossfn, optimizer, scheduler, dataset, batch_size=16):
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model.eval()
    total_loss, acc = 0.0, 0.0
    with torch.no_grad():
        for batch in tqdm(generator):
            x_ids = batch['input_ids'].to(device)
            x_mask =  batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(x_ids, attention_mask=x_mask)
            loss = lossfn(outputs.float(), labels.float())

            total_loss += loss.item()
            pred_labels = torch.argmax(outputs, axis=1)
            acc += (pred_labels == torch.argmax(labels, axis=1)).sum().item()
    total_loss /= len(dataset)
    acc /= len(dataset)
    return total_loss, acc


def train(
    bert="bert",
    loss='dims',
    epochs=7,
    batch_size=16,
    seed=0,
    save=False,
    alpha=2,
):
    trainset = SSTDataset("train",bert=bert)
    devset = SSTDataset("dev",bert=bert)
    testset = SSTDataset("test",bert=bert)

    torch.manual_seed(seed)


    if bert=="bert":
        model = BERT(labels = 4)
    elif bert=="roberta":
        model = RoBERTa(labels = 4)

    model = model.to(device)
    if loss=='dims':
        lossfn = DiMSLoss(alpha=alpha).to(device)
    if loss=='oll':
        lossfn = OLLoss(alpha=alpha).to(device)
    elif loss=='cross':
        lossfn = CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(1, epochs):
        train_loss, train_acc = train_one_epoch(
            model, lossfn, optimizer, scheduler, trainset, batch_size=batch_size
        )
        val_loss, val_acc = evaluate_one_epoch(
            model, lossfn, optimizer, scheduler, devset, batch_size=batch_size
        )
        test_loss, test_acc = evaluate_one_epoch(
            model, lossfn, optimizer, scheduler, testset, batch_size=batch_size
        )
        logger.info(f"epoch={epoch}")
        logger.info(
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, test_loss={test_loss:.4f}"
        )
        logger.info(
            f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, test_acc={test_acc:.3f}"
        )
        if save:
            #torch.save(model, f"{bert}_{loss}_e{epoch}_{test_acc:.3f}.pickle")
            with open("./database.json", "r") as json_file:
                database = json.load(json_file)
            seed = str(seed)
            try:
                database[bert+loss+seed+str(alpha)] = max(float(database[bert+loss+seed+str(alpha)]), test_acc)
            except:
                database[bert+loss+seed+str(alpha)] = test_acc
            with open("./database.json", "w") as json_file:
                json.dump(database, json_file, indent=4)


    logger.success("Done!")
