import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.MRContrastiveDatasetH5 import MRContrastiveDatasetH5
from dataset.MRContrastiveFilterDataset import MRContrastiveFilterDataset
from models.classifier_model import ClassificationModel
import click
import wandb
from tqdm import tqdm
from torchmetrics import Precision, Recall, AUROC, Accuracy
from utils.utils import set_seed

torch.cuda.empty_cache()

ROOT_PATH = "./results"
wandb.init(
   project = "tgv_downsteam"
)

@click.command()
@click.option('--h5_path_train', '-p', help='Path to the csv file with train data information.', required=True)
@click.option('--csv_path_train_filter', '-t', help='Path to the csv file with train data information.', required=True)
@click.option('--h5_path_val', '-v', help='Path to the csv file with val data information.', required=True)
@click.option('--batch_size', '-b', help='Traning batch size.', default = 64, type = int)
@click.option('--epochs', '-e', help='Number of epochs.', default = 100, type = int)
@click.option('--store', '-s', help='Where you want to store the models and results.', required=True, type = str)
@click.option('--previous_epochs', '-u', help='Number of epochs in previous training.', required=False, default=0, type = int)
@click.option('--restart_training', '-r', help='Path to the model you want to train further.', required=False, default="", type = str)
@click.option('--frozen', '-f', help='Whether the backbone should be frozen.', required=False, default=True, type = bool)
@click.option('--model_path', '-m', help='Path to the model you want to use to generate representation.', required=True)
@click.option('--test', help='Whether you want to test the model after training.', required=False, default=False, type = bool)
@click.option('--h5_test', help='If you want to test the model, declare the path to the h5 file.', required=False, default="", type = str)

def main(h5_path_train, csv_path_train_filter, store, h5_path_val, batch_size, epochs, restart_training, previous_epochs, frozen, model_path, test, h5_test):
    print("TRAINING MULTILABEL CLASSIFICATION")
    set_seed()
    # setup wandb to log the training information
    # setup the training data
    store = os.path.join(ROOT_PATH, store)
    os.makedirs(store, exist_ok=True)
    classifier_folder = os.path.join(store, "multilabel_cad")
    os.makedirs(classifier_folder, exist_ok=True)

    train_data = MRContrastiveFilterDataset(
        h5_path_train, 
        csv_path_train_filter, 
        augmentation_rate=0.4
    )
    val_data = MRContrastiveDatasetH5(
        h5_path_val, 
        augmentation_rate=-1
    )
        
    train_loader = DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_data,
        batch_size = batch_size,
        shuffle = False,
        num_workers=4
    )

    classifier = ClassificationModel(
        backbone_path=model_path, 
        backbone_dim=2048, 
        num_classes=4,
        freeze_backbone=frozen
    )

    if len(restart_training) > 0:
        state_dict = torch.load(restart_training, map_location='cpu')
        classifier.load_state_dict(state_dict)

    classifier.train()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.cuda.device_count() > 1:
            print("MULTIPLE CUDA DEVICES")
            classifier = torch.nn.DataParallel(classifier)  
            torch.backends.cudnn.benchmark = True
        classifier.to(device)
    else:
        device = torch.device("cpu")
   
    if frozen:
        lr = 3e-3
    else:
        lr = 3e-4

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    average = 'macro'
    precision_metric = Precision(num_labels=4, average=average, task="multilabel")
    recall_metric = Recall(num_labels=4, average=average, task="multilabel")
    auc_metric = AUROC(num_labels=4, average=average, task="multilabel")
    accuracy_metric = Accuracy(num_labels=4, average=average, task="multilabel")

    best_auc = 0
    for epoch in range(previous_epochs, epochs):
        epoch_loss = []
        epoch_val_loss = []

        print("EPOCH: ", epoch)
        classifier.train()
        for data in tqdm(train_loader):
            _, logits = classifier(data['scan'].float().to(device))
            loss = criterion(logits, data['label'].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.detach().cpu().numpy())
            wandb.log({"batch_train_loss: ": loss})
        wandb.log({"epoch_train_loss: ": np.mean(np.array(epoch_loss))})

        classifier.eval()
        for data in tqdm(val_loader):
            with torch.no_grad():
                _, logits = classifier(data['scan'].float().to(device))
                val_loss = criterion(logits, data['label'].to(device))
                preds = torch.sigmoid(logits.detach())
                predictions = (preds.detach().cpu() > 0.5).int()
                precision_metric.update(predictions.cpu(), data['label'].cpu().long())
                recall_metric.update(predictions.cpu(), data['label'].cpu().long())
                auc_metric.update(preds.cpu(), data['label'].cpu().long())  
                accuracy_metric.update(predictions.cpu(), data['label'].cpu().long())
                epoch_val_loss.append(val_loss.detach().cpu().numpy())
                wandb.log({"batch_val_loss": val_loss})
        
        torch.save(classifier.state_dict(), os.path.join(classifier_folder, 'last.pth'))
                
        precision = precision_metric.compute()
        recall = recall_metric.compute()
        auc = auc_metric.compute()
        accuracy = accuracy_metric.compute()
        wandb.log({
            "epoch_val_loss": np.mean(epoch_val_loss),
            "precision": precision.item(),
            "recall": recall.item(),
            "auc": auc.item(),
            "accuracy": accuracy.item()
        })

        if auc.item() > best_auc:
            torch.save(classifier.state_dict(), os.path.join(classifier_folder, 'best.pth'))
            best_auc = auc.item()
        
        precision_metric.reset()
        recall_metric.reset()
        auc_metric.reset()
        accuracy_metric.reset()

    if test:
        print('Running test...')
        test_auc = AUROC(num_labels=4, average=average, task="multilabel")
        state_dict = torch.load(os.path.join(classifier_folder, 'best.pth'), map_location=device)
        classifier.load_state_dict(state_dict)
        classifier.eval()
        test_data = MRContrastiveDatasetH5(h5_test, augmentation_rate=-1)
        loader = DataLoader(
            test_data,
            batch_size = batch_size,
        )
        for data in tqdm(loader):
            logits = classifier(data['scan'].float().to(device))
            y_hat = torch.sigmoid(logits)
            test_auc.update(y_hat.detach().cpu(), data['label'].cpu().long())
        auc = test_auc.compute().item()
        wandb.log({
            "test_auc": auc
        })

if __name__ == '__main__':
    main()