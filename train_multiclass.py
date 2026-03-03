import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.DVMClassificationDataset import DVMClassificationDataset
from models.classifier_model import ClassificationModel
import click
import wandb
from tqdm import tqdm
from torchmetrics import Accuracy
from utils.utils import set_seed

torch.cuda.empty_cache()

ROOT_PATH = "./results"
wandb.init(
   project = "tgv_multiclass"
)

@click.command()
@click.option('--csv_path_train', '-p', help='Path to the csv file with train data information.', required=True)
@click.option('--csv_path_train_labels', help='Path to the csv file with train data information.', required=True)
@click.option('--csv_path_val', '-v', help='Path to the csv file with val data information.', required=True)
@click.option('--csv_path_val_labels', help='Path to the csv file with val data information.', required=True)
@click.option('--csv_path_test', '-t', help='Path to the csv file with val data information.', required=False, default='')
@click.option('--csv_path_test_labels', help='Path to the csv file with val data information.', required=False, default='')
@click.option('--batch_size', '-b', help='Traning batch size.', default = 256, type = int)
@click.option('--epochs', '-e', help='Number of epochs.', default = 500, type = int)
@click.option('--store', '-s', help='Where you want to store the models and results.', required=True, type = str)
@click.option('--previous_epochs', '-u', help='Number of epochs in previous training.', required=False, default=0, type = int)
@click.option('--restart_training', '-r', help='Path to the model you want to train further.', required=False, default="", type = str)
@click.option('--frozen', '-f', help='Whether the backbone should be frozen.', required=False, default=True, type = bool)
@click.option('--model_path', '-m', help='Path to the model you want to use to generate representation.', required=False, default='')
@click.option('--test', help='Whether you want to run test.', required=False, type=bool, default=False)
@click.option('--num_workers', '-w', help='Number of workers you want to use.', required=False, default=8, type = int)
@click.option('--lr', help='Learning rate.', required=False, default=3e-4, type = float)

def main(csv_path_train, csv_path_train_labels, store, csv_path_val, csv_path_val_labels, csv_path_test, csv_path_test_labels, lr, batch_size, epochs, restart_training, previous_epochs, frozen, model_path, test, num_workers):
    print("TRAINING CLASSIFICATION")
    set_seed()
    # setup wandb to log the training information
    # setup the training data
    store = os.path.join(ROOT_PATH, store)
    os.makedirs(store, exist_ok=True)
    classifier_folder = os.path.join(store, "dvm_classification")
    os.makedirs(classifier_folder, exist_ok=True)

    train_data = DVMClassificationDataset(csv_path_train, csv_path_train_labels, augmentation_rate=0.95)
    val_data = DVMClassificationDataset(csv_path_val,csv_path_val_labels,  augmentation_rate=0.0)
        
    train_loader = DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_data,
        batch_size = batch_size,
        shuffle = False,
        num_workers=num_workers
    )

    classifier = ClassificationModel(
        backbone_path=model_path, 
        freeze_backbone=frozen, 
        num_classes=286,
        backbone_dim=2048,
        dim=2
    )

    if len(restart_training) > 0:
        print('loading checkpoint')
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
   
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_accuracy = Accuracy(num_classes=286, task="multiclass").to(device)
    val_accuracy = Accuracy(num_classes=286, task="multiclass").to(device)
    best_acc = 0

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
            y_hat = torch.softmax(logits, dim=1)
            train_accuracy.update(y_hat, data['label'].to(device))
            wandb.log({"batch_train_loss: ": loss})

        train_acc = train_accuracy.compute().item()
        train_accuracy.reset()
        wandb.log({"train_acc" : train_acc, "epoch_train_loss: ": np.mean(np.array(epoch_loss))})


        classifier.eval()
        for data in tqdm(val_loader):
                _, logits = classifier(data['scan'].float().to(device))
                val_loss = criterion(logits, data['label'].to(device))
                epoch_val_loss.append(val_loss.detach().cpu().numpy())
                y_hat = torch.softmax(logits, dim=1)
                val_accuracy.update(y_hat, data['label'].to(device))
                wandb.log({"batch_val_loss": val_loss})
        
        torch.save(classifier.state_dict(), os.path.join(classifier_folder, 'last.pth'))
        val_acc = val_accuracy.compute().item()
        wandb.log({
            "epoch_val_loss": np.mean(epoch_val_loss),
            "val_acc": val_acc
        })
        val_accuracy.reset()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(classifier.state_dict(), os.path.join(classifier_folder, 'best.pth'))
    
    if test:
        print("Running test...")
        test_accuracy = Accuracy(num_classes=286, task="multiclass").to(device)
        state_dict = torch.load(os.path.join(classifier_folder, 'best.pth'), map_location=device)
        classifier.load_state_dict(state_dict)
        classifier.eval()
        test_data = DVMClassificationDataset(
            csv_path_test, 
            csv_path_test_labels,  
            augmentation_rate=0.0
        )
        loader = DataLoader(
            test_data,
            batch_size = batch_size,
            shuffle = False,
            num_workers=num_workers
        )
        for data in tqdm(loader):
            logits = classifier(data['scan'].float().to(device))
            y_hat = torch.softmax(logits, dim=1)
            test_accuracy.update(y_hat, data['label'].to(device))
        test_acc = test_accuracy.compute().item()
        wandb.log({
           "test_acc": test_acc
        })

if __name__ == '__main__':
    main()