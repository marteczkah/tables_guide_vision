import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.MRContrastiveDatasetH5 import MRContrastiveDatasetH5
from dataset.MRContrastiveFilterDataset import MRContrastiveFilterDataset
from models.regression_model import RegressionModel
import click
import wandb
from tqdm import tqdm
from utils.utils import get_bias, set_seed

torch.cuda.empty_cache()

ROOT_PATH = "./results"
wandb.init(
   project = "tgv_regression"
)

@click.command()
@click.option('--h5_path_train', '-p', help='Path to the csv file with train data information.', required=True)
@click.option('--csv_path_train_filter', '-t', help='Path to the csv file with train data information.', required=True)
@click.option('--h5_path_val', '-v', help='Path to the csv file with val data information.', required=True)
@click.option('--h5_path_test', help='Path to the csv file with val data information.', required=False, default="")
@click.option('--batch_size', '-b', help='Traning batch size.', default = 512,type = int)
@click.option('--epochs', '-e', help='Number of epochs.', default = 50, type = int)
@click.option('--store', '-s', help='Where you want to store the models and results.', required=True, type = str)
@click.option('--previous_epochs', '-u', help='Number of epochs in previous training.', required=False, default=0, type = int)
@click.option('--frozen', '-f', help='Whether the backbone should be frozen.', required=False, default=False, type = bool)
@click.option('--attribute', '-a', help='Attribute to be used for regression.', required=True, type = str)
@click.option('--model_path', '-m', help='Path to the model you want to use to generate representation.', required=True)
@click.option('--dim', '-d', help='Image size.', required=False, default=3, type = int)
@click.option('--test', help='Image size.', required=False, default=False, type = bool)
@click.option('--num_workers', '-w', help='Number of workers you want to use.', required=False, default=8, type = int)

def main(h5_path_train, h5_path_test, csv_path_train_filter, store, h5_path_val, batch_size, epochs, previous_epochs, frozen, attribute, model_path, dim, test, num_workers):
    print("TRAINING")
    set_seed()
    # setup wandb to log the training information
    # setup the training data
    store = os.path.join(ROOT_PATH, store)
    os.makedirs(store, exist_ok=True)
    regres_folder = os.path.join(store, "regression")
    os.makedirs(regres_folder, exist_ok=True)
    
    train_data = MRContrastiveFilterDataset(
        h5_path_train, csv_path_train_filter,
        augmentation_rate=0.0, 
        attribute=attribute
    )
    val_data = MRContrastiveDatasetH5(
        h5_path_val, 
        augmentation_rate=-1, 
        attribute=attribute
    )
    
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
    
    regressor = RegressionModel(
        backbone_path=model_path, 
        backbone_dim=2048,
        bias=get_bias(attribute, False), 
        freeze_backbone=frozen, 
        dim=dim
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.cuda.device_count() > 1:
            print("MULTIPLE CUDA DEVICES")
            regressor = torch.nn.DataParallel(regressor)  
            torch.backends.cudnn.benchmark = True
        regressor.to(device)
    else:
        device = torch.device("cpu")
   
    optimizer = torch.optim.AdamW(regressor.parameters(), lr=3e-4)
    criterion = nn.HuberLoss()
    evaluation_criterion = nn.L1Loss()
    best_mae = float('inf')

    for epoch in range(previous_epochs, epochs + previous_epochs):
        epoch_loss = []
        epoch_val_loss = []

        regressor.train()
        print("EPOCH: ", epoch)
        for data in tqdm(train_loader):
            out = regressor(data['scan'].float().to(device))
            loss = criterion(out, data['attribute'].float().to(device).unsqueeze(dim=0).transpose(1,0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.detach().cpu().numpy())
            wandb.log({"batch_train_loss: ": loss})
        wandb.log({"epoch_train_loss: ": np.mean(np.array(epoch_loss))})

        regressor.eval()
        for data in tqdm(val_loader):
            with torch.no_grad():
                out = regressor(data['scan'].float().to(device))
                val_loss = evaluation_criterion(out, data['attribute'].float().to(device).unsqueeze(dim=0).transpose(1,0))
                epoch_val_loss.append(val_loss.detach().cpu().numpy())
                wandb.log({"batch_val_loss": val_loss})
        cur_mae = np.mean(np.array(epoch_val_loss))
        wandb.log({"epoch_val_loss": cur_mae})
        
        torch.save(regressor.state_dict(), os.path.join(regres_folder, 'last.pth'))
        if best_mae > cur_mae:
            best_mae = cur_mae
            torch.save(regressor.state_dict(), os.path.join(regres_folder, 'best.pth'))
    
    if test:
        test_mae = []
        print('Running test...')
        state_dict = torch.load(os.path.join(regres_folder, 'best.pth'), map_location=device)
        regressor.load_state_dict(state_dict)
        regressor.eval()        
        test_data = MRContrastiveDatasetH5(
            h5_path_test, 
            augmentation_rate=-1, 
            attribute=attribute)
        loader = DataLoader(
            test_data,
            batch_size = batch_size,
        )
        for data in tqdm(loader):
            out = regressor(data['scan'].float().to(device))
            cur_mae = evaluation_criterion(out, data['attribute'].float().to(device).unsqueeze(dim=0).transpose(1,0))
            test_mae.append(cur_mae.detach().cpu().numpy())
        mae = np.mean(np.array(test_mae))
        print(mae)
        wandb.log({
            "test_auc": mae
        })

if __name__ == '__main__':
    main()