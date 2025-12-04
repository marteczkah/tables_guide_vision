import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset.MRContrastiveDatasetH5 import MRContrastiveDatasetH5
from losses.losses import MultiplePairsContrastiveLoss
import click
import wandb
from tqdm import tqdm
from models.resnet3d import ResNet3d
from models.resnet2d import ResNet2d
from utils.utils import set_seed

torch.cuda.empty_cache()

ROOT_PATH = "./final_models"
wandb.init(
   project = "tgv"
)

@click.command()
@click.option('--csv_path_train', '-p', help='Path to the csv file with train data information.', required=True)
@click.option('--csv_path_val', '-v', help='Path to the csv file with val data information.', required=True)
@click.option('--batch_size', '-b', help='Traning batch size.', default = 64,type = int)
@click.option('--epochs', '-e', help='Number of epochs.', default = 100, type = int)
@click.option('--temperature', '-t', help='Loss temperature.', default = 0.1, type = float)
@click.option('--store', '-s', help='Where you want to store the models and results.', required=True, type = str)
@click.option('--restart_training', '-r', help='Path to the model you want to train further.', required=False, default=False, type = bool)
@click.option('--previous_epochs', '-u', help='Number of epochs in previous training.', required=False, default=0, type = int)
@click.option('--lr', '-l', help='Learning rate.', required=False, default=1e-3, type = float)
@click.option('--thres', '-h', help='What threshold is the training on.', required=False, default=0.05, type = float)
@click.option('--augment', help='Percent of augmentations.', required=False, default=0.95, type = float)
@click.option('--data_dim', '-d', help='Data type: 2 for 2D (dvm) or 3 for 3D (cardiac mr).', required=False, default=3, type = int)

def main(csv_path_train, store, csv_path_val, batch_size, epochs, temperature, restart_training, previous_epochs, lr, thres, augment, data_dim):
    print("TRAINING")
    set_seed()
    store = os.path.join(ROOT_PATH, store)
    os.makedirs(store, exist_ok=True)
    model_folder = os.path.join(store, "models")
    os.makedirs(model_folder, exist_ok=True)
    
    if augment > 0:
        print('augmentations true')
        augment_org = True
    else:
        augment_org = False

    train_loader = DataLoader(
        MRContrastiveDatasetH5(csv_path_train, augment_org=augment_org, augmentation_rate=augment),
        batch_size = batch_size,
        shuffle = True,
        drop_last=True,
        num_workers=2
    )
    val_loader = DataLoader(
        MRContrastiveDatasetH5(csv_path_val, augment_org=False, augmentation_rate=-1),
        batch_size = batch_size,
        shuffle = True,
        drop_last=True,
        num_workers=2
    )

    if data_dim == 3:
        model = ResNet3d()
    else:
        model = ResNet2d

    if restart_training:
        models_paths = os.listdir(model_folder)
        models_paths = [p for p in models_paths if p.startswith('model') and p.endswith('.pth')]
        models_paths.sort(key=lambda x: int(x.split('model')[1].split('.pth')[0]))
        latest_path = os.path.join(model_folder, models_paths[-1])
        previous_epochs = int(models_paths[-1].split('model')[1].split('.pth')[0]) + 1
        state_dict = torch.load(latest_path, map_location=torch.device("cpu"))
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    else:
        previous_epochs = 0

    if torch.cuda.is_available():
        print(torch.cuda.device_count())
        device = torch.device("cuda")
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)  
            torch.backends.cudnn.benchmark = True
        model = model.to(device)  
    else:
        device = torch.device("cpu")
    
   
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = MultiplePairsContrastiveLoss(temperature=temperature, similarity_thr=thres, device=device)

    for epoch in range(previous_epochs, epochs):
        epoch_loss = []
        epoch_val_loss = []

        print("EPOCH: ", epoch)
        for data in tqdm(train_loader):
            projection, _ = model(data['scan'].float().to(device))
            loss = criterion(projection, data["continuous"].squeeze().to(device), data['categorical'].squeeze().to(device), len_categorical=data['categorical'].squeeze().shape[-1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.detach().cpu().numpy())
            wandb.log({"batch_train_loss: ": loss})
        wandb.log({"epoch_train_loss: ": np.mean(np.array(epoch_loss))})
        torch.save(model.state_dict(), os.path.join(model_folder, 'model_state_dict' +str(epoch) +'.pth'))

        for data in tqdm(val_loader):
            with torch.no_grad():
                projection, _ = model(data['scan'].float().to(device))
                val_loss = criterion(projection, data["continuous"].squeeze().to(device), data['categorical'].squeeze().to(device), len_categorical=data['categorical'].squeeze().shape[-1])
                epoch_val_loss.append(val_loss.detach().cpu().numpy())
                wandb.log({"batch_val_loss": val_loss})
        wandb.log({"epoch_val_loss": np.mean(np.array(epoch_val_loss))})

if __name__ == '__main__':
    main()