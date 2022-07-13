from __future__ import absolute_import, division, print_function

import argparse
from copy import deepcopy
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data.batch import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data import *
#from networks import *
from networks import *
import os
from time import gmtime, strftime
from torch.utils.tensorboard import SummaryWriter

import copy 
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import json
import wandb

import wandb

wandb.init(project="gnn")


def permute_indices(molecules: Batch) -> Batch:
    # Permute the node indices within a molecule, but not across them.
    ranges = [
        (i, j) for i, j in zip(molecules.ptr.tolist(), molecules.ptr[1:].tolist())
    ]
    permu = torch.cat([torch.arange(i, j)[torch.randperm(j - i)] for i, j in ranges])

    n_nodes = molecules.x.size(0)
    inits = torch.arange(n_nodes)
    # For the edge_index to work, this must be an inverse permutation map.
    translation = {k: v for k, v in zip(permu.tolist(), inits.tolist())}

    permuted = deepcopy(molecules)
    permuted.x = permuted.x[permu]
    # Below is the identity transform, by construction of our permutation.
    permuted.batch = permuted.batch[permu]
    permuted.edge_index = (
        permuted.edge_index.cpu()
        .apply_(translation.get)
        .to(molecules.edge_index.device)
    )
    return permuted


def compute_loss(
    model: nn.Module, molecules: Batch, criterion: Callable
) -> torch.Tensor:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    l = get_labels(molecules)
    l = l.to(device)
    new_shape = (len(l), 1)
    l = l.view(new_shape)


    #check model name and labels
    if 'mlp' in str(type(model)).lower():
        x = get_mlp_features(molecules)
        x = x.to(device)
        tag_scores = model(x)

    if 'gnn' in str(type(model)).lower():
        molecules = molecules.to(device)
        
        molecules['x'] = molecules['x'][:, :5]
        tag_scores = model(molecules['x'], molecules['edge_index'], molecules['edge_attr'], molecules['batch'])

    loss = criterion(l, tag_scores)


    return loss


def evaluate_model(
    model: nn.Module, data_loader: DataLoader, criterion: Callable, permute: bool
) -> float:
    # # check if permute is true or flase is true
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    for molecules in data_loader:
        loss_ep = 0
        if permute:
            molecules = permute_indices(molecules)
        
        with torch.no_grad():
            loss = compute_loss(model, molecules, criterion)
            loss_ep+=loss.item()
    avg_loss = round(loss_ep/len(data_loader), 20)
    
    return avg_loss


def train(
    model: nn.Module, lr: float, batch_size: int, epochs: int, seed: int, data_dir: str
):
    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Loading the dataset
    #check model name 
    model_name = args.model
    train, valid, test = get_qm9(data_dir, model.device)
    trainloader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        exclude_keys=["pos", "idx", "z", "name"],
    )
    valloader = DataLoader(
        valid, batch_size=batch_size, exclude_keys=["pos", "idx", "z", "name"]
    )
    test_dataloader = DataLoader(
        test, batch_size=batch_size, exclude_keys=["pos", "idx", "z", "name"]
    )
    
    now = datetime.now()
    today = datetime.today()
    wandb_run_name = wandb.run.name
    print('WANB Project name', wandb_run_name)
    checkpoint_name = os.path.join('logs', wandb_run_name)
    tfb_name_train = os.path.join(checkpoint_name + '/train')
    tfb_name_valid  = os.path.join(checkpoint_name + '/valid')
    if not os.path.exists(tfb_name_train):
        os.makedirs(tfb_name_train)
    if not os.path.exists(tfb_name_valid):
        os.makedirs(tfb_name_valid)
    wandb.tensorboard.patch(root_logdir=checkpoint_name)

    print('Saving logs and tensorboard logs to', checkpoint_name)
    writer = SummaryWriter(tfb_name_train)
    v_writer = SummaryWriter(tfb_name_valid)
    
    print('tfb train path', tfb_name_train)
    print('tfb_valid path', tfb_name_valid)
    save_graph = 'True'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device', device)
    #train mlp here 
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 135], gamma=0.1)
    criterion = nn.MSELoss()

    t_ls=[]
    v_ls = []

    min_valid_loss = np.inf
    #epoch=1
    path = checkpoint_name
    best_fname = 'best_model.pth'
    best_fname = os.path.join(path, best_fname)
    if save_graph:
      if not(os.path.exists(path)):
        os.makedirs(path)
    print('Model', model)
    wandb.define_metric("epoch")
    wandb.define_metric("train_loss", step_metric = "epoch")
    wandb.define_metric("valid_loss", step_metric = "epoch")


    for epoch in tqdm(range(epochs), desc="Epochs", position=0):  
        loss_ep = 0
        v_loss_ep = 0
        model.train()
        
        for i, (molecules) in tqdm(enumerate(trainloader),  desc="Iterations", position=1, leave=False):

            molecules = molecules.to(device)
            loss = compute_loss(model, molecules, criterion)
            loss_ep+=loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        scheduler.step()
        print('Epoch', int(epoch+1))
        ep_loss = round(loss_ep/len(trainloader), 20)
        print("\t\tTraining data loss", ep_loss)
        writer.add_scalar(model_name + '_' + 'Loss',ep_loss, epoch+1)
        t_ls.append(ep_loss)
        wandb.log({"train_loss": loss_ep/len(trainloader)})
        wandb.log({'epoch':epoch+1})
        
        model.eval()
        with torch.no_grad():
            for molecules in valloader:
                molecules = molecules.to(device)
                vloss = compute_loss(model, molecules, criterion)
                v_loss_ep += vloss.item()
            #tfb
        v_ep_loss = round(v_loss_ep/len(valloader),2)
        print('\tValidation loss',v_ep_loss )
        v_writer.add_scalar(model_name + '_' + 'Loss', v_ep_loss,  epoch +1)
        wandb.log({"valid_loss": v_ep_loss/len(valloader)})
        wandb.log({'epoch':epoch+1})

        #print('\tValidation Accuracy',v_ep_acc)
        v_ls.append(v_ep_loss)
    
        #model saving best 
        #check validloss to the loss and measure it 
        #early stop 
        if min_valid_loss > v_ep_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{v_ep_loss:.6f}) \t Saving and Copying model' )
            min_valid_loss = v_ep_loss
            model_best = copy.deepcopy(model)
            if save_graph:
              print('\tSaving model to ', best_fname)
              torch.save(model.state_dict(), best_fname)
            
    print('--Finished Training')
    model = deepcopy(model_best)

    #plot loss curves, train and validation seperately
    plt.title(model_name + 'Train_Loss')
    plt.xlabel('Epoch') 
    plt.ylabel('Loss')
    plt.plot(t_ls,label="train")
    plt.legend()
    fname = str(model_name) + '_Train_loss'  + '.png'
    fname = os.path.join(checkpoint_name, fname)
    
    plt.savefig(fname,bbox_inches='tight')
    #plt.show()
    plt.clf()

    #accuracy plotter for train and validation
    plt.title(model_name + 'Validation_Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.plot(v_ls,label="validation")
    plt.legend()
    
    fname = str(model_name) + 'Validation_Loss'  + '.png'
    fname = os.path.join(path, fname)
    plt.savefig(fname,bbox_inches='tight')
    print('\t Saved graphs to', fname)

    



    val_losses = v_ls
    test_loss = evaluate_model(model, test_dataloader, criterion, permute=False)
    permuted_test_loss = evaluate_model(model, test_dataloader, criterion, permute=True)
    print('********** Running Permutated Test **********')

    print('Test Loss', test_loss)
    wandb.log({"test_loss": test_loss})

    print('Test Loss with Permutation', permuted_test_loss)
    wandb.log({"permuted_test_loss": permuted_test_loss})
    d_loss = {}
    d_loss['train_loss'] = t_ls 
    d_loss['validation loss'] = v_ls
    d_loss['test_loss'] = test_loss
    d_loss['test_loss_permute'] = permuted_test_loss
    json_file = os.path.join(checkpoint_name, model_name+'.json')
    with open(json_file, 'w') as fp:
        json.dump(d_loss, fp, indent=4)

    print('Note: All tensorboard logs, graphs, loss values, best model etc have been saved to ', checkpoint_name)
    
    return model, test_loss, permuted_test_loss, v_ep_loss


def main(**kwargs):
    """main handles the arguments, instantiates the correct model, tracks the results, and saves them."""
    which_model = kwargs.pop("model")
    mlp_hidden_dims = kwargs.pop("mlp_hidden_dims")
    gnn_hidden_dims = kwargs.pop("gnn_hidden_dims")
    gnn_num_blocks = kwargs.pop("gnn_num_blocks")

    # Set default device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if which_model == "mlp":
        model = MLP(FLAT_INPUT_DIM, mlp_hidden_dims, 1)
    elif which_model == "gnn":
        model = GNN(
            n_node_features=Z_ONE_HOT_DIM,
            n_edge_features=EDGE_ATTR_DIM,
            n_hidden=gnn_hidden_dims,
            n_output=1,
            num_convolution_blocks=gnn_num_blocks,
        )
    else:
        raise NotImplementedError("only mlp and gnn are possible models.")

    model.to(device)
    wandb.watch(model, log="all", log_freq=10)
    model = train(
        model, **kwargs
    )

    # plot the loss curve, etc. below.


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument(
        "--model",
        default="gnn",
        type=str,
        choices=["mlp", "gnn"],
        help="Select between training an mlp or a gnn.",
    )
    parser.add_argument(
        "--mlp_hidden_dims",
        default=[128, 128, 128, 128],
        type=int,
        nargs="+",
        help='Hidden dimensionalities to use inside the mlp. To specify multiple, use " " to separate them. Example: "256 128"',
    )
    parser.add_argument(
        "--gnn_hidden_dims",
        default=64,
        type=int,
        help="Hidden dimensionalities to use inside the mlp. The same number of hidden features are used at every layer.",
    )
    parser.add_argument(
        "--gnn_num_blocks",
        default=2,
        type=int,
        help="Number of blocks of GNN convolutions. A block may include multiple different kinds of convolutions (see GNN comments)!",
    )

    # Optimizer hyperparameters
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate to use")
    parser.add_argument("--batch_size", default=128, type=int, help="Minibatch size")

    # Other hyperparameters
    parser.add_argument("--epochs", default=10, type=int, help="Max number of epochs")

    # Technical
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed to use for reproducing results"
    )
    parser.add_argument(
        "--data_dir",
        default="data/",
        type=str,
        help="Data directory where to store/find the qm9 dataset.",
    )

    args = parser.parse_args()
    kwargs = vars(args)
    wandb.config.update(args)
     

    main(**kwargs)
