import torch

def train(
    model : torch.nn.Module,
    loss_fn : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    epochs : int,
    dataset : torch.utils.data.Dataset,
):
    
    ...