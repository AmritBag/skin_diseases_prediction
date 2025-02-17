# Dependencies
import torch
from tqdm import tqdm

def train_one_epoch(model,dataloader,loss_fn,optimizer,device):
    """
    Train the model for one eapoch

    Arguments:
    ----------
        model {nn.Module}       : CNN model to Train
        dataloader {Dataloader} : dataloader for training data
        loss_fn {nn.Module}     : loss function
        optimizer{Optimizer}    : optimizer
        device {torch.dvice}    : device for run the model on

    Returns:
    --------
    accuracy {float} : the accuracy of model on the currect epoch
    avg_loss {float} : the average loss over the epoch
     """
    # Set the model in trainig mode
    model.train()
    # Initialize epoch_loss,currect prediction and total prediction
    epoch_loss = 0.0
    correct    = 0
    total      = 0
    for inputs ,targets in tqdm(dataloader,desc="Training") :
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        # Feed the model the input data
        outputs = model(inputs)

        # Calculate the loss
        loss    = loss_fn(outputs, targets)

        # Backpropagate the loss
        loss.backward()

        # Optimize the loss
        optimizer.step()
        
        # Get the predictions
        epoch_loss  += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct     += (predicted == targets).sum().item()
        total       += targets.size(0)

    # Calculate the accuracy    
    accuracy = 100 * correct / total

    return epoch_loss / len(dataloader), accuracy