from autograd.BaseGraph import Graph
from autograd.Nodes import relu, Linear, MSE, sigmoid
from utils.scheduler import exponential_decay
import numpy as np
from utils.data_processor import create_flower_dataloaders
import pickle
import torch

# set random seed
np.random.seed(0)
torch.manual_seed(0)

# Basic settings
data_root = "./flowers"
vis_root = "./vis"
model_save_path = "./model/Best_MLPAE.pkl"
batch_size = 16
num_epochs = 100
early_stopping_patience = 5
IMG_WIDTH, IMG_HEIGHT = 24, 24

scheduler = exponential_decay(initial_learning_rate=1, decay_rate=0.9, decay_epochs=5)
training_dataloader, validation_dataloader = create_flower_dataloaders(batch_size, data_root, IMG_WIDTH, IMG_HEIGHT)

model = Graph([
    Linear(3 * IMG_HEIGHT * IMG_WIDTH, 256), relu(),
    Linear(256, 128), relu(),
    Linear(128, 64), relu(),
    Linear(64, 128), relu(),
    Linear(128, 256), relu(),
    Linear(256, 3 * IMG_HEIGHT * IMG_WIDTH), sigmoid()
])

loss_fn_node = MSE()

save_model_name = f"Best_MLPAE.pkl"
min_valid_loss = float('inf')
avg_train_loss = 10000.
avg_valid_loss = 10000.

for epoch in range(num_epochs):

    train_losses = []

    # Adjust the learning rate
    lr = scheduler(epoch)
    step_num = len(training_dataloader)

    # Training all batches
    for images, _ in training_dataloader:
        images = images.detach().numpy().reshape(images.shape[0], -1)
        model.flush()
        loss_fn_node.flush()

        # Forward pass
        output = model.forward(images)
        loss = loss_fn_node.forward(output, images)

        # Backward pass
        grad = loss_fn_node.backward()
        model.backward(grad)

        # Update model parameters
        model.optimstep(lr)

        train_losses.append(loss)

        # train_losses.append(loss.item())
    avg_train_loss = sum(train_losses) / len(train_losses)
    # print(sum(train_losses))

    # Validation every 3 epochs
    if epoch % 3 == 0:
        valid_losses = []
        for images, _ in validation_dataloader:
            # images = images.view(images.shape[0], -1)  # Flatten the images
            images = images.detach().numpy().reshape(images.shape[0], -1)  # Flatten images

            model.flush()
            loss_fn_node.flush()

            # Forward pass
            output = model.forward(images)
            val_loss = loss_fn_node.forward(output, images)

            valid_losses.append(val_loss)

        avg_valid_loss = sum(valid_losses) / len(valid_losses)
        if avg_valid_loss < min_valid_loss:
            min_valid_loss = avg_valid_loss
            pickle.dump(model, open(model_save_path, 'wb'))

        # Implement early stopping
        early_stopping_counter = 0
        if avg_valid_loss >= min_valid_loss:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                break
        else:
            early_stopping_counter = 0

    print(f"Epoch: {epoch}, Train Loss: {avg_train_loss}, Valid Loss: {avg_valid_loss}")
