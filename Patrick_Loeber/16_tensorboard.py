import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import numpy as np
import torch.utils
import torch.utils.data
import torchvision
from torchvision.transforms import transforms
import sys 

######################### TENSORBOARD ##########################
from torch.utils.tensorboard import SummaryWriter 
# default `log_dir` is "runs" - we'll be more specific here

writer = SummaryWriter('runs/mnist1')
################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 16
num_epochs = 2
learning_rate = 0.01
input_size = 784
hidden_size = 120
n_classes = 10

transform = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train = True,
                                           transform=transform,
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transform,
                                          download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=0)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=0)

examples = next(iter(test_loader))
example_data, example_targets = examples

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(example_data[i][0], cmap='gray')

#################### TENSORBOARD ######################
img_grid = torchvision.utils.make_grid(example_data)
writer.add_image('mnist_images', img_grid)
# writer.close()
# sys.exit()
#######################################################

class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size, n_classes):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x 
    

model = NeuralNet(input_size, hidden_size, n_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#################### TENSORBOARD ######################
writer.add_graph(model, example_data.reshape(-1, 28*28).to(device))
# writer.close()
# sys.exit()
#######################################################

# Train the model
running_loss = 0.0
running_correct = 0
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, dim=1)
        running_loss += loss.item()
        running_correct += (labels == predicted).sum().item()

        if i+1 % 100 == 0:
            print(f'Epoch: [{epoch+1}/{num_epochs}], Step: [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            #################### TENSORBOARD ######################
            writer.add_scalar('training_loss', running_loss / 100, epoch * n_total_steps + i)
            running_accuracy = running_correct / 100 / predicted.size(0)
            writer.add_scalar('accuracy', running_accuracy, epoch * n_total_steps + i)
            running_correct = 0
            running_loss = 0.0
            #######################################################

writer.close()
sys.exit()

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
class_labels = []
class_preds = []

with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for i, (images, labels) in enumerate(test_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predictions = torch.max(outputs, dim=1)

        n_samples += labels.size(0)
        n_correct += (labels == predictions).sum().item()

        class_probs_batch = [F.softmax(output, dim=0) for output in outputs]

        class_preds.append(class_probs_batch)
        class_labels.append(labels)

        print(class_preds)
        print(class_labels)

    # 10000, 10 and 10000, 1
    # stack concatenates tensors along a new dimension
    # cat concatenates tensors in the given dimension
    class_preds = torch.cat([torch.stack(batch) for batch in class_preds])
    class_labels = torch.cat(class_labels)

    print("############## CAT ##############")
    print(class_preds)
    print(class_labels)
    print("############################")


    acc = 100 * (n_correct / n_samples)
    print(f'Accuracy of the model is {acc}%')

    #################### TENSORBOARD ######################
    classes = range(10)
    for i in classes:
        labels_i = class_labels == i
        preds_i = class_preds[:, i]
        writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
        writer.close()
    #######################################################



