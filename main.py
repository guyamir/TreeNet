import torch
from torch.optim import lr_scheduler
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from nn import treeNet
from loader import load
import time
import copy
import torchvision

n_epochs = 2
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = True
torch.manual_seed(random_seed)

# #load data
# l = load('wine',1,0.2)
# X_train, X_valid, y_train, y_valid = torch.from_numpy(l.X_train).cuda(), torch.from_numpy(l.X_valid).cuda(), torch.from_numpy(l.y_train).cuda(), torch.from_numpy(l.y_valid).cuda()

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)

# import matplotlib.pyplot as plt

# fig = plt.figure()
# for i in range(6):
#   plt.subplot(2,3,i+1)
#   plt.tight_layout()
#   plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#   plt.title("Ground Truth: {}".format(example_targets[i]))
#   plt.xticks([])
#   plt.yticks([])
# fig
# plt.show()

# Construct our model by instantiating the class defined above

example_data_vec = example_data.reshape([example_data.shape[0],-1]).cuda()

network = treeNet(example_data_vec, example_targets) #change input and output soon
network.cuda()

import matplotlib.pyplot as plt

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
fig

#learning rate scheduler:
lr = 1e-2
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    target = target.cuda()
    data = data.reshape([data.shape[0],-1]).cuda()
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward(retain_graph=True)
    optimizer.step()
    if batch_idx % log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
            (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
        torch.save(network.state_dict(), './results/model.pth')
        torch.save(optimizer.state_dict(), './results/optimizer.pth') 
    network.dTree.pi = network.dTree.iter_pi(network.dTree.P,network.dTree.pi,network.dTree.mu) #change iterpi to not get or recieve things
   

def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      target = target.cuda()
      data = data.reshape([data.shape[0],-1]).cuda()
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

# criterion = torch.nn.NLLLoss(reduction='mean')
# optimizer = torch.optim.SGD(model.parameters(), lr)

test()
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()

fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
fig

# since = time.time()

# best_model_wts = copy.deepcopy(model.state_dict())
# best_acc = 0.0

# # pat = []
# # pav = []
# for epoch in range(1, n_epochs+1):

#     print('Epoch {}/{}'.format(epoch, n_epochs))
#     print('-' * 10)

#     for phase in ['train', 'val']:
#         if phase == 'train':
            
#             exp_lr_scheduler.step()
#             model.train()  # Set model to training mode
#             x = X_train
#             y = y_train
#         else:
#             model.eval()   # Set model to evaluate mode
#             x = X_valid
#             y = y_valid

#         running_loss = 0.0
#         running_corrects = 0

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward
#         # track history if only in train
#         with torch.set_grad_enabled(phase == 'train'):
#             y_pred = model(x)
#             _, preds = torch.max(y_pred, 1)
#             loss = criterion(y_pred, y)

#             # backward + optimize only if in training phase
#             if phase == 'train':
#                 model.dTree.pi = model.dTree.iter_pi(model.dTree.P,model.dTree.pi,model.dTree.mu)
#                 loss.backward(retain_graph=True)
#                 optimizer.step()

#         # statistics
#         running_loss += loss.item() * x.size(0)
#         running_corrects += torch.sum(preds == y)

#         epoch_loss = running_loss / len(x) ##dataset_sizes[phase]
#         epoch_acc = running_corrects.double() / len(x) #dataset_sizes[phase]

#         print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#             phase, epoch_loss, epoch_acc))

#         # deep copy the model
#         if phase == 'val' and epoch_acc > best_acc:
#             best_acc = epoch_acc
#             best_model_wts = copy.deepcopy(model.state_dict())

#     print()

# time_elapsed = time.time() - since
# print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
# print('Best val Acc: {:4f}'.format(best_acc))
