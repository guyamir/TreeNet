import torch
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import pandas as pd
from nn import treeNet
from loader import load
import time
import copy

n_epochs = 100

#load data
l = load('wine',1,0.2)
X_train, X_valid, y_train, y_valid = torch.from_numpy(l.X_train).cuda(), torch.from_numpy(l.X_valid).cuda(), torch.from_numpy(l.y_train).cuda(), torch.from_numpy(l.y_valid).cuda()

# Construct our model by instantiating the class defined above
model = treeNet(int(X_train.size(1)), int(y_train.max()+1), max_depth=2) #change input and output soon

#learning rate scheduler:
lr = 1e-2
optimizer_ft = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.NLLLoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr)

since = time.time()

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

# pat = []
# pav = []
for epoch in range(1, n_epochs+1):

    print('Epoch {}/{}'.format(epoch, n_epochs))
    print('-' * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            
            exp_lr_scheduler.step()
            model.train()  # Set model to training mode
            x = X_train
            y = y_train
        else:
            model.eval()   # Set model to evaluate mode
            x = X_valid
            y = y_valid

        running_loss = 0.0
        running_corrects = 0

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            y_pred = model(x)
            _, preds = torch.max(y_pred, 1)
            loss = criterion(y_pred, y)

            # backward + optimize only if in training phase
            if phase == 'train':
                model.iter_pi(y_train)
                loss.backward(retain_graph=True)
                optimizer.step()

        # statistics
        running_loss += loss.item() * x.size(0)
        running_corrects += torch.sum(preds == y)

        epoch_loss = running_loss / len(x) ##dataset_sizes[phase]
        epoch_acc = running_corrects.double() / len(x) #dataset_sizes[phase]

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))

        # deep copy the model
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print()

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))




    # keep track of training and validation loss
    # train_loss = 0.0
    # valid_loss = 0.0

    #  # train the model #
    # model.train()
    # # clear the gradients of all optimized variables
    # optimizer.zero_grad()
    # # forward pass: compute predicted outputs by passing inputs to the model
    # output = model(x)
    # # calculate the batch loss
    # loss = criterion(output, y)
    # # backward pass: compute gradient of the loss with respect to model parameters
    # loss.backward()
    # # perform a single optimization step (parameter update)
    # optimizer.step()
    # # update training loss
    # train_loss += loss.item()*x.size(0)
    
    # # model.eval()
    # # # forward pass: compute predicted outputs by passing inputs to the model
    # # output = model(X_valid)
    # # # calculate the batch loss
    # # loss = criterion(output, y_valid)
    # # # update average validation loss 
    # # valid_loss += loss.item()*X_valid.size(0)
    
    # # calculate average losses
    # train_loss = train_loss/len(x)
    # valid_loss = valid_loss/len(X_valid)

    # print(f'epoch {epoch}, train_loss {train_loss}')#, valid_loss {valid_loss}')#, training accuracy {t_accuracy}')#', validation accuracy {v_accuracy}')

    # model.eval()
    # # forward pass: compute predicted outputs by passing inputs to the model
    # output = model(X_valid)
    # # calculate the batch loss
    # loss = criterion(output, y_valid)
    # # update average validation loss 
    # valid_loss += loss.item()*X_valid.size(0)

    # print(f'valid_loss {valid_loss}')

#     # Forward pass: Compute predicted y by passing x to the model
#     y_pred = model(x).cuda()

#     _, y_m = y_pred.max(1)
#     t_accuracy = (y == y_m).sum().item()/len(y)

#     # yv_pred = model(X_valid).cuda()
    
#     # _, yv_m = yv_pred.max(1)
#     # v_accuracy = (y_valid == yv_m).sum().item()/len(y_valid)
#     pat.append(t_accuracy)
#     # pav.append(v_accuracy)

#     # Compute and print loss
#     loss = criterion(torch.log(y_pred), y)
#     print(f'epoch {epoch}, NLLLoss {loss.item()}, training accuracy {t_accuracy}')#', validation accuracy {v_accuracy}')

#     # Zero gradients, perform a backward pass, and update the weights.
#     optimizer.zero_grad()
#     loss.backward(retain_graph=True)
#     optimizer.step()
#     model.dTree.pi = model.dTree.iter_pi(model.dTree.P,model.dTree.pi,model.dTree.mu)

# df = pd.DataFrame(pat)
# df.to_csv (r'a.csv', index = None, header=True)

# print (df)
# # plt.plot(vat)
# # plt.plot(pat)
# # plt.show()
