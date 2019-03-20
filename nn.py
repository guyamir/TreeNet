import torch

from tree import DecisionTree

class treeNet(torch.nn.Module):
    def __init__(self,x,y,depth = 2):
        """
        In the constructor we instantiate nn.Linear modules and assign them as
        member variables.
        """

        H = 100
        D_out = 1
        self.dTree = DecisionTree(x, y, idxs=range(len(y)), depth = 2)

        super(treeNet, self).__init__()
        self.linear1 = torch.nn.Linear(self.dTree.D_in, H).cuda()
        self.theta = torch.nn.ModuleList([torch.nn.Linear(self.dTree.D_in, D_out) for i in range(self.dTree.nNodes)]).cuda()
        self.sigmoid = torch.nn.Sigmoid().cuda()

    def forward(self, x, idxs = None):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # if phase is 'train':
        #     self.dTree.mu = self.dTree.mu_train
        #     self.dTree.pi = self.dTree.iter_pi(self.dTree.P,self.dTree.pi,self.dTree.mu)
        # elif phase is 'val':
        #     self.dTree.mu = self.dTree.mu_val
        if idxs is None: 
            idxs = range(len(x))
        
        # h = self.linear1(x.float()).clamp(min=0).cuda()
        # print(f'h {h}')
        y_pred = self.dTree.plant(x,self.theta, idxs = idxs)
        # _, y_pred = y_pred_onehot.max(1)#convert from one-hot encoding to vector


        return y_pred