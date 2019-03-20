import torch
import numpy as np

class node():

    def __init__(self, h, theta, tree, mu = None, idxs = None):       
        # idxs = tree.b
        tree.Nodes.append(self)
        self.counter = len(tree.Nodes) - 1
        
        if idxs is None: 
            idxs = np.arange(len(tree.y)) 

        if(self.counter == 0):
            self.father = None
            self.depth = 0  
        else:
            self.father = tree.Nodes[self.counter-1]
            self.depth = self.father.depth+1
        
        self.mu = mu.cuda()
        
        self.d = torch.sigmoid(theta[self.counter](h[idxs].float()))#.cuda() 

        l_mu = torch.mul(self.mu,self.d)
        
        r_mu = torch.mul(self.mu,(1-self.d))

        if(self.depth<tree.depth-1):
            self.lhs = node(h=h, theta=theta, tree = tree, mu = l_mu, idxs = idxs)
            self.rhs = node(h=h, theta=theta, tree = tree, mu = r_mu, idxs = idxs)
        else:
            self.lhs = leaf(tree = tree, mu = l_mu, idxs = idxs)
            self.rhs = leaf(tree = tree, mu = r_mu, idxs = idxs)
            
class leaf():
    def __init__(self, tree, mu, idxs):
        tree.Leaves.append(self)
        self.mu = mu
        self.counter = len(tree.Leaves)-1
        tree.mu[idxs,self.counter] = self.mu[:,0]