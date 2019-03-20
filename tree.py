import torch
import node
import numpy as np
class DecisionTree():
        
    def __init__(self, x, y, idxs=None, depth = 4):

        #add more feature in order to similate sklearn D-tree
        ################################################################################################
        self.depth = depth
        
        # self.x = torch.from_numpy(x[idxs]).float()

        if idxs == None:
            idxs = range(len(y))

        self.x = x[idxs]

        self.y = y[idxs].long()

        self.nLeaves = 2**depth
        
        self.nClasses = y.max()+1
        
        self.nNodes = self.node_calc()

        self.N = len(self.y)

        self.D_in = self.x.size(1)
        
        # self.mu_train = torch.ones(len(y),self.nLeaves).cuda()

        self.mu = torch.ones(len(y),self.nLeaves).cuda()

        self.pi = (1/self.nClasses.item())*torch.ones([self.nLeaves,self.nClasses]).cuda()

        self.P = self.Probability().cuda()
        
        if idxs is None: 
            self.idxs = np.arange(len(y)) 
        else: 
            self.idxs = idxs

        self.N = len(self.idxs)

    #iterate pi
    def iter_pi(self,P,pi_0,mu):
        pi = torch.zeros([self.nLeaves,self.nClasses]).cuda()
        z = torch.zeros(self.nLeaves).cuda()
        
        for l in range(self.nLeaves):
            for i in range(self.N):
                if(P[i,self.y[i]]!=0): #self.mu[i,l]!=0 or 
                    pi[l,self.y[i]] = pi[l,self.y[i]].clone() + pi_0[l,self.y[i]].clone()*mu[i,l].clone()/P[i,self.y[i]].clone() 
            z[l] = torch.sum(pi[l,:],0).cuda()
            # print(f'before normalizing: pi[l,:] {pi[l,:]}')
            pi[l,:] = pi[l,:].clone()/z[l].clone()
            # print(f'after normalizing: pi[l,:] {pi[l,:]}')

        return pi.cuda()

    #calculate probability of sample x being in class y
    def Probability(self):
        return self.mu.clone().mm(self.pi) #return self.mu.mm(self.pi) 

    #calculating number of nodes from tree depth:
    def node_calc(self):
        nNodes = 0
        for i in range(self.depth): nNodes += 2**i
        return nNodes

    def plant(self,h, theta, idxs = None):

        # reboot the nodes and leaves
        self.Nodes = []
        self.Leaves = []
        
        self.mu = torch.ones(len(idxs),self.nLeaves).cuda()

        if idxs is None: 
            idxs = len(h) 
        
        # plant the root node:(self, x, y, idxs=None, depth = 2, n_epochs = 100)
        self.root = node.node(h, theta, self, mu = torch.ones([len(idxs),1]), idxs = idxs)
        
        # calculate new probability
        self.P = self.Probability()
        
        # P_log = torch.log(self.P)
        return self.P