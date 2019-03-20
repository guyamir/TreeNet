import torch

class treeNet(torch.nn.Module):
    def __init__(self,D_in,nClasses,max_depth = 2):
        """
        In the constructor we instantiate nn.Linear modules and assign them as
        member variables.
        """

        #tree parameters:
        self.max_depth = max_depth
        self.nLeaves = 2**max_depth
        self.pi = (1/nClasses)*torch.ones([self.nLeaves,nClasses]).cuda()
        nNodes = self.node_calc()

        H = 100
        D_out = 1

        #network parameters:
        super(treeNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H).cuda()
        self.theta = torch.nn.ModuleList([torch.nn.Linear(H, D_out) for i in range(nNodes)]).cuda()
        self.sigmoid = torch.nn.Sigmoid().cuda()


    def forward(self, x, idxs = None):
        
        self.Nodes = 0
        self.Leaves = 0

        if idxs == None:
            self.idxs = range(len(x))
        else:
            self.idxs = idxs
        
        h = self.linear1(x.float()).clamp(min=0).cuda()
        y_pred = self.tree(h = h)



        return y_pred

    def tree(self, h, idxs=None):
        #add more feature in order to similate sklearn D-tree

        # self.P = self.Probability().cuda()

        if idxs is None: 
            idxs = range(len(h))

        self.N = len(idxs)

        # reboot the nodes and leaves
        
        self.mu = torch.ones(len(idxs),self.nLeaves).cuda()
        
        # plant the root node:(self, x, y, idxs=None, depth = 2, n_epochs = 100)
        self.node(h, mu = torch.ones([len(idxs),1]).cuda(), depth = 0)
        
        # calculate new probability
        self.P = self.Probability(self.mu)
        
        # P_log = torch.log(self.P)
        return self.P

    def node(self, h, mu = None, depth = 0):

        if mu is None:
            mu = torch.ones([len(self.idxs),1]).cuda()

        counter = self.Nodes
        self.Nodes += 1

        d = torch.sigmoid(self.theta[counter](h[self.idxs])).cuda() 

        l_mu = torch.mul(mu,d)
        
        r_mu = torch.mul(mu,(1-d))

        if(depth<self.max_depth-1):
            self.node(h=h, mu = l_mu, depth = depth+1)
            self.node(h=h, mu = r_mu, depth = depth+1)
        else:
            self.leaf(mu = l_mu)
            self.leaf(mu = r_mu)
            
    def leaf(self, mu):
        counter = self.Leaves
        self.Leaves += 1 

        self.mu[self.idxs,counter] = mu[:,0]

    def iter_pi(self,y):
        pi = torch.zeros(self.pi.size()).cuda()
        z = torch.zeros(self.nLeaves).cuda()
        # print(f'self.pi {self.pi}')
        # print(f'self.mu {self.mu}')
        # print(f'self.P {self.P}')

        for l in range(self.nLeaves):
            for i in range(self.N):
                if(self.P[i,y[i]]!=0): #self.mu[i,l]!=0 or 
                    pi[l,y[i]] = pi[l,y[i]].clone() + self.pi[l,y[i]].clone()*self.mu[i,l].clone()/self.P[i,y[i]].clone() 
                    # print(f'pi[l,y[i]] {pi[l,y[i]]}')
            z[l] = torch.sum(pi[l,:],0).cuda()
            # print(f'before normalizing: pi[l,:] {pi[l,:]}')
            pi[l,:] = pi[l,:].clone()/z[l].clone()
            # print(f'after normalizing: pi[l,:] {pi[l,:]}')
        self.pi = pi

    def Probability(self,mu):
        return mu.clone().mm(self.pi) #return self.mu.mm(self.pi) 

    #calculating number of nodes from tree depth:
    def node_calc(self):
        nNodes = 0
        for i in range(self.max_depth): nNodes += 2**i
        return nNodes