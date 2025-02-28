import torch
"""Set the optimizer for learnable thresholds in UnGSL"""
self.optim_ungsl= torch.optim.Adam(self.model.ungsl_parameters(), lr=self.conf.training["ungsl_lr"])

class SparseUnGSL(nn.Module):
    def __init__(self, dataset=None,conf=None):
        super(SparseUnGSL, self).__init__()
        """Set node-wise learnable thresholds"""
        thresholds=torch.nn.Parameter(torch.FloatTensor(dataset.n_nodes,1))
        print(thresholds.is_leaf)
        thresholds.data.fill_(conf.training['init_value'])
        self.thresholds=thresholds
        print(thresholds.is_leaf)

        self.Beta = conf.training["beta"]
        
        """Load node uncertainty vector"""
        Entropy=None
        device="cuda:0"
        if conf.dataset["name"]=="cora":
            Entropy=torch.load("/home/hs/OpenGSL/"+"GSLCoraEntropy"+".pt",map_location=device)
        self.confidence_vector=torch.tensor(torch.exp(-Entropy),device = device)
    def forward(self,learned_adj):
        learned_adj = learned_adj.to_sparse()
        indices= learned_adj._indices()
        values = learned_adj._values()
        dst = indices[1, :]
        confidence_values = self.confidence_vector[dst]
        row_indices = indices[0]
        weight = torch.sigmoid(confidence_values - self.thresholds[row_indices].flatten())/0.5
        masks=torch.where(weight>=1,weight,self.Beta)
        """you can print self.thresholds[:10]  to verify that the UnGSL is working correctly."""
        new_values=values*masks
        tensor_learned_adj=torch.sparse_coo_tensor(indices, new_values,learned_adj.shape)
        learned_adj = tensor_learned_adj.to_dense()
        return learned_adj

class UnGSL(nn.Module):
    def __init__(self, dataset=None,conf=None):
        super(UnGSL, self).__init__()

        """Set node-wise learnable thresholds"""
        thresholds=torch.nn.Parameter(torch.FloatTensor(dataset.n_nodes,1))
        print(thresholds.is_leaf)
        thresholds.data.fill_(conf.training['init_value'])
        self.thresholds=thresholds
        print(thresholds.is_leaf)

        self.Beta = conf.training["beta"]

        """Load node uncertainty vector"""
        device="cuda:0"
        Entropy=None
        if conf.dataset["name"]=="cora":
            Entropy=torch.load("/home/hs/OpenGSL/"+"GSLCoraEntropy"+".pt",map_location=device)      
        self.confidence_matrix = torch.tensor(torch.exp(-Entropy).view(-1, 1).expand(-1, len(Entropy)).t(),device = device)
    def forward(self,learned_adj):
        if learned_adj.is_sparse:
            learned_adj=learned_adj.to_dense()
        confidence_matrix=self.confidence_matrix * ( (learned_adj>0).int())
        weight = torch.sigmoid(confidence_matrix-self.thresholds)/0.5
        """you can print self.thresholds[:10]  to verify that the UnGSL is working correctly."""
        mask= torch.where(weight>=1,weight,self.Beta)
        learned_adj=learned_adj*mask
        return learned_adj