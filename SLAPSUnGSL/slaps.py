import torch
import torch.nn.functional as F
# from .GCN3 import GraphConvolution, GCN
import math
import dgl
import numpy as np
from opengsl.module.encoder import GCNEncoder, APPNPEncoder
from opengsl.module.functional import apply_non_linearity, normalize, symmetry, knn
from opengsl.module.metric import InnerProduct
from opengsl.module.transform import KNN
import torch.nn as nn

class SparseUnGSL(nn.Module):
    def __init__(self, dataset=None,conf=None):
        super(SparseUnGSL, self).__init__()
        """Set node-wise thresholds"""
        thresholds=torch.nn.Parameter(torch.FloatTensor(dataset.n_nodes,1))
        print(thresholds.is_leaf)
        thresholds.data.fill_(conf.training['init_value'])
        self.thresholds=thresholds
        print(thresholds.is_leaf)
        self.Beta = conf.training["beta"]
        Entropy=None
        device="cuda:7"
        if conf.dataset["name"]=="cora":
            Entropy=torch.load("/home/hs/OpenGSL/"+"SLAPSCoraEntropy"+".pt",map_location=device)
        elif conf.dataset["name"]=="citeseer":
            Entropy=torch.load("/home/hs/OpenGSL/"+"SLAPSCiteseerEntropy"+".pt",map_location=device)
        elif conf.dataset["name"]=="roman-empire":
            Entropy=torch.load("/home/hs/NOpenGSL/"+"SLAPSRomanEntropy"+".pt",map_location=device)
        elif conf.dataset["name"]=="pubmed":
            Entropy=torch.load("/home/hs/NOpenGSL/"+"SLAPSPubmedEntropy"+".pt",map_location=device)
        elif conf.dataset["name"]=="flickr":
            Entropy=torch.load("/home/hs/NOpenGSL/"+"SLAPSFlickrEntropy"+".pt",map_location=device)
        self.confidence_vector=torch.exp( -Entropy ).to(device)
        #self.confidence_matrix = confidence_vector.view(-1, 1).expand(-1, len(confidence_vector)).t().to("cuda:6")
        print(self.confidence_vector.size())
    def forward(self,learned_adj):
        learned_adj = learned_adj.to_sparse()
        indices= learned_adj.indices()
        values = learned_adj.values()
        dst = indices[1, :]
        confidence_values = self.confidence_vector[dst]
        row_indices = indices[0]
        weight = torch.sigmoid(confidence_values - self.thresholds[row_indices].flatten())/0.5
        masks = torch.where(weight>=1, weight,self.Beta)
        new_values=values*masks
        tensor_learned_adj=torch.sparse_coo_tensor(indices, new_values,learned_adj.shape)
        new_learned_adj = tensor_learned_adj.to_dense()
        return new_learned_adj

class MLP(torch.nn.Module):
    def __init__(self, nlayers, isize, hsize, osize, features, mlp_epochs, k, knn_metric, non_linearity, i, mlp_act):
        super(MLP, self).__init__()

        self.layers = torch.nn.ModuleList()
        if nlayers == 1:
            self.layers.append(torch.nn.Linear(isize, hsize))
        else:
            self.layers.append(torch.nn.Linear(isize, hsize))
            for _ in range(nlayers - 2):
                self.layers.append(torch.nn.Linear(hsize, hsize))
            self.layers.append(torch.nn.Linear(hsize, osize))

        self.input_dim = isize
        self.output_dim = osize
        self.features = features
        self.mlp_epochs = mlp_epochs
        self.k = k
        self.knn_metric = knn_metric
        self.non_linearity = non_linearity
        self.i = i
        self.mlp_act = mlp_act
        self.mlp_knn_init()

    def internal_forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                if self.mlp_act == "relu":
                    h = F.relu(h)
                elif self.mlp_act == "tanh":
                    h = F.tanh(h)
        return h

    def mlp_knn_init(self):
        self.layers.to(self.features.device)
        if self.input_dim == self.output_dim:
            print("MLP full")
            for layer in self.layers:
                layer.weight = torch.nn.Parameter(torch.eye(self.input_dim))
        else:
            optimizer = torch.optim.Adam(self.parameters(), 0.01)
            labels = KNN(self.k,metric=self.knn_metric)(self.features)
            for epoch in range(1, self.mlp_epochs):
                self.train()
                logits = self.forward(self.features)
                loss = F.mse_loss(logits, labels, reduction='sum')
                if epoch % 10 == 0:
                    print("MLP loss", loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def forward(self, features):
        embeddings = self.internal_forward(features)
        embeddings = F.normalize(embeddings, dim=1, p=2)
        similarities = InnerProduct()(embeddings)
        similarities = knn(similarities, self.k + 1)
        similarities = apply_non_linearity(similarities, self.non_linearity, self.i)
        return similarities

class GCN_DAE(torch.nn.Module):
    def __init__(self, cfg_model, nlayers, in_dim, hidden_dim, nclasses, dropout, dropout_adj, features, k, knn_metric, i_,
                 non_linearity, normalization, mlp_h, mlp_epochs, mlp_act):
        super(GCN_DAE, self).__init__()

        if cfg_model['type'] == 'gcn':
            self.layers = GCNEncoder(in_dim, hidden_dim, nclasses, n_layers=nlayers, dropout=dropout, spmm_type=0)
        elif cfg_model['type'] == 'appnp':
            self.layers = APPNPEncoder(in_dim, hidden_dim, nclasses, spmm_type=1,
                               dropout=dropout, K=cfg_model['appnp_k'], alpha=cfg_model['appnp_alpha'])

        self.dropout_adj = torch.nn.Dropout(p=dropout_adj)
        self.normalization = normalization

        self.graph_gen = MLP(2, features.shape[1], math.floor(math.sqrt(features.shape[1] * mlp_h)),
                                mlp_h, features, mlp_epochs, k, knn_metric, non_linearity, i_,
                                mlp_act).to("cuda")

    def get_adj(self, h):
        Adj_ = self.graph_gen(h)
        Adj_ = symmetry(Adj_)
        Adj_ = normalize(Adj_, self.normalization, False)
        return Adj_

    def forward(self, features, x,UnGSL=None):  # x corresponds to masked_features
        Adj_ = self.get_adj(features)

        if UnGSL is not None:
            Adj_  =Adj_.to_sparse()
            Adj_=UnGSL(Adj_)
        Adj = self.dropout_adj(Adj_).to_sparse()
        x = self.layers(x, Adj)

        return x, Adj_

class GCN_C(torch.nn.Module):
    def __init__(self, cfg_model, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj):
        super(GCN_C, self).__init__()

        if cfg_model['type'] == 'gcn':
            self.layers = GCNEncoder(in_channels, hidden_channels, out_channels, n_layers=num_layers, dropout=dropout, spmm_type=0)
        elif cfg_model['type'] == 'appnp':
            self.layers = APPNPEncoder(in_channels, hidden_channels, out_channels, spmm_type=1,
                               dropout=dropout, K=cfg_model['appnp_k'], alpha=cfg_model['appnp_alpha'])

        self.dropout_adj = torch.nn.Dropout(p=dropout_adj)

    def forward(self, x, adj_t):
        adj_t = adj_t
        Adj = self.dropout_adj(adj_t).to_sparse()
        x = self.layers(x, Adj)
        return x



class SLAPS(torch.nn.Module):
    def __init__(self, num_nodes, num_features, num_classes, features, device, conf):
        super(SLAPS, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_classes = num_classes
        self.device = device
        self.conf = conf

        self.gcn_dae = GCN_DAE(self.conf.model, nlayers=self.conf.model['nlayers_adj'], in_dim=num_features, hidden_dim=self.conf.model['hidden_adj'], nclasses=num_features,
                             dropout=self.conf.model['dropout1'], dropout_adj=self.conf.model['dropout_adj1'],
                             features=features, k=self.conf.model['k'], knn_metric=self.conf.model['knn_metric'], i_=self.conf.model['i'],
                             non_linearity=self.conf.model['non_linearity'], normalization=self.conf.model['normalization'], mlp_h=self.num_features,
                             mlp_epochs=self.conf.model['mlp_epochs'], mlp_act=self.conf.model['mlp_act'])
        self.gcn_c = GCN_C(self.conf.model, in_channels=num_features, hidden_channels=self.conf.model['hidden'], out_channels=num_classes,
                            num_layers=self.conf.model['nlayers'], dropout=self.conf.model['dropout2'], dropout_adj=self.conf.model['dropout_adj2'])


    def forward(self, features,UnGSL=None):
        loss_dae, Adj = self.get_loss_masked_features(features,UnGSL=UnGSL)
        logits = self.gcn_c(features, Adj)
        if len(logits.shape) > 1:
            logits = logits.squeeze(1)
        
        return logits, loss_dae, Adj

    def get_loss_masked_features(self, features,UnGSL=None):
        if self.conf.dataset['feat_type'] == 'binary':
            mask = self.get_random_mask_binary(features, self.conf.training['ratio'], self.conf.training['nr'])
            masked_features = features * (1 - mask)

            logits, Adj = self.gcn_dae(features, masked_features,UnGSL)
            indices = mask > 0
            loss = F.binary_cross_entropy_with_logits(logits[indices], features[indices], reduction='mean')
        elif self.conf.dataset['feat_type'] == 'continuous':
            mask = self.get_random_mask_continuous(features, self.conf.training['ratio'])
            # noise = torch.normal(0.0, 1.0, size=features.shape).cuda()
            # masked_features = features + (noise * mask)
            masked_features = features * (1 - mask)

            logits, Adj = self.gcn_dae(features, masked_features,UnGSL)
            indices = mask > 0
            loss = F.binary_cross_entropy_with_logits(logits[indices], features[indices], reduction='mean')
        else:
            raise ValueError("Wrong feat_type in dataset_configure.")
    
        return loss, Adj
    
    def get_random_mask_binary(self, features, r, nr):
        nones = torch.sum(features > 0.0).float()
        nzeros = features.shape[0] * features.shape[1] - nones
        pzeros = nones / nzeros / r * nr
        probs = torch.zeros(features.shape, device='cuda:7')
        probs[features == 0.0] = pzeros
        probs[features > 0.0] = 1 / r
        mask = torch.bernoulli(probs)
        return mask

    def get_random_mask_continuous(self, features, r):
        probs = torch.full(features.shape, 1 / r, device='cuda:7')
        mask = torch.bernoulli(probs)
        return mask
