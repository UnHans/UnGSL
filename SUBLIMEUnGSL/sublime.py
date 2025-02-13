import dgl
import torch
import torch.nn as nn
from opengsl.module.encoder import GCNEncoder, APPNPEncoder, GINEncoder
import dgl.function as fn
import numpy as np
import torch.nn.functional as F
import copy
import torch.nn as nn

EOS = 1e-10


def get_feat_mask(features, mask_rate):
    feat_node = features.shape[1]
    mask = torch.zeros(features.shape)
    samples = np.random.choice(feat_node, size=int(feat_node * mask_rate), replace=False)
    mask[:, samples] = 1
    return mask.cuda(), samples


def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list



def dgl_graph_to_torch_sparse_cuda(dgl_graph):
    values = dgl_graph.edata['w']
    rows_, cols_ = dgl_graph.edges()
    indices = torch.cat((torch.unsqueeze(rows_, 0), torch.unsqueeze(cols_, 0)), 0)
    torch_sparse_mx = torch.sparse.FloatTensor(indices, values)
    return torch_sparse_mx
def torch_sparse_to_dgl_graph_cuda(torch_sparse_mx):
    torch_sparse_mx = torch_sparse_mx.coalesce()
    indices = torch_sparse_mx.indices()
    values = torch_sparse_mx.values()
    rows_, cols_ = indices[0,:], indices[1,:]
    dgl_graph = dgl.graph((rows_, cols_), num_nodes=torch_sparse_mx.shape[0], device='cuda:0')
    dgl_graph.edata['w'] = torch.tensor(values,device = "cuda:0")
    return dgl_graph
def dgl_graph_to_torch_sparse(dgl_graph):
    values = dgl_graph.edata['w'].cpu().detach()
    rows_, cols_ = dgl_graph.edges()
    indices = torch.cat((torch.unsqueeze(rows_, 0), torch.unsqueeze(cols_, 0)), 0).cpu()
    torch_sparse_mx = torch.sparse.FloatTensor(indices, values)
    return torch_sparse_mx


def torch_sparse_to_dgl_graph(torch_sparse_mx):
    torch_sparse_mx = torch_sparse_mx.coalesce()
    indices = torch_sparse_mx.indices()
    values = torch_sparse_mx.values()
    rows_, cols_ = indices[0,:], indices[1,:]
    dgl_graph = dgl.graph((rows_, cols_), num_nodes=torch_sparse_mx.shape[0], device='cuda:0')
    dgl_graph.edata['w'] = torch.tensor(values.detach(),device = "cuda:0")
    return dgl_graph


class GCNConv_dgl(nn.Module):
    # to be changed to pyg in future versions
    def __init__(self, input_size, output_size):
        super(GCNConv_dgl, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x, g):
        with g.local_scope():
            g.ndata['h'] = self.linear(x)
            g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum(msg='m', out='h'))
            return g.ndata['h']


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
        
        """Load node confidence vector"""
        Entropy=None
        confidence_vector=None
        device="cuda:0"
        if conf.dataset["name"]=="blogcatalog":
            confidence_vector=torch.load("/home/hs/NOpenGSL/"+"SUBLIMEBlogCLLossConf"+".pt",map_location=device)
        self.confidence_vector=torch.tensor(confidence_vector,device = device)
        #self.confidence_matrix = confidence_vector.view(-1, 1).expand(-1, len(confidence_vector)).t().to("cuda:6")
        print(self.confidence_vector.size(),self.confidence_vector[:5])
    def forward(self,learned_adj):
        # learned_adj=dgl_graph_to_torch_sparse_cuda(learned_adj)
        learned_adj = learned_adj.to_sparse()
        indices= learned_adj._indices()
        values = learned_adj._values()
        dst = indices[1, :]
        confidence_values = self.confidence_vector[dst]
        row_indices = indices[0]
        weight = torch.sigmoid(confidence_values - self.thresholds[row_indices].flatten())/0.5
        masks=torch.where(weight>=1,weight,self.Beta)
        new_values=values*masks
        print(self.thresholds[:10])
        tensor_learned_adj=torch.sparse_coo_tensor(indices, new_values,learned_adj.shape)
        learned_adj = tensor_learned_adj.to_dense()
        # print(len(tensor_learned_adj._values()))
        # learned_adj=torch_sparse_to_dgl_graph_cuda(tensor_learned_adj)
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


        """Load node confidence vector"""
        device="cuda:0"
        if conf.dataset["name"]=="pubmed":
            confidence_vector=torch.load("/home/hs/OpenGSL/"+"SUBLIMEPubmedCLLossConf"+".pt",map_location=device)
        elif conf.dataset["name"]=="roman-empire":
            confidence_vector=torch.load("/home/hs/OpenGSL/"+"SUBLIMERomanCLLossConf"+".pt",map_location=device)
        elif conf.dataset["name"]=="blogcatalog":
            confidence_vector=torch.load("/home/hs/NOpenGSL/"+"SUBLIMEBlogCLLossConf"+".pt",map_location=device)
        elif conf.dataset["name"]=="cora":
            confidence_vector=torch.load("/home/hs/NOpenGSL/"+"SUBLIMECoraCLLossConf"+".pt",map_location=device)        
        self.confidence_matrix = torch.tensor(confidence_vector.view(-1, 1).expand(-1, len(confidence_vector)).t(),device = device)
        print(self.confidence_matrix.size(),self.confidence_matrix[95:97,:5])
        self.AdjThreshold=conf.AdjThreshold
    def forward(self,learned_adj):
        # learned_adj=dgl_graph_to_torch_sparse_cuda(learned_adj)
        if learned_adj.is_sparse:
            learned_adj=learned_adj.to_dense()
        confidence_matrix=self.confidence_matrix * ( (learned_adj>self.AdjThreshold).int())
        weight = torch.sigmoid(confidence_matrix-self.thresholds)/0.5
        mask= torch.where(weight>=1,weight,self.Beta)
        learned_adj=learned_adj*mask
        # learned_adj=learned_adj.to_sparse()
        # learned_adj=torch_sparse_to_dgl_graph_cuda(learned_adj)
        return learned_adj





class GraphEncoder(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, sparse, conf=None):

        super(GraphEncoder, self).__init__()
        self.dropout = dropout
        self.sparse = sparse
        self.gnn_encoder_layers = nn.ModuleList()
        if sparse:
            self.gnn_encoder_layers.append(GCNConv_dgl(in_dim, hidden_dim))
            for _ in range(nlayers - 2):
                self.gnn_encoder_layers.append(GCNConv_dgl(hidden_dim, hidden_dim))
            self.gnn_encoder_layers.append(GCNConv_dgl(hidden_dim, emb_dim))
        else:
            if conf.model['type']=='gcn':
                self.model = GCNEncoder(nfeat=in_dim, nhid=hidden_dim, nclass=emb_dim, n_layers=nlayers, dropout=dropout,
                                 input_layer=False, output_layer=False, spmm_type=0)
            elif conf.model['type']=='appnp':
                self.model = APPNPEncoder(in_dim, hidden_dim, emb_dim,
                                    dropout=dropout, K=conf.model['K'],
                                    alpha=conf.model['alpha'])
            elif conf.model['type'] == 'gin':
                self.model = GINEncoder(in_dim, hidden_dim, emb_dim,
                               nlayers, conf.model['mlp_layers'])
        self.proj_head = nn.Sequential(nn.Linear(emb_dim, proj_dim), nn.ReLU(inplace=True),
                                           nn.Linear(proj_dim, proj_dim))

    def forward(self, x, Adj_):

        if self.sparse:
            for conv in self.gnn_encoder_layers[:-1]:
                x = conv(x, Adj_)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.gnn_encoder_layers[-1](x, Adj_)
        else:
            x = self.model(x, Adj_)
        #print(x)
        z = self.proj_head(x)
        return z, x


class GCL(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj, sparse, conf=None):
        super(GCL, self).__init__()

        self.encoder = GraphEncoder(nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, sparse, conf)
        self.dropout_adj = dropout_adj
        self.sparse = sparse

    def forward(self, x, Adj_, branch=None):

        # edge dropping
        if self.sparse:
            if branch == 'anchor':
                Adj = copy.deepcopy(Adj_)
            else:
                Adj = Adj_
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj, training=self.training)
        else:
            Adj = F.dropout(Adj_, p=self.dropout_adj, training=self.training)

        # get representations
        #print(type(x))
        z, embedding = self.encoder(x, Adj)
        return z, embedding

    @staticmethod
    def calc_loss(x, x_aug, temperature=0.2):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)   # 计算的是cos相似度
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

        # all_loss0= - torch.log(loss_0).detach()
        # all_loss1= - torch.log(loss_1).detach()
        # allCLloss=(all_loss0+all_loss1)/2.0


        loss_0 = - torch.log(loss_0).mean()
        loss_1 = - torch.log(loss_1).mean()
        loss = (loss_0 + loss_1) / 2.0
        return loss#,allCLloss


class GCN_SUB(nn.Module):
    # TODO
    # to be changed to pyg in future versions
    def __init__(self, nfeat, nhid, nclass, n_layers=5, dropout=0.5, dropout_adj=0.5, sparse=0):
        super(GCN_SUB, self).__init__()
        self.layers = nn.ModuleList()
        self.sparse = sparse
        self.dropout_adj_p = dropout_adj
        self.dropout = dropout

        if sparse:
            
            self.layers.append(GCNConv_dgl(nfeat, nhid))
            for _ in range(n_layers - 2):
                self.layers.append(GCNConv_dgl(nhid, nhid))
            self.layers.append(GCNConv_dgl(nhid, nclass))
        else:
            self.model = GCNEncoder(nfeat=nfeat, nhid=nhid, nclass=nclass, n_layers=n_layers, dropout=dropout,
                             input_layer=False, output_layer=False, spmm_type=0)

    def forward(self, x, Adj):

        if self.sparse:
            Adj = copy.deepcopy(Adj)
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj_p, training=self.training)
        else:
            Adj = F.dropout(Adj, p=self.dropout_adj_p, training=self.training)

        if self.sparse:
            for i, conv in enumerate(self.layers[:-1]):
                x = conv(x, Adj)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.layers[-1](x, Adj)
            #print(x.size(),CAL_FEATURES.size())
            return x.squeeze(1)#,CAL_FEATURES
        else:
            #print("AAAAAAAAAAAAA")
            return self.model(x, Adj)


if __name__ == '__main__':
    pass




