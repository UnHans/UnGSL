import torch
import torch.nn as nn
from opengsl.module.functional import normalize
from opengsl.module.encoder import GCNDiagEncoder, GCNEncoder, APPNPEncoder, GINEncoder
from opengsl.module.fuse import Interpolate
from opengsl.module.transform import Normalize, KNN, Symmetry
from opengsl.module.metric import InnerProduct
import torch.nn.functional as F
def remove_top_k_percent(adj, k_percent):
    print("drop %",k_percent)
    # Clone the adjacency matrix to avoid in-place modifications
    adj_copy = adj.clone()

    # Calculate the number of elements to remove per row
    num_elements_to_remove = (adj != 0).sum(dim=1).float() * (k_percent / 100.0)
    num_elements_to_remove = num_elements_to_remove.int()  # Convert to int

    # Iterate through each row and remove top K percent of non-zero elements
    for i in range(adj.size(0)):
        # Get the non-zero elements and their indices
        row = adj_copy[i]
        non_zero_values = row[row != 0]
        non_zero_indices = torch.nonzero(row != 0, as_tuple=True)[0]

        if len(non_zero_values) > 0:
            # Find the indices of the top K percent highest values
            num_remove = num_elements_to_remove[i].item()
            if num_remove > 0:
                top_k_values, top_k_indices = torch.topk(non_zero_values, num_remove, largest=True)
                
                # Set the top K percent values to zero
                row[non_zero_indices[top_k_indices]] = 0.0

    return adj_copy
class BetaReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input,Beta):

        ctx.save_for_backward(input)
        return torch.where(input>=1.,input,torch.tensor(Beta) )

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <1.] = 0
        return grad_input,None


class MyRelu(nn.Module):
    def __init__(self,Beta):
        super().__init__()
        self.Beta=Beta
    def forward(self, x):
        out = BetaReLU.apply(x,self.Beta)
        return out

class GRCN(torch.nn.Module):

    def __init__(self, num_nodes, num_features, num_classes, device, conf,run_time=None):
        super(GRCN, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        if conf.model['type'] == 'gcn':
            self.conv_task = GCNEncoder(num_features, conf.model['n_hidden'], num_classes, conf.model['n_layers'],
                                 conf.model['dropout'], conf.model['input_dropout'], conf.model['norm'],
                                 conf.model['n_linear'], conf.model['spmm_type'], conf.model['act'],
                                 conf.model['input_layer'], conf.model['output_layer'])
        elif conf.model['type'] == 'appnp':
            self.conv_task = APPNPEncoder(num_features, conf.model['n_hidden'], num_classes,
                               dropout=conf.model['dropout'], K=conf.model['K_APPNP'],
                               alpha=conf.model['alpha'], spmm_type=1)
        elif conf.model['type'] == 'gin':
            self.conv_task = GINEncoder(num_features, conf.model['n_hidden'], num_classes,
                               conf.model['n_layers'], conf.model['mlp_layers'], spmm_type=1)
        self.model_type = conf.gsl['model_type']
        if conf.gsl['model_type'] == 'diag':
            self.conv_graph = GCNDiagEncoder(2, num_features)
        else:
            self.conv_graph = GCNEncoder(num_features, conf.gsl['n_hidden_1'], conf.gsl['n_hidden_2'], conf.gsl['n_layers'],
                             conf.gsl['dropout'], conf.gsl['input_dropout'], conf.gsl['norm'],
                             conf.gsl['n_linear'], conf.gsl['spmm_type'], conf.gsl['act'],
                             conf.gsl['input_layer'], conf.gsl['output_layer'])

        self.K = conf.gsl['K']
        self._normalize = conf.gsl['normalize']   # 用来决定是否对node embedding进行normalize
        self.metric = InnerProduct()
        self.normalize_a = Normalize(add_loop=False)
        self.normalize_e = Normalize('row-norm', p=2)
        self.knn = KNN(self.K, sparse_out=True)
        self.sym = Symmetry(1)
        self.fuse = Interpolate(1, 1)
        delta=torch.nn.Parameter(torch.FloatTensor(num_nodes,1))
        print(delta.is_leaf)
        delta.data.fill_(conf.training['init_value'])
        self.delta=delta
        self.BetaRelu=MyRelu(Beta=conf.training["beta"])
        self.Beta = conf.training["beta"]
        self.run_time=run_time
        print(self.run_time)
        Entropy=None
        device="cuda:0"
        if conf.training["dataset"]=="blogcatalog":
            Entropy=torch.load("/home/hs/OpenGSL/"+"GRCNBlogEntropy"+".pt",map_location=device)
        elif conf.training["dataset"]=="roman":
            Entropy=torch.load("/home/hs/OpenGSL/"+"GRCNRomanEntropy"+".pt",map_location=device)
        elif conf.training["dataset"]=="citeseer":
            Entropy=torch.load("/home/hs/OpenGSL/"+"GRCNCiteseerEntropy"+".pt",map_location=device)
        elif conf.training["dataset"]=="cora":
            Entropy=torch.load("/home/hs/OpenGSL/"+"GRCNCoraEntropy"+".pt",map_location=device)
        confidence_vector=torch.exp( -Entropy )
        self.confidence_matrix = confidence_vector.view(-1, 1).expand(-1, len(confidence_vector)).t().to(device)
        print(self.confidence_matrix.size())
    def graph_parameters(self):
        return list(self.conv_graph.parameters())

    def base_parameters(self):
        return list(self.conv_task.parameters())
    def delta_parameters(self):
        return [self.delta]

    def cal_similarity_graph(self, node_embeddings):
        # 一个2head的相似度计算
        # 完全等价于普通cosine
        similarity_graph = self.metric(node_embeddings[:, :int(self.num_features/2)])
        similarity_graph += self.metric(node_embeddings[:, int(self.num_features/2):])
        return similarity_graph

    def _sparse_graph(self, raw_graph,uncertainty=None):
        new_adj = self.knn(adj=raw_graph)
        new_adj = self.sym(new_adj)
        if uncertainty is not None:
            confidence = torch.exp(-uncertainty).view(-1, 1).expand(-1, len(uncertainty)).t()
            new_adj=(new_adj.to_dense())
            confidence_matrix=confidence * ( (new_adj>0).int()) 
            weight = torch.sigmoid(confidence_matrix-self.delta)/0.5
            mask= torch.where(weight>=1,weight,self.Beta)
            new_adj=new_adj*mask
            new_adj=new_adj.to_sparse()
        return new_adj

    def _node_embeddings(self, input, Adj):
        norm_Adj = self.normalize_a(Adj)
        node_embeddings = self.conv_graph(input, norm_Adj)
        if self._normalize:
            node_embeddings = self.normalize_e(node_embeddings)
        return node_embeddings

    def forward(self, input, Adj,Test=False,uncertainty=None,Edge_variance=None,drop_ratio=0,entropy=None):
        adjs = {}
        Adj.requires_grad = False
        node_embeddings = self._node_embeddings(input, Adj)
        Adj_new = self.cal_similarity_graph(node_embeddings)
        Adj_new= self._sparse_graph(Adj_new,uncertainty=uncertainty)
        Adj_final = self.fuse(Adj_new, Adj)
        # if Test:
        #     Adj_final=Adj_final.to_dense()
        #     Adj_final=Adj_final.to_dense()
        #     entropy=1/torch.sum(Adj_final,dim=1)
        #     entropy=entropy * (Adj_final>0)
        #     mask=remove_top_k_percent(entropy,drop_ratio)
        #     mask=mask>0
        #     # print(torch.sum(mask>0,dim=1).flatten())
        #     Adj_final=(Adj_final*mask).to_sparse()
        # if Test:
        #     Adj_final=Adj_final.to_dense()
            # Adj_final=Adj_final.to_dense()
            # entropy=entropy * (Adj_final>0)
            # mask=remove_top_k_percent(entropy,drop_ratio)
            # mask=mask>0
            # # print(torch.sum(mask>0,dim=1).flatten())
            # Adj_final=(Adj_final*mask).to_sparse()
        Adj_final_norm = self.normalize_a(Adj_final.coalesce())
        x = self.conv_task(input, Adj_final_norm)
        adjs['new'] = Adj_new
        adjs['final'] = Adj_final

        return x, adjs, Adj_new
