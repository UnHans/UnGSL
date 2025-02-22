# import dgl
import torch
import torch.nn as nn

from layers import AnchorGCNLayer
from utils import *


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

class AnchorUnGSL(nn.Module):
    def __init__(self,dataset=None,init_value=None,beta=None):
        super(AnchorUnGSL, self).__init__()
        num_nodes=0
        """Load node entropy vector and transform it to confidence matrix"""
        if dataset=="cora":
            Entropy = torch.load("PROSECoraEntropy.pt")
            num_nodes=Entropy.size()[0]
        elif dataset=="citeseer":
            Entropy=torch.load("PROSECiteseerEntropy.pt") 
            num_nodes=Entropy.size()[0]
        elif dataset=="pubmed":
            Entropy=torch.load("PROSEPubmedEntropy.pt") 
            num_nodes=Entropy.size()[0]
        self.Beta=beta
        confidence_vector = torch.exp( -Entropy.detach() )
        print(confidence_vector.size())
        self.confidence_matrix = confidence_vector.view(-1, 1).expand(-1, len(confidence_vector)).t().to("cuda:0")

        """Set node-wise learnable thresholds"""
        thresholds=torch.nn.Parameter(torch.FloatTensor(num_nodes,1))
        print(thresholds.is_leaf)
        thresholds.data.fill_(init_value)
        print(init_value)
        self.thresholds=thresholds

    def forward(self,learned_adj,anchor_node_idx=None):
        if learned_adj.is_sparse:
            learned_adj=learned_adj.to_dense()
        if anchor_node_idx is not None:
            confidence_matrix=self.confidence_matrix[:,anchor_node_idx]
        confidence_matrix=confidence_matrix * ( (learned_adj>=0).int())
        weights = torch.sigmoid(confidence_matrix-self.thresholds)/0.5
        masks = torch.where(weights>=1, weights, self.Beta)
        learned_adj=learned_adj * masks
        if learned_adj.is_sparse:
            learned_adj=learned_adj.to_sparse()
        return learned_adj

class Stage_GNN_learner(nn.Module):
    def __init__(self, isize, osize, head_num, sparse, ks, anchor_adj_fusion_ratio, epsilon):
        super(Stage_GNN_learner, self).__init__()


        self.weight_tensor1 = torch.Tensor(head_num, isize)
        self.weight_tensor1 = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor1))

        self.weight_tensor2 = torch.Tensor(head_num, osize)
        self.weight_tensor2 = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor2))

        self.sparse = sparse
        # self.act = act
        self.anchor_adj_fusion_ratio = anchor_adj_fusion_ratio
        self.epsilon = epsilon
        ## stage module
        self.ks = ks
        self.l_n = len(self.ks)

        if self.l_n > 0:
            # down score
            self.score_layer = AnchorGCNLayer(isize, 1)

    def build_epsilon_neighbourhood(self, attention, epsilon, markoff_value):
        mask = (attention > epsilon).detach().float()
        weighted_adjacency_matrix = attention * mask + markoff_value * (1 - mask)
        return weighted_adjacency_matrix


    def knn_anchor_node(self, context, anchors, weight_tensor, k = 100, b = 500):
        expand_weight_tensor = weight_tensor.unsqueeze(1) # 6,1,32
        if len(context.shape) == 3:
            expand_weight_tensor = expand_weight_tensor.unsqueeze(1)
            
        # context -  N * 32
        # context.unsqueeze(0) - 1 * N * 32
        context_fc = context.unsqueeze(0) * expand_weight_tensor
        # context_fc - 6 * N * 32
        context_norm = F.normalize(context_fc, p=2, dim=-1)
        # context_norm - 6 * N * 32

        anchors_fc = anchors.unsqueeze(0) * expand_weight_tensor
        anchors_norm = F.normalize(anchors_fc, p=2, dim=-1)  # 6 * anchor_num * 32

        attention = torch.matmul(context_norm, anchors_norm.transpose(-1, -2)).mean(0)
        markoff_value = 0

        # index = 0
        # values = torch.zeros(context_norm.shape[1] * (k + 1)).cuda()
        # rows = torch.zeros(context_norm.shape[1] * (k + 1)).cuda()
        # cols = torch.zeros(context_norm.shape[1] * (k + 1)).cuda()
        # # norm_row = torch.zeros(context_norm.shape[1]).cuda()
        # # norm_col = torch.zeros(context_norm.shape[1]).cuda()
        # while index < context_norm.shape[1]:
        #     if (index + b) > (context_norm.shape[1]):
        #         end = context_norm.shape[1]
        #     else:
        #         end = index + b
        #     sub_tensor = context_norm[:,index:index + b,:]
        #     # similarities = torch.matmul(sub_tensor, context_norm.transpose(-1, -2)).mean(0)
        #     similarities = torch.matmul(sub_tensor, anchors_norm.transpose(-1, -2)).mean(0)
        #     #------start---------
        #     similarities_ = self.build_epsilon_neighbourhood(similarities, 0.1, markoff_value)
        #     # or inds
        #     # #-------end--------
        #     vals, inds = similarities_.topk(k=k + 1, dim=-1)
        #     values[index * (k + 1):(end) * (k + 1)] = vals.view(-1)
        #     cols[index * (k + 1):(end) * (k + 1)] = inds.view(-1)
        #     rows[index * (k + 1):(end) * (k + 1)] = torch.arange(index, end).view(-1, 1).repeat(1, k + 1).view(-1)
        #     # norm_row[index: end] = torch.sum(vals, dim=1)
        #     # norm_col.index_add_(-1, inds.view(-1), vals.view(-1))
        #     index += b
        # rows = rows.long()
        # cols = cols.long()

        # rows_ = torch.cat((rows, cols))
        # cols_ = torch.cat((cols, rows))
        # values_ = torch.cat((values, values))
        # values_ = F.relu(values)
        # indices = torch.cat((torch.unsqueeze(rows, 0), torch.unsqueeze(cols, 0)), 0)
        # attention = torch.sparse.FloatTensor(indices, values_)
        return attention


    def forward_anchor(self, features, ori_adj, anchor_nodes_idx, encoder, fusion_ratio):
        node_anchor_adj = self.knn_anchor_node(features, features[anchor_nodes_idx], self.weight_tensor1)
        node_anchor_adj = self.build_epsilon_neighbourhood(node_anchor_adj, self.epsilon, 0)

        if self.l_n > 0:
            indices_list = []

            n_node = features.shape[0]
            pre_idx = torch.range(0, n_node-1).long()

            embeddings_ = features
            adj_ = ori_adj
            for i in range(self.l_n): # [0,1,2]
                # self.score_layer = nn.Linear(osize, 1)
                y = F.sigmoid(self.score_layer(embeddings_[pre_idx,:], adj_).squeeze())

                score, idx = torch.topk(y, max(2, int(self.ks[i]*adj_.shape[0])))
                _, indices = torch.sort(idx)
                new_score = score[indices]
                new_idx = idx[indices]

                # global node index
                pre_idx=pre_idx.to(features.device)
                pre_idx = pre_idx[new_idx]
                
                indices_list.append(pre_idx)

                adj_ = extract_subgraph(adj_, new_idx)
                

                mask_score = torch.zeros(n_node).to(features.device)
                mask_score[pre_idx] = new_score
                embeddings_ = torch.mul(embeddings_, torch.unsqueeze(mask_score, -1) + torch.unsqueeze(1-mask_score, -1).detach())


            for j in reversed(range(self.l_n)):
                node_anchor_vec = encoder(embeddings_, node_anchor_adj, True, False)
                node_vec = encoder(embeddings_, ori_adj, False, False)
                node_vec = fusion_ratio * node_anchor_vec + (1 - fusion_ratio) * node_vec
                
                new_node_anchor_adj = self.knn_anchor_node(node_vec, node_vec[anchor_nodes_idx], self.weight_tensor2)
                new_node_anchor_adj = self.build_epsilon_neighbourhood(new_node_anchor_adj, self.epsilon, 0)

                # modify the node_anchor subgraph
                mask = torch.ones(n_node).to(features.device)
                mask[indices_list[j]] = self.anchor_adj_fusion_ratio
                node_anchor_adj = torch.mul(node_anchor_adj, torch.unsqueeze(mask, -1)) + torch.mul(new_node_anchor_adj, torch.unsqueeze(1-mask, -1).detach())
            
        return node_anchor_adj
