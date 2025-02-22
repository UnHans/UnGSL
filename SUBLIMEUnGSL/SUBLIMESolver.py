from copy import deepcopy
from opengsl.module.model.sublime import dgl_graph_to_torch_sparse_cuda,torch_sparse_to_dgl_graph, torch_sparse_to_dgl_graph_cuda,GCL, get_feat_mask, split_batch, dgl_graph_to_torch_sparse, GCN_SUB,UnGSL,SparseUnGSL
import torch
import time
from .solver import Solver
from opengsl.module.functional import normalize, symmetry
from opengsl.module.graphlearner import AttLearner, MLPLearner
from opengsl.module.metric import FGP
import copy
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import nni

def sparse_matrix_combine(adj1,adj2,weight1,weight2):
        indices1= adj1._indices()
        values1 = adj1._values()    
        values1 = weight1 * values1

        indices2= adj2._indices()
        values2 = adj2._values()
        values2 = weight2 * values2

        combined_indices = torch.cat([indices1, indices2], dim=1)
        combined_values = torch.cat([values1, values2])

        fuse_adj = torch.sparse_coo_tensor(combined_indices, combined_values,adj1.shape)
        fuse_adj = fuse_adj.coalesce()

        return torch_sparse_to_dgl_graph(fuse_adj)
class SUBLIMESolver(Solver):
    '''
    A solver to train, evaluate, test SUBLIME in a run.

    Parameters
    ----------
    conf : argparse.Namespace
        Config file.
    dataset : opengsl.data.Dataset
        The dataset.

    Attributes
    ----------
    method_name : str
        The name of the method.

    Examples
    --------
    >>> # load dataset
    >>> import opengsl.dataset
    >>> dataset = opengsl.data.Dataset('cora', feat_norm=True)
    >>> # load config file
    >>> import opengsl.config.load_conf
    >>> conf = opengsl.config.load_conf('sublime', 'cora')
    >>>
    >>> solver = SUBLIMESolver(conf, dataset)
    >>> # Conduct a experiment run.
    >>> acc, new_structure = solver.run_exp(split=0, debug=True)
    '''
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)
        self.method_name = "sublime"
        print("Solver Version : [{}]".format("sublime"))
        self.run_time=None
        self.dataset=dataset
        edge_index = self.adj.coalesce().indices().cpu()
        loop_edge_index = torch.stack([torch.arange(self.n_nodes), torch.arange(self.n_nodes)])
        edges = torch.cat([edge_index, loop_edge_index], dim=1)
        self.adj = torch.sparse.FloatTensor(edges, torch.ones(edges.shape[1]), [self.n_nodes, self.n_nodes]).to(self.device).coalesce()
        # self.allCLloss = torch.full((self.n_nodes,),0.0).to("cuda:0")

    def loss_gcl(self, model, graph_learner, features, anchor_adj,UnGSL):

        # view 1: anchor graph
        st = time.time()
        if self.conf.maskfeat_rate_anchor:
            
            mask_v1, _ = get_feat_mask(features, self.conf.maskfeat_rate_anchor) 
            features_v1 = features * (1 - torch.tensor(mask_v1,device = "cuda:0"))
        else:
            features_v1 = copy.deepcopy(features)

        z1, _ = model(features_v1, anchor_adj, 'anchor')

        # view 2: learned graph
        if self.conf.maskfeat_rate_learner:
            mask, _ = get_feat_mask(features, self.conf.maskfeat_rate_learner)
            features_v2 = features * (1 - torch.tensor(mask,device = "cuda:0"))
        else:
            features_v2 = copy.deepcopy(features)
        learned_adj = graph_learner(features)   # 这个learned adj是有自环的
        if not self.conf.sparse:
            learned_adj = symmetry(learned_adj)
            learned_adj = normalize(learned_adj, add_loop=False)

        """modify while training"""
        # st = time.time()
        learned_adj=UnGSL(learned_adj)
        # ed = time.time()
        # print(learned_adj.num_edges)
        z2, _ = model(features_v2, learned_adj, 'learner')

        # compute loss
        if self.conf.contrast_batch_size:#!!!!!!!!!!!!!!
            node_idxs = list(range(features.shape[0]))
            batches = split_batch(node_idxs, self.conf.contrast_batch_size)
            loss = 0
            for batch in batches:
                weight = len(batch) / features.shape[0]
                loss1 = model.calc_loss(z1[batch], z2[batch])
                loss += loss1 * weight
                # print(loss1)
                # self.allCLloss[torch.tensor(batch)] = batch_loss

        else:
            loss= model.calc_loss(z1, z2)
            # self.allCLloss=allCLloss
        return loss, learned_adj

    def train_gcn(self, adj, debug=False):
        model= GCN_SUB(nfeat=self.dim_feats, nhid=self.conf.hidden_dim_cls, nclass=self.num_targets,
                        n_layers=self.conf.n_layers_cls, dropout=self.conf.dropout_cls,
                        dropout_adj=self.conf.dropedge_cls, sparse=self.conf.sparse).to(self.device)
        optim = torch.optim.Adam(model.parameters(), lr=self.conf.lr_cls, weight_decay=self.conf.w_decay_cls)
        t = time.time()
        improve_1 = ''
        best_loss_val = 10
        best_acc_val = 0
        for epoch in range(self.conf.epochs_cls):
            improve_2 = ''
            t0 = time.time()
            model.train()
            optim.zero_grad()
            # forward and backward
            output= model(self.feats, adj)
            loss_train = self.loss_fn(output[self.train_mask], self.labels[self.train_mask])
            acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(), output[self.train_mask].detach().cpu().numpy())
            loss_train.backward()
            optim.step()
            # Evaluate
            loss_val, acc_val = self.evaluate(model, self.val_mask, adj)
            
            # save
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                best_loss_val = loss_val
                improve_2 = '*'
                if acc_val > self.result['valid']:
                    
                    self.total_time = time.time()-self.start_time
                    improve_1 = '*'
                    self.best_val_loss = loss_val
                    self.result['valid'] = acc_val
                    self.result['train'] = acc_train
                    self.weights = deepcopy(model.state_dict())
                    st=time.time()
                    current_adj = adj#dgl_graph_to_torch_sparse(adj).to_dense() if self.conf.sparse else adj
                    ed=time.time()
                    # print(ed-st,"1")
                    st=time.time()
                    self.adjs['final'] = current_adj.clone()
                    ed=time.time()
                    # print(ed-st,"2")
                    self.best_graph_test = deepcopy(adj)

            if debug:
                print("Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                    epoch+1, time.time() -t0, loss_train.item(), acc_train, loss_val, acc_val, improve_2))


        print('Time(s) {:.4f} | Loss(val):{:.4f} | Acc(val):{:.4f} | {}'.format(time.time()-t, best_loss_val, best_acc_val, improve_1))
        nni.report_intermediate_result(best_acc_val)

    def evaluate(self, model, test_mask, adj):
        '''
        Evaluation procedure of GRCN.

        Parameters
        ----------
        model : torch.nn.Module
            model.
        test_mask : torch.tensor
            A boolean tensor indicating whether the node is in the data set.
        adj : torch.tensor
            Adjacency matrix.

        Returns
        -------
        loss : float
            Evaluation loss.
        metric : float
            Evaluation metric.
        '''
        model.eval()
        with torch.no_grad():
            output = model(self.feats, adj)
            # if len(test_mask)==48603:
            #     print(self.run_time)
            #     softmax_func=torch.nn.Softmax(dim=1)
            #     prob_matrix=softmax_func(output)
            #     uncertrainy = -torch.sum(prob_matrix * torch.log(prob_matrix), dim=1)
            #     torch.save(uncertrainy,"SUBLIMEArxivEntropy"+str(self.run_time)+".pt")
            #     print("Arxiv entropy saved")
        logits = output[test_mask]
        labels = self.labels[test_mask]
        loss=self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy())

    def test(self):
        '''
        Test procedure of SUBLIME.

        Returns
        -------
        loss : float
            Evaluation loss.
        metric : float
            Evaluation metric.
        '''
        model = GCN_SUB(nfeat=self.dim_feats, nhid=self.conf.hidden_dim_cls, nclass=self.num_targets,
                        n_layers=self.conf.n_layers_cls, dropout=self.conf.dropout_cls,
                        dropout_adj=self.conf.dropedge_cls, sparse=self.conf.sparse).to(self.device)
        model.load_state_dict(self.weights)
        adj = self.best_graph_test
        return self.evaluate(model, self.test_mask, adj)

    def learn(self, debug=False,run_time=None):
        '''
        Learning process of SUBLIME.

        Parameters
        ----------
        debug : bool
            Whether to print statistics during training.

        Returns
        -------
        result : dict
            A dict containing train, valid and test metrics.
        graph : torch.tensor
            The learned structure.
        '''
        self.run_time=run_time
        anchor_adj = normalize(self.anchor_adj_raw, add_loop=False)
        if self.conf.sparse:
            anchor_adj_torch_sparse = copy.deepcopy(anchor_adj)
            anchor_adj = torch_sparse_to_dgl_graph(anchor_adj)
        for epoch in range(1, self.conf.epochs + 1):
            # st = time.time()
            # Contrastive Learning
            self.model.train()
            self.graph_learner.train()
            loss, Adj = self.loss_gcl(self.model, self.graph_learner, self.feats, anchor_adj,self.ungsl)
            # Adj是有自环且normalized
            self.optimizer_cl.zero_grad()
            self.optimizer_learner.zero_grad()
            self.optim_ungsl.zero_grad()
            loss.backward()
            self.optimizer_cl.step()
            self.optimizer_learner.step()
            self.optim_ungsl.step()

            # Structure Bootstrapping
            if (1 - self.conf.tau) and (self.conf.c == 0 or epoch % self.conf.c == 0):
                if self.conf.sparse:
                    learned_adj_torch_sparse = dgl_graph_to_torch_sparse(Adj)
                    """modify while training.............................."""
                    anchor_adj_torch_sparse = anchor_adj_torch_sparse * self.conf.tau \
                                              + learned_adj_torch_sparse * (1 - self.conf.tau)
                    anchor_adj = torch_sparse_to_dgl_graph(anchor_adj_torch_sparse)
                else:
                    anchor_adj = anchor_adj * self.conf.tau + Adj.detach() * (1 - self.conf.tau)
            if debug:
                #print("S")
                print("Epoch {:05d} | CL Loss {:.4f}".format(epoch, loss.item()))

            # Evaluate via Node Classification
            if epoch % self.conf.eval_freq == 0:

                self.model.eval()
                self.graph_learner.eval()

                f_adj = Adj
                if self.conf.sparse:
                    f_adj.edata['w'] = f_adj.edata['w'].detach()
                else:
                    f_adj = f_adj.detach()
                self.train_gcn(f_adj, debug)

            # if epoch==self.conf.epochs:#!!!!!!!!!!!!!!!!!!
            #     print(self.allCLloss.size())
            #     torch.save(self.allCLloss,"SUBLIMEArxivCLLoss"+str(self.run_time)+".pt")
            #     print("saved")
            #     print("Epoch {:05d} | CL Loss {:.4f}".format(epoch, loss.item()))
        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test = self.test()
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result, self.adjs

    def set_method(self,run_time=None):
        '''
        Function to set the model and necessary variables for each run, automatically called in function `set`.

        '''
        if self.conf.sparse:
            self.anchor_adj_raw = self.adj
        else:
            self.anchor_adj_raw = self.adj.to_dense()
        anchor_adj = normalize(self.anchor_adj_raw, add_loop=False)
        self.normalized_adj = anchor_adj
        if self.conf.type_learner == 'fgp':
            self.graph_learner = FGP(self.feats.shape[0])
            self.graph_learner.reset_parameters(self.feats.cpu(), self.conf.k, self.conf.sim_function, 6)
        elif self.conf.type_learner == 'mlp':
            self.graph_learner = MLPLearner(2, self.feats.shape[1], self.conf.k, 6, self.conf.sparse,
                                            self.conf.activation_learner)
        elif self.conf.type_learner == 'att':
            self.graph_learner = AttLearner(2, self.feats.shape[1], self.conf.k, 6, self.conf.sparse,
                                            self.conf.activation_learner)
        self.graph_learner = self.graph_learner.to(self.device)
        self.model = GCL(nlayers=self.conf.n_layers, in_dim=self.dim_feats, hidden_dim=self.conf.n_hidden,
                    emb_dim=self.conf.n_embed, proj_dim=self.conf.n_proj,
                    dropout=self.conf.dropout, dropout_adj=self.conf.dropedge_rate, sparse=self.conf.sparse,
                         conf=self.conf).to(self.device)
        self.ungsl=UnGSL(dataset=self.dataset,conf=self.conf).to(self.device)
        #self.ungsl=SparseUnGSL(dataset=self.dataset,conf=self.conf).to(self.device)
        self.optimizer_cl = torch.optim.Adam(self.model.parameters(), lr=self.conf.lr, weight_decay=self.conf.wd)
        self.optimizer_learner = torch.optim.Adam(self.graph_learner.parameters(), lr=self.conf.lr,
                                             weight_decay=self.conf.wd)
        self.optim_ungsl= torch.optim.Adam(self.ungsl.parameters(), lr=self.conf.training["ungsl_lr"])