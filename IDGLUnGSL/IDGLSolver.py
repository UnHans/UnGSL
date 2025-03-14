from copy import deepcopy
from opengsl.module.model.idgl import IDGL, sample_anchors, diff, compute_anchor_adj,UnGSL
import torch
import torch.nn.functional as F
import time
import numpy as np
from .solver import Solver
from opengsl.module.functional import normalize
from opengsl.module.regularizer import connectivity_regularizer, smoothness_regularizer, norm_regularizer
import nni

class IDGLSolver(Solver):
    '''
    A solver to train, evaluate, test IDGL in a run.

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
    >>> conf = opengsl.config.load_conf('idgl', 'cora')
    >>>
    >>> solver = IDGLSolver(conf, dataset)
    >>> # Conduct a experiment run.
    >>> acc, new_structure = solver.run_exp(split=0, debug=True)
    '''
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)
        self.method_name = "idgl"
        print("Solver Version : [{}]".format("idgl"))
        self.run_epoch = self._scalable_run_whole_epoch if self.conf.model['scalable_run'] else self._run_whole_epoch
        self.normalized_adj = normalize(self.adj,add_loop=conf.dataset["add_loop"])
        if not self.conf.model['scalable_run']:
            self.normalized_adj = self.normalized_adj.to_dense()
        edge_index = self.adj.coalesce().indices().cpu()
        loop_edge_index = torch.stack([torch.arange(self.n_nodes), torch.arange(self.n_nodes)])
        edges = torch.cat([edge_index, loop_edge_index], dim=1)
        self.adj = torch.sparse.FloatTensor(edges, torch.ones(edges.shape[1]), [self.n_nodes, self.n_nodes]).to(self.device).coalesce()
        
    def _run_whole_epoch(self, mode='train', debug=False,run_time=None):
        # prepare
        training = mode == 'train'
        if mode == 'train':
            idx = self.train_mask
        elif mode == 'valid':
            idx = self.val_mask
        else:
            idx = self.test_mask
            #print(torch.sum( (self.adj).to_dense() [idx],dim=1))
        self.model.train(training)
        network = self.model

        # The first iter
        features = F.dropout(self.feats, self.conf.gsl['feat_adj_dropout'], training=training)
        init_node_vec = features
        init_adj = self.normalized_adj
        cur_raw_adj, cur_adj = network.learn_graph(network.graph_learner, init_node_vec, self.conf.gsl['graph_skip_conn'], graph_include_self=self.conf.gsl['graph_include_self'], init_adj=init_adj)
        # cur_raw_adj是根据输入Z直接产生的adj, cur_adj是前者归一化并和原始adj加权求和的结果
        """i move this part from learn_graph"""
        cur_adj=self.UnGSL(cur_adj)
        if self.conf.gsl['graph_skip_conn'] in (0, None):
            if self.conf.gsl['graph_include_self']:
                cur_adj = cur_adj + torch.eye(cur_adj.size(0))
        else:
            cur_adj = self.conf.gsl['graph_skip_conn'] * init_adj + (1 - self.conf.gsl['graph_skip_conn']) * cur_adj
        """                            """
        cur_raw_adj = F.dropout(cur_raw_adj, self.conf.gsl['feat_adj_dropout'], training=training)
        cur_adj = F.dropout(cur_adj, self.conf.gsl['feat_adj_dropout'], training=training)
        if self.conf.model['type'] == 'gcn':
            node_vec, output = network.encoder(init_node_vec, cur_adj)
        else:
            node_vec, output = network.encoder([init_node_vec, cur_adj, False])
        score = self.metric(self.labels[idx].cpu().numpy(), output[idx].detach().cpu().numpy())
        loss1 = self.loss_fn(output[idx], self.labels[idx])
        loss1 += self.get_graph_loss(cur_raw_adj, init_node_vec)
        first_raw_adj, first_adj = cur_raw_adj, cur_adj

        # the following iters
        if training:
            eps_adj = float(self.conf.gsl['eps_adj'])
        else:
            eps_adj = float(self.conf.gsl['test_eps_adj'])
        pre_raw_adj = cur_raw_adj
        pre_adj = cur_adj
        loss = 0
        iter_ = 0
        while (iter_ == 0 or diff(cur_raw_adj, pre_raw_adj, first_raw_adj).item() > eps_adj) and iter_ < self.conf.training['max_iter']:
            iter_ += 1
            pre_adj = cur_adj
            pre_raw_adj = cur_raw_adj
            cur_raw_adj, cur_adj = network.learn_graph(network.graph_learner2, node_vec, self.conf.gsl['graph_skip_conn'], graph_include_self=self.conf.gsl['graph_include_self'], init_adj=init_adj)
            
            """i move this part from learn_graph"""
            cur_adj=self.UnGSL(cur_adj)
            if self.conf.gsl['graph_skip_conn'] in (0, None):
                if self.conf.gsl['graph_include_self']:
                    cur_adj = cur_adj + torch.eye(cur_adj.size(0))
            else:
                cur_adj = self.conf.gsl['graph_skip_conn'] * init_adj + (1 - self.conf.gsl['graph_skip_conn']) * cur_adj
            """                                """
            update_adj_ratio = self.conf.gsl['update_adj_ratio']
            cur_adj = update_adj_ratio * cur_adj + (1 - update_adj_ratio) * first_adj   # 这里似乎和论文中有些出入？？
            if self.conf.model['type'] == 'gcn':
                node_vec, output = network.encoder(init_node_vec, cur_adj, self.conf.gsl['gl_dropout'])
            else:
                node_vec, output = network.encoder([init_node_vec, cur_adj, False])
            score = self.metric(self.labels[idx].cpu().numpy(), output[idx].detach().cpu().numpy())
            # if mode=="test" and len(idx)==1000:#
            #     softmax_func=torch.nn.Softmax(dim=1)
            #     prob_matrix=softmax_func(output)
            #     uncertrainy = -torch.sum(prob_matrix * torch.log(prob_matrix), dim=1)
            #     torch.save(uncertrainy,"IDGLCiteseerEntropy"+str(run_time)+".pt")
            #     print("entropy"+str(run_time)+"saved")
            loss += self.loss_fn(output[idx], self.labels[idx])
            loss += self.get_graph_loss(cur_raw_adj, init_node_vec)

        if iter_ > 0:
            loss = loss / iter_ + loss1
        else:
            loss = loss1

        if training:
            self.optimizer.zero_grad()
            self.optimizer_ungsl.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimizer_ungsl.step()
        return loss, score, first_raw_adj, cur_raw_adj, cur_adj

    def _scalable_run_whole_epoch(self, mode='train', debug=False,run_time=None,init_node_vec=None,init_anchor_vec=None,sampled_node_idx = None):
        # prepare
        training = mode == 'train'
        if mode == 'train':
            idx = self.train_mask
        elif mode == 'valid':
            idx = self.val_mask
        else:
            idx = self.test_mask
        self.model.train(training)
        network = self.model

        # init
        init_adj = self.normalized_adj
        features = F.dropout(self.feats, self.conf.gsl['feat_adj_dropout'], training=training)
        # init_node_vec = features
        init_node_vec=init_node_vec
        init_anchor_vec = init_anchor_vec
        sampled_node_idx = sampled_node_idx
        # init_anchor_vec, sampled_node_idx = sample_anchors(init_node_vec, self.conf.model['num_anchors'])
        # the first iter
        # Compute n x s node-anchor relationship matrix
        cur_node_anchor_adj = network.learn_graph(network.graph_learner, init_node_vec, anchor_features=init_anchor_vec)
        """modify learned graph structure"""
        cur_node_anchor_adj=self.UnGSL(cur_node_anchor_adj)

        """                              """
        # Compute s x s anchor graph
        cur_anchor_adj = compute_anchor_adj(cur_node_anchor_adj)
        cur_node_anchor_adj = F.dropout(cur_node_anchor_adj, self.conf.gsl['feat_adj_dropout'], training=training)
        cur_anchor_adj = F.dropout(cur_anchor_adj, self.conf.gsl['feat_adj_dropout'], training=training)
        first_init_agg_vec, init_agg_vec, node_vec, output = network.encoder(init_node_vec, init_adj, cur_node_anchor_adj, self.conf.gsl['graph_skip_conn'])
        anchor_vec = node_vec[sampled_node_idx]
        first_node_anchor_adj, first_anchor_adj = cur_node_anchor_adj, cur_anchor_adj
        score = self.metric(self.labels[idx].cpu().numpy(), output[idx].detach().cpu().numpy())
        loss1 = self.loss_fn(output[idx], self.labels[idx])
        loss1 += self.get_graph_loss(cur_anchor_adj, init_anchor_vec)
        
        # the following iters
        if training:
            eps_adj = float(self.conf.gsl['eps_adj'])
        else:
            eps_adj = float(self.conf.gsl['test_eps_adj'])

        pre_node_anchor_adj = cur_node_anchor_adj
        loss = 0
        iter_ = 0
        while (iter_ == 0 or diff(cur_node_anchor_adj, pre_node_anchor_adj, cur_node_anchor_adj).item() > eps_adj) and iter_ < self.conf.training['max_iter']:
            iter_ += 1
            pre_node_anchor_adj = cur_node_anchor_adj
            # Compute n x s node-anchor relationship matrix
            cur_node_anchor_adj = network.learn_graph(network.graph_learner2, node_vec, anchor_features=anchor_vec)
            """modify learned graph structure"""
            cur_node_anchor_adj=self.UnGSL(cur_node_anchor_adj)
            """                              """
            
            # Compute s x s anchor graph
            cur_anchor_adj = compute_anchor_adj(cur_node_anchor_adj)
            update_adj_ratio = self.conf.gsl['update_adj_ratio']
            _, _, node_vec, output = network.encoder(init_node_vec, init_adj, cur_node_anchor_adj, self.conf.gsl['graph_skip_conn'],
                                           first=False, first_init_agg_vec=first_init_agg_vec, init_agg_vec=init_agg_vec, update_adj_ratio=update_adj_ratio,
                                           dropout=self.conf.gsl['gl_dropout'], first_node_anchor_adj=first_node_anchor_adj)
            anchor_vec = node_vec[sampled_node_idx]
            score = self.metric(self.labels[idx].cpu().numpy(), output[idx].detach().cpu().numpy())
            loss += self.loss_fn(output[idx], self.labels[idx])
            loss += self.get_graph_loss(cur_anchor_adj, init_anchor_vec)
 
        # if mode=="test" and len(idx)==48603:#""
        #     softmax_func=torch.nn.Softmax(dim=1)
        #     prob_matrix=softmax_func(output)
        #     uncertrainy = -torch.sum(prob_matrix * torch.log(prob_matrix), dim=1)
        #     torch.save(uncertrainy,"IDGLArxivEntropy"+str(self.run_time)+".pt")
        #     print("entropy"+str(self.run_time)+"saved")
        if iter_ > 0:
            loss = loss / iter_ + loss1
        else:
            loss = loss1
        if training:
            self.optimizer.zero_grad()
            self.optimizer_ungsl.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimizer_ungsl.step()

        return loss, score, cur_anchor_adj, 0, 0

    def get_graph_loss(self, out_adj, features):
        # Graph regularization
        graph_loss = 0
        # L = torch.diagflat(torch.sum(out_adj, -1)) - out_adj
        # graph_loss += self.conf.training['smoothness_ratio'] * torch.trace(torch.mm(features.transpose(-1, -2), torch.mm(L, features))) / int(np.prod(out_adj.shape))
        # ones_vec = torch.ones(out_adj.size(-1)).to(self.device)
        # graph_loss += -self.conf.training['degree_ratio'] * torch.mm(ones_vec.unsqueeze(0), torch.log(torch.mm(out_adj, ones_vec.unsqueeze(-1)) + 1e-12)).squeeze() / out_adj.shape[-1]
        # graph_loss += self.conf.training['sparsity_ratio'] * torch.sum(torch.pow(out_adj, 2)) / int(np.prod(out_adj.shape))
        graph_loss += self.conf.training['smoothness_ratio'] * smoothness_regularizer(features, out_adj) / (out_adj.shape[0]*out_adj.shape[1])
        graph_loss += self.conf.training['degree_ratio'] * connectivity_regularizer(out_adj) / out_adj.shape[-1]
        graph_loss += self.conf.training['sparsity_ratio'] * norm_regularizer(out_adj, 'fro') / (out_adj.shape[0]*out_adj.shape[1])
        return graph_loss

    def learn(self, debug=False,run_time=None):
        '''
        Learning process of IDGL.

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
        wait = 0
        self.run_time=run_time
        features = F.dropout(self.feats, self.conf.gsl['feat_adj_dropout'], training=True)
        init_node_vec = features
        init_anchor_vec, sampled_node_idx = sample_anchors(init_node_vec, self.conf.model['num_anchors'])
        for epoch in range(self.conf.training['max_epochs']):
            t = time.time()
            improve = ''
            #print(epoch)
            # training phase
            loss_train, acc_train, _, _, _ = self.run_epoch(mode='train', debug=debug,init_node_vec=init_node_vec,init_anchor_vec=init_anchor_vec,sampled_node_idx=sampled_node_idx)

            # validation phase
            with torch.no_grad():
                loss_val, acc_val, _, _, _ = self.run_epoch(mode='valid', debug=debug,init_node_vec=init_node_vec,init_anchor_vec=init_anchor_vec,sampled_node_idx=sampled_node_idx)
                nni.report_intermediate_result(acc_val)
            if loss_val < self.best_val_loss:
                wait = 0
                self.total_time = time.time()-self.start_time
                self.best_val_loss = loss_val
                self.weights = deepcopy(self.model.state_dict())
                self.result['train'] = acc_train
                self.result['valid'] = acc_val
                improve = '*'
            else:
                wait += 1
                if wait == self.conf.training['patience']:
                    print('Early stop!')
                    break

            # print
            if debug:
                #print(debug)
                print("Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t, loss_train.item(), acc_train, loss_val, acc_val, improve))

        # test
        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        self.model.load_state_dict(self.weights)
        with torch.no_grad():
            loss_test, acc_test, first_adj, cur_adj, final_adj = self.run_epoch(mode='test', debug=debug,init_node_vec=init_node_vec,init_anchor_vec=init_anchor_vec,sampled_node_idx=sampled_node_idx)
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        self.result['test']=acc_test
        self.adjs['first_adj'] = first_adj
        self.adjs['cur_adj'] = cur_adj
        #print(torch.sum( (cur_adj.to_dense()>0)))
        self.adjs['final'] = final_adj
        torch.cuda.empty_cache()
        #print(final_adj)
        return self.result, self.adjs

    def set_method(self,run_time=None):
        '''
        Function to set the model and necessary variables for each run, automatically called in function `set`.

        '''
        self.run_time=run_time
        self.model = IDGL(self.conf, self.dim_feats, self.num_targets).to(self.device)
        self.UnGSL=UnGSL(conf=self.conf).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'], weight_decay=self.conf.training['weight_decay'])
        self.optimizer_ungsl=torch.optim.Adam(self.UnGSL.parameters(),lr=self.conf.training["ungsl_lr"])
        self.adjs['first_adj'] = None
        self.adjs['cur_adj'] = None
