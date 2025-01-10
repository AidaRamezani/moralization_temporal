import torch.nn as nn
import torch
from torch_geometric.nn.models import GAT, GCN
from torch_geometric.nn import Sequential
from torch_geometric.nn.norm.layer_norm import LayerNorm
from torch_geometric.nn.models.basic_gnn import BasicGNN
from typing import Optional, List
from torch import Tensor
  

class BasicLinkPredictor(nn.Module):
  def select_model(self, encoder_model_name,
                    in_channels,
                    hidden_channels,
                    num_layers,
                    num_heads,
                    out_channels: Optional[int] = None,
                    dropout: float = 0.0) -> BasicGNN:
        assert encoder_model_name in ['GAT','GAT2','GCN'] 
        if encoder_model_name == 'GAT': #other architectures to try
            gnn = GAT(in_channels= in_channels, hidden_channels=hidden_channels, num_layers=num_layers,
                      out_channels=out_channels, dropout=dropout, v2 = False, heads = num_heads,add_self_loops = False)
        elif encoder_model_name == 'GAT2': #other architectures to try
            gnn = GAT(in_channels= in_channels, hidden_channels=hidden_channels, num_layers=num_layers,
          out_channels=out_channels, dropout=dropout, v2 = True, heads = num_heads)
        elif encoder_model_name == 'GCN': 
            gnn = GCN(in_channels= in_channels, hidden_channels=hidden_channels, num_layers=num_layers,
                      out_channels=out_channels, dropout=dropout, heads = num_heads, add_self_loops = False,
                      jk = 'max', act = 'tanh'
                      )
        return gnn


class BasicPropertyPredictor(nn.Module):
    def __init__(self,
                 in_channels,
                 dropout: float = 0.0):
        super(BasicPropertyPredictor, self).__init__()
        
        self.linear2 = nn.Linear(in_channels, in_channels, bias = True)
        self.linear = nn.Linear(in_channels, 1, bias = False)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU(inplace = False)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.add_linear = False
    
    def forward(self, x: Tensor):
        node_feature = self.dropout(x)
        node_feature = self.linear2(node_feature)
        node_feature = self.relu(node_feature)
        node_feature = self.linear(node_feature)
        parameter = node_feature
        parameter = parameter.squeeze(1)
        return parameter
    


class ParameterPropertyPredictor(BasicLinkPredictor):
    def __init__(self, n,
                 in_channels,
                 hidden_channels,
                 out_channels: Optional[int] = None,
                 num_layers = 2,
                 num_heads = 5,
                 dropout: float = 0.0,
                 encoder_model_name = 'GCN',
                 reduce = 'mean',
                 add_linear = False):
        super(ParameterPropertyPredictor, self).__init__()
        assert reduce in ['mean','sum','add','mult', 'concat','forward','both'], 'reduce must be one of mean, sum, add, mult, concat,foward, linking, both'
        self.n = n
        self.encoder_model = self.select_model(encoder_model_name,
                                             in_channels,
                                             hidden_channels,
                                             num_layers,
                                             num_heads,
                                             out_channels,
                                             dropout)
        self.decoder_model = self.select_model(encoder_model_name,
                                              in_channels,
                                              hidden_channels,
                                              num_layers,
                                              num_heads,
                                              out_channels,
                                              dropout)
        self.reduce = reduce
        self.encoder_norm = LayerNorm(out_channels, mode = 'node')
        self.decoder_norm = LayerNorm(out_channels, mode = 'node')
        
        linear_dim = out_channels
        self.linear = nn.Linear(linear_dim, 1, bias = False)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU(inplace = False)
        self.emb_linar = nn.Linear(in_channels, out_channels, bias = False)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.add_linear = False
        if add_linear:
            self.add_linear = True
            self.forward_linear = nn.Linear(out_channels, out_channels, bias = False)
            self.backward_linear = nn.Linear(out_channels, out_channels, bias = False)

    
    def forward(self, x: Tensor,
                edge_index: Tensor,
                edge_weight: Optional[Tensor] = None,
                edge_attr: Optional[Tensor] = None,
                num_sampled_nodes_per_hop: Optional[List[int]] = None,
                num_sampled_edges_per_hop: Optional[List[int]] = None):

        assert x.shape[0] == self.n
        
        forward_embeddings = self.encoder_model.forward(x, edge_index,
                                              edge_weight = edge_weight,
                                              edge_attr = edge_attr,
                                              num_sampled_nodes_per_hop = num_sampled_nodes_per_hop,
                                              num_sampled_edges_per_hop = num_sampled_edges_per_hop)
       
        forward_embeddings = self.encoder_norm(forward_embeddings)

        backward_embeddings = self.decoder_model.forward(x, edge_index,
                                                              edge_weight = edge_weight,
                                                              edge_attr = edge_attr,
                                                              num_sampled_nodes_per_hop = num_sampled_nodes_per_hop,
                                                              num_sampled_edges_per_hop = num_sampled_edges_per_hop)
        backward_embeddings = self.decoder_norm(backward_embeddings)
        if self.add_linear:
            forward_embeddings = self.forward_linear(forward_embeddings)
            backward_embeddings = self.backward_linear(backward_embeddings)
        
        if self.reduce == 'mean':  #other architectues to try
            node_embeddings = (forward_embeddings + backward_embeddings) / 2
        elif self.reduce in ['sum', 'add']:
            node_embeddings = forward_embeddings + backward_embeddings
        elif self.reduce == 'mult':
            node_embeddings = forward_embeddings * backward_embeddings
        elif self.reduce == 'concat':
            node_embeddings = torch.cat((forward_embeddings, backward_embeddings), dim = 1)
        elif self.reduce == 'linking': 
            similarities = torch.matmul(forward_embeddings, backward_embeddings.T) 
            node_embeddings = torch.matmul(similarities, forward_embeddings) 
        elif self.reduce == 'forward': #GCN
            node_embeddings = forward_embeddings
            
        elif self.reduce == 'both': #Residual GCN
            emb_1 = self.dropout(x)
            node_feature_1 = self.emb_linar(emb_1)
            node_feature_1 = self.relu(node_feature_1)
            node_feature_2 = forward_embeddings
            node_embeddings = node_feature_1 + node_feature_2
            
        node_embeddings = self.batch_norm(node_embeddings)
        node_feature = self.linear(node_embeddings)
        parameter = node_feature
        parameter = parameter.squeeze(1)
        return parameter
    





# class PropertyPredictor(BasicLinkPredictor):
#     def __init__(self, n,
#                  in_channels,
#                  hidden_channels,
#                  out_channels: Optional[int] = None,
#                  num_layers = 2,
#                  num_heads = 5,
#                  dropout: float = 0.0,
#                  encoder_model_name = 'GAT',
#                  reduce = 'mean',
#                  add_linear = False):
#         super(PropertyPredictor, self).__init__()
#         assert reduce in ['mean','sum','add','mult','forward','both'], 'reduce must be one of mean, sum, add, mult,foward, both, it is {}'.format(reduce)
#         self.n = n
#         self.dropout = nn.Dropout(p=dropout)
#         self.encoder_model = self.select_model(encoder_model_name,
#                                              in_channels,
#                                              hidden_channels,
#                                              num_layers,
#                                              num_heads,
#                                              out_channels,
#                                              dropout)
#         self.decoder_model = self.select_model(encoder_model_name,
#                                               in_channels,
#                                               hidden_channels,
#                                               num_layers,
#                                               num_heads,
#                                               out_channels,
#                                               dropout)
#         self.reduce = reduce
#         self.encoder_norm = LayerNorm(out_channels, mode = 'node')
#         self.decoder_norm = LayerNorm(out_channels, mode = 'node')
#         linear_dim = out_channels 
#         self.linear = nn.Linear(linear_dim, 2, bias = False)
#         self.emb_linar = nn.Linear(in_channels, out_channels, bias = False)
#         self.relu = nn.ReLU(inplace = False)
#         self.tanh = nn.Tanh()
#         self.sigmoid = nn.Sigmoid()
#         self.add_linear = False
#         if add_linear:
#             self.add_linear = True
#             self.forward_linear = nn.Linear(out_channels, out_channels, bias = False)
#             self.backward_linear = nn.Linear(out_channels, out_channels, bias = False)

    
#     def forward(self, x: Tensor,
#                 edge_index: Tensor,
#                 edge_weight: Optional[Tensor] = None,
#                 edge_attr: Optional[Tensor] = None,
#                 num_sampled_nodes_per_hop: Optional[List[int]] = None,
#                 num_sampled_edges_per_hop: Optional[List[int]] = None):

#         assert x.shape[0] == self.n
        
#         forward_embeddings = self.encoder_model.forward(x, edge_index,
#                                               edge_weight = edge_weight,
#                                               edge_attr = edge_attr,
#                                               num_sampled_nodes_per_hop = num_sampled_nodes_per_hop,
#                                               num_sampled_edges_per_hop = num_sampled_edges_per_hop)
        
#         forward_embeddings = self.encoder_norm(forward_embeddings)

#         backward_embeddings = self.decoder_model.forward(x, edge_index,
#                                                               edge_weight = edge_weight,
#                                                               edge_attr = edge_attr,
#                                                               num_sampled_nodes_per_hop = num_sampled_nodes_per_hop,
#                                                               num_sampled_edges_per_hop = num_sampled_edges_per_hop)
#         backward_embeddings = self.decoder_norm(backward_embeddings)
#         if self.add_linear:
#             forward_embeddings = self.forward_linear(forward_embeddings)
#             backward_embeddings = self.backward_linear(backward_embeddings)
        
#         if self.reduce == 'mean':
#             node_embeddings = (forward_embeddings + backward_embeddings) / 2
#         elif self.reduce in ['sum', 'add']:
#             node_embeddings = forward_embeddings + backward_embeddings
#         elif self.reduce == 'mult':
#             node_embeddings = forward_embeddings * backward_embeddings
#         elif self.reduce == 'linking':
#             similarities = torch.matmul(forward_embeddings, backward_embeddings.T) #shape = (n, n)
#             node_embeddings = torch.matmul(similarities, forward_embeddings) #Should I use forward or backward?
#         elif self.reduce == 'forward':
#             node_embeddings = forward_embeddings
#         elif self.reduce == 'both':
#             emb_1 = self.dropout(x)
#             node_feature_1 = self.emb_linar(emb_1)
#             node_feature_1 = self.relu(node_feature_1)
#             node_feature_2 = forward_embeddings
#             node_embeddings = node_feature_1 + node_feature_2

#         node_feature = self.linear(node_embeddings)
        
#         assert node_feature.shape == (self.n, 2)
#         # node_feature = node_feature.squeeze(1)
#         return node_feature
    


# class BasicEncoderLinkPredictor(BasicLinkPredictor): #Simple encoder model
#     def __init__(self,
#                  in_channels,
#                  hidden_channels,
#                  dropout: float = 0.0):
#         super(BasicEncoderLinkPredictor, self).__init__()
       
#         self.linear1 = nn.Linear(in_channels, hidden_channels, bias = True)
#         self.relu = nn.ReLU(inplace = False)
#         self.leakyrelu = nn.LeakyReLU(inplace = False)
#         self.linear2 = nn.Linear(in_channels, hidden_channels, bias = True)
#         self.norm = nn.LayerNorm(in_channels)
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax(dim = 1)
#         self.dropout = nn.Dropout(p=dropout)
        
    
#     def forward(self, emb:Tensor, all_emb:Tensor):
#         '''
#         emb: Tensor of shape (batch_size, in_channels)
#         all_emb: Tensor of shape (n, in_channels)
#         '''
        
#         assert emb.shape[1] == all_emb.shape[1]
        
#         emb = self.dropout(emb)
#         emb = self.linear1(emb)
#         all_embs = self.linear2(all_emb)
#         emb = self.leakyrelu(emb)
#         all_embs = self.leakyrelu(all_embs)
#         # emb = self.leakyrelu(emb)
#         # emb = self.linear2(emb) #shape = (batch_size, in_channels)
#         # emb = self.norm(emb)
#         similarities = torch.matmul(emb, all_embs.T) #shape = (batch_size, n)
#         # similarities = self.softmax(similarities) #shape = (batch_size, n)
#         similarities = self.leakyrelu(similarities) #shape = (batch_size, n)
#         return similarities


# class SequentialLinkPredictor(BasicLinkPredictor):
#     def __init__(self, n,
#                  in_channels,
#                  hidden_channels,
#                  out_channels: Optional[int] = None,
#                  num_layers = 2,
#                  num_heads = 5,
#                  dropout: float = 0.0,
#                  encoder_model_name = 'GAT',
#                  device = 'cpu'):
#         super(SequentialLinkPredictor, self).__init__()
#         self.n = n
#         self.forward_dropout = nn.Dropout(p=dropout)
#         self.backward_dropout = nn.Dropout(p=dropout)

#         self.encoder_gnn_forward = self.select_model(encoder_model_name,
#                                              in_channels,
#                                              hidden_channels,
#                                              num_layers,
#                                              num_heads,
#                                              out_channels,
#                                              dropout)
#         self.forward_layernorm = LayerNorm(out_channels, mode= 'node')
#         self.backward_layernorm = LayerNorm(out_channels, mode= 'node')
#         self.forward_linear = nn.Linear(out_channels, out_channels)
#         self.backward_linear = nn.Linear(out_channels, out_channels)

        
#         self.encoder_gnn_backward = self.select_model(encoder_model_name,
#                                               in_channels,
#                                               hidden_channels,
#                                               num_layers,
#                                               num_heads,
#                                               out_channels,
#                                               dropout)
#         # self.decoder_model = Sequential('x, edge_index','edge_weight', [
#         #     nn.Dropout(p=dropout), 'x -> x',
#         #     (self.decoder_activation, 'x, edge_index, edge_weight -> x'),
#         #     LayerNorm(out_channels, mode= 'node'),
#         #     nn.Linear(out_channels, out_channels),
#         #     # nn.ReLU(inplace=True),
#         # ])
#         self.decoder_activation = nn.Softmax(dim = 1)
#     def forward(self, x: Tensor,
#                 edge_index: Tensor,
#                 edge_weight: Optional[Tensor] = None,
#                 edge_attr: Optional[Tensor] = None,
#                 num_sampled_nodes_per_hop: Optional[List[int]] = None,
#                 num_sampled_edges_per_hop: Optional[List[int]] = None):

#         assert x.shape[0] == self.n
#         new_x = self.forward_dropout(x)

#         forward_embeddings = self.encoder_gnn_forward.forward(new_x, edge_index,
#                                               edge_weight = edge_weight)
        
#         forward_embeddings = self.forward_layernorm(forward_embeddings)
#         forward_embeddings = self.forward_linear(forward_embeddings)    

#         backward_embeddings = self.encoder_gnn_backward.forward(new_x, edge_index, edge_weight = edge_weight)
#         backward_embeddings = self.backward_layernorm(backward_embeddings)
#         backward_embeddings = self.backward_linear(backward_embeddings)


#         new_A = torch.matmul(forward_embeddings, backward_embeddings.T)
#         row_sum = torch.sum(new_A, dim = 1)
#         diag_norm = torch.norm(new_A.diagonal())
#         # new_A = new_A.fill_diagonal_(0.0) #TODO-check
#         # new_A = self.decoder_activation(new_A)

#         return new_A, diag_norm, row_sum
    

# class EncoderLinkPredictor(BasicLinkPredictor):
#     def __init__(self, n,
#                  in_channels,
#                  hidden_channels,
#                  out_channels: Optional[int] = None,
#                  num_layers = 2,
#                  num_heads = 5,
#                  dropout: float = 0.0,
#                  encoder_model_name = 'GAT',
#                  activate = 'relu',
#                  device = 'cpu'):
#         super(EncoderLinkPredictor, self).__init__()
#         self.n = n
#         self.encoder_forward_model = self.select_model(encoder_model_name,
#                                              in_channels,
#                                              hidden_channels,
#                                              num_layers,
#                                              num_heads,
#                                              out_channels,
#                                              dropout)
#         self.encoder_norm = LayerNorm(out_channels, mode= 'node')
#         self.encoder_backward_model = self.select_model(encoder_model_name,
#                                               in_channels,
#                                               hidden_channels,
#                                               num_layers,
#                                               num_heads,
#                                               out_channels,
#                                               dropout)
#         self.decoder_norm = LayerNorm(out_channels, mode= 'node')
#         self.linear = nn.Linear(n, n)
#         self.relu = nn.ReLU(inplace = False)
#         self.edge_norm = nn.LayerNorm((n, n), elementwise_affine=False, device = device)
#         self.sigmoid = nn.Sigmoid()
#         if activate == 'relu':
#             self.decoder_activation = nn.ReLU(inplace = False)
#         elif activate == 'softmax':
#             self.decoder_activation = nn.Softmax(dim = 1)
#         else:
#             raise ValueError('activation function not supported')
#     def forward(self, x: Tensor,
#                 edge_index: Tensor,
#                 edge_weight: Optional[Tensor] = None,
#                 edge_attr: Optional[Tensor] = None,
#                 num_sampled_nodes_per_hop: Optional[List[int]] = None,
#                 num_sampled_edges_per_hop: Optional[List[int]] = None):

#         assert x.shape[0] == self.n

#         forward_embeddings = self.encoder_forward_model.forward(x, edge_index,
#                                               edge_weight = edge_weight,
#                                               edge_attr = edge_attr,
#                                               num_sampled_nodes_per_hop = num_sampled_nodes_per_hop,
#                                               num_sampled_edges_per_hop = num_sampled_edges_per_hop)
#         #TODO do I need this?
#         forward_embeddings = self.encoder_norm(forward_embeddings)
        
#         backward_embeddings = self.encoder_backward_model.forward(x, edge_index,
#                                                               edge_weight = edge_weight,
#                                                               edge_attr = edge_attr,
#                                                               num_sampled_nodes_per_hop = num_sampled_nodes_per_hop,
#                                                               num_sampled_edges_per_hop = num_sampled_edges_per_hop)
#         backward_embeddings = self.decoder_norm(backward_embeddings)
        
#         new_A = torch.matmul(forward_embeddings, backward_embeddings.T)
#         new_A = self.linear(new_A)
#         row_sum = torch.sum(new_A, dim = 1)
#         diag_norm = torch.norm(new_A.diagonal())
#         return new_A, diag_norm, row_sum

