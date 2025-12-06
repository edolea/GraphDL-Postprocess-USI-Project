from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Literal
from tsl.nn.layers import GatedGraphNetwork, NodeEmbedding, BatchNorm
from tsl.nn.models import GraphWaveNetModel
from spatiotemporal_postprocessing.nn.probabilistic_layers import dist_to_layer
from spatiotemporal_postprocessing.nn.prototypes import TCNLayer
from torch_geometric.nn import GATv2Conv


class LayeredGraphRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout_p = 0.1, mode: Literal['forwards', 'backwards'] = 'forwards', **kwargs) -> None:
        super().__init__(**kwargs)
        layers_ = []
        
        self.input_encoder = nn.Linear(input_size, hidden_size)
        
        for _ in range(n_layers):
            layers_.append(GatedGraphNetwork(input_size=hidden_size*2,
                                             output_size=hidden_size))

        
        self.mp_layers = torch.nn.ModuleList(layers_)
        self.state_size = hidden_size * n_layers
        self.n_layers = n_layers
        self.mode = mode
        self.dropout = nn.Dropout(p=dropout_p)
            
    def iterate_layers(self, state, x, edge_index):
        output = []
        state_ = rearrange(state, "b n ... (h l) -> l b n ... h", l=self.n_layers)
        for l, layer in enumerate(self.mp_layers):
            state_in_ = state_[l]
            
            input_ = torch.concatenate([state_in_, x], dim=-1) # recurrency 
            input_ = self.dropout(input_)
            
            x = layer(input_, edge_index) # potential to do x += here
            if isinstance(x, tuple):
                x = x[0] # if cell is a GAT, it returns the alphas
            output.append(x)
        
        return torch.cat(output, dim=-1)   
        
        
    def forward(self, x, edge_index):
        batch_size, win_size, num_nodes, num_feats = x.size()
        state = torch.zeros(batch_size, num_nodes, self.state_size, device=x.device)

        states = []
        # iterate forwards or backwards in time
        t0 = 0 if self.mode == 'forwards' else win_size - 1
        tn = win_size if self.mode == 'forwards' else -1 
        step = 1 if self.mode == 'forwards' else -1
        
        for t in range(t0, tn, step):
            x_ = self.input_encoder(x[:,t])

            # iterate over the depth
            state_out = self.iterate_layers(state=state, x=x_, edge_index=edge_index)
            
            state = state_out + state # skip connection in time
            
            if self.mode == 'forwards':
                states.append(state)
            else: 
                states.insert(0, state)

        
        return torch.stack(states, dim=1)
            
            
class BiDirectionalSTGNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_stations, output_dist: str, n_layers=1, dropout_p = 0.1, **kwargs) -> None:
        super().__init__()
        
        self.encoder = nn.Linear(input_size, hidden_size)
        self.station_embeddings = NodeEmbedding(n_stations, hidden_size)
        self.forward_model = LayeredGraphRNN(input_size=hidden_size, hidden_size=hidden_size, n_layers=n_layers, mode='forwards', dropout_p=dropout_p)
        self.backward_model = LayeredGraphRNN(input_size=hidden_size, hidden_size=hidden_size, n_layers=n_layers, mode='backwards', dropout_p=dropout_p)
        
        self.output_distr = dist_to_layer[output_dist](input_size=hidden_size)
            
        self.skip_conn = nn.Linear(input_size, 2*hidden_size*n_layers)
        
        self.readout = nn.Sequential(
            nn.Linear(2*hidden_size*n_layers, hidden_size),
            BatchNorm(in_channels=hidden_size, track_running_stats=False),
            nn.SiLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_size, hidden_size)
        )
        
        
    def forward(self, x, edge_index):
        x0 = x
        x = self.encoder(x)
        x = x + self.station_embeddings()
        states_forwards = self.forward_model(x, edge_index)  
        states_backwards = self.backward_model(x, edge_index)
        
        states = torch.concatenate([states_forwards, states_backwards], dim=-1)
        states = states + self.skip_conn(x0) # skip conn 
        
        output = self.readout(states)
        
        return self.output_distr(output)
    
    

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_dist: str, dropout_p, activation: str = "relu", **kwargs):
        super().__init__()
        
        activation_map = {
            "relu": nn.ReLU,
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU
        }
        layers = []
        self.input_size = input_size
        for hs in hidden_sizes:
            layers.append(nn.Linear(input_size, hs))
            layers.append(activation_map[activation]())
            layers.append(nn.Dropout(p=dropout_p))
            
            input_size = hs 
        self.layers = nn.Sequential(*layers)
        
        self.output_distr = dist_to_layer[output_dist](input_size=hidden_sizes[-1])
        
        self.skip_conn = nn.Linear(self.input_size, hidden_sizes[-1])
        
    def forward(self, x, **kwargs):
        # ignore edge index
        x_skip = self.skip_conn(x)
        
        x = self.layers(x)
        x = x + x_skip
        
        return self.output_distr(x)
    
    
class WaveNet(nn.Module):

    def __init__(self, input_size, time_steps, hidden_size, n_stations, output_dist, ff_size=256, n_layers=6, temporal_kernel_size=3, spatial_kernel_size=2, **kwargs):
        super().__init__()

        self.wavenet = GraphWaveNetModel(input_size=input_size,
                          output_size=hidden_size,
                          horizon=time_steps,
                          hidden_size=hidden_size,
                          ff_size=ff_size,
                          n_layers=n_layers,
                          temporal_kernel_size=temporal_kernel_size, spatial_kernel_size=spatial_kernel_size,
                          dilation=2, dilation_mod=3, n_nodes=n_stations)
        self.output_distr = dist_to_layer[output_dist](input_size=hidden_size)

    def forward(self, x, edge_index):
        output = self.wavenet(x, edge_index)

        return self.output_distr(output)

# ###########################
# ###### OUR VERSIONS #######
# ###########################

# """
# Model0: baseline
# """

# class Model0TCN(nn.Module):
#     """
#     TCN-only baseline model (Model0) for statistical postprocessing.

#     Architecture:
#     - same as TCN_GNN by MeteoSwiss, but with no graph information
#     (NOTE: this is a downgrade, the purpose is only to check if the graph realtions actually improve our modeling)

#     Args:
#         input_size: Number of input features per station
#         hidden_channels: Hidden dimension size
#         n_stations: Number of weather stations
#         output_dist: Distribution type (e.g., 'LogNormal')
#         num_layers: Number of TCN layers (default: 4)
#         kernel_size: Convolution kernel size (default: 3)
#         dropout_p: Dropout probability (default: 0.2)
#     """

#     def __init__(self, input_size, hidden_channels, n_stations, output_dist: str, num_layers, kernel_size, dropout_p, **kwargs):
#         super().__init__()

#         self.encoder = nn.Linear(input_size, hidden_channels)

#         self.rearrange_for_tcn = Rearrange('b t n c -> (b n) c t')                # [batch, time, stations, features] to [batch*stations, features, time]
#         self.rearrange_from_tcn = Rearrange('(b n) c t -> b t n c', n=n_stations) # [batch*stations, channels, time] back to [batch, time, stations, channels]

#         tcn_layers = []
#         for l in range(num_layers):
#             dilation = 2**l
#             in_channels = hidden_channels

#             tcn_layers.append(
#                 TCNLayer(
#                     in_channels=in_channels,
#                     out_channels=hidden_channels,
#                     kernel_size=kernel_size,
#                     dilation=dilation,
#                     dropout_p=dropout_p,
#                     causal_conv=False  # always bidirectional
#                 )
#             )

#         self.tcn_layers = nn.ModuleList(tcn_layers)

#         self.output_distr = dist_to_layer[output_dist](input_size=hidden_channels)

#     def forward(self, x, edge_index=None): # Graph edge indices (accepted but ignored for compatibility)
        
#         x = self.encoder(x)                       # [batch, time, stations, features] -> [batch, time, stations, hidden]
#         x = self.rearrange_for_tcn(x)             # [batch, time, stations, hidden] -> [batch*stations, hidden, time]

#         skips = []
#         for tcn_layer in self.tcn_layers:
#             x, skip = tcn_layer(x) # TCNLayer returns (residual_output, skip_output)
#             skips.append(skip)

#         skips_stack = torch.stack(skips, dim=-1)  # [batch*stations, hidden, time, num_layers]
#         skips_sum = skips_stack.sum(dim=-1)       # [batch*stations, hidden, time]
#         output = skips_sum + x                    # [batch*stations, hidden, time]
#         output = self.rearrange_from_tcn(output)  # [batch*stations, hidden, time] -> [batch, time, stations, hidden]

#         return self.output_distr(output)
   

# """
# STGNN1
# """

# class GraphAttentionLayer(nn.Module):
#     """
#     (Replaces the baseline's GatedGraphNetwork from MeteoSwiss)

#     Graph Attention Layer for spatiotemporal message passing.
#     Allows nodes to dynamically weight the importance of their neighbors.
#     """
#     def __init__(self, in_features, out_features, num_heads=4, dropout_p=0.1, concat=True):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.num_heads = num_heads
#         self.concat = concat
        
#         self.W = nn.Linear(in_features, out_features * num_heads, bias=False)
#         self.a_src = nn.Parameter(torch.Tensor(num_heads, out_features))
#         self.a_dst = nn.Parameter(torch.Tensor(num_heads, out_features))
        
#         self.leaky_relu = nn.LeakyReLU(0.2)
#         self.dropout = nn.Dropout(dropout_p)
        
#         self.reset_parameters()
        
#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.W.weight)
#         nn.init.xavier_uniform_(self.a_src.unsqueeze(1))
#         nn.init.xavier_uniform_(self.a_dst.unsqueeze(1))
        
#     def forward(self, x, edge_index):
#         batch_size, num_nodes, _ = x.size()
        
#         # Transform features
#         h = self.W(x)
#         h = h.view(batch_size, num_nodes, self.num_heads, self.out_features)
#         h = h.permute(0, 2, 1, 3)
        
#         # Compute attention scores
#         src_idx, dst_idx = edge_index[0], edge_index[1]
#         alpha_src = (h[:, :, src_idx] * self.a_src.unsqueeze(0).unsqueeze(2)).sum(dim=-1)
#         alpha_dst = (h[:, :, dst_idx] * self.a_dst.unsqueeze(0).unsqueeze(2)).sum(dim=-1)
#         alpha = self.leaky_relu(alpha_src + alpha_dst)
        
#         # Softmax normalization per node
#         alpha_max = torch.zeros(batch_size, self.num_heads, num_nodes, device=x.device)
#         alpha_max.scatter_reduce_(2, dst_idx.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1), 
#                                    alpha, reduce='amax', include_self=False)
#         alpha_max = alpha_max[:, :, dst_idx]
#         alpha = torch.exp(alpha - alpha_max)
        
#         alpha_sum = torch.zeros(batch_size, self.num_heads, num_nodes, device=x.device)
#         alpha_sum.scatter_add_(2, dst_idx.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1), alpha)
#         alpha_sum = alpha_sum[:, :, dst_idx] + 1e-10
#         alpha = alpha / alpha_sum
#         alpha = self.dropout(alpha)
        
#         # Aggregate with attention weights
#         h_edges = h[:, :, src_idx] * alpha.unsqueeze(-1)
#         out = torch.zeros(batch_size, self.num_heads, num_nodes, self.out_features, device=x.device)
#         out.scatter_add_(2, dst_idx.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(
#             batch_size, self.num_heads, -1, self.out_features), h_edges)
        
#         if self.concat:
#             out = out.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, -1)
#         else:
#             out = out.mean(dim=1)
            
#         return out

# class EnhancedLayeredGraphRNN(nn.Module):
#     """
#     (Replaces the baseline's LayeredGraphRNN from MeteoSwiss)

#     Enhanced Graph RNN with attention mechanisms and multi-scale temporal aggregation.
#     """
#     def __init__(self, input_size, hidden_size, n_layers=2, dropout_p=0.1, 
#                  mode: Literal['forwards', 'backwards'] = 'forwards', num_heads=4, **kwargs):
#         super().__init__(**kwargs)
        
#         self.hidden_size = hidden_size
#         self.n_layers = n_layers
#         self.mode = mode
        
#         # Input encoding with 2-layer MLP
#         self.input_encoder = nn.Sequential(
#             nn.Linear(input_size, hidden_size),
#             nn.LayerNorm(hidden_size),
#             nn.SiLU(),
#             nn.Linear(hidden_size, hidden_size)
#         )
        
#         # Graph Attention Layers
#         self.gat_layers = nn.ModuleList([
#             GraphAttentionLayer(
#                 in_features=hidden_size * 2,
#                 out_features=hidden_size,
#                 num_heads=num_heads, # TODO to remove if testing on single head attention 
#                 dropout_p=dropout_p,
#                 concat=True
#             ) for _ in range(n_layers)
#         ])

#         # Projection layers for multi-head outputs
#         self.head_projection = nn.ModuleList([
#             nn.Linear(hidden_size * num_heads, hidden_size) 
#             for _ in range(n_layers)
#         ])
        
#         # Layer normalization
#         self.layer_norms = nn.ModuleList([
#             nn.LayerNorm(hidden_size) for _ in range(n_layers)
#         ])
        
#         self.dropout = nn.Dropout(p=dropout_p)
#         self.state_size = hidden_size * n_layers
        
#     def iterate_layers(self, state, x, edge_index):
#         output = []
#         state_ = rearrange(state, "b n ... (h l) -> l b n ... h", l=self.n_layers)
        
#         for l, (gat_layer, proj_layer, layer_norm) in enumerate(
#             zip(self.gat_layers, self.head_projection, self.layer_norms)
#         ):
#             state_in_ = state_[l]
            
#             # Concatenate hidden state with input
#             input_ = torch.concatenate([state_in_, x], dim=-1)
            
#             # Apply graph attention
#             h = gat_layer(input_, edge_index)
#             h = proj_layer(h)
#             h = self.dropout(h)
            
#             # Residual connection
#             h = h + x
#             h = layer_norm(h)
            
#             output.append(h)
#             x = h
        
#         return torch.cat(output, dim=-1)
    
#     def forward(self, x, edge_index):
#         batch_size, win_size, num_nodes, num_feats = x.size()
#         state = torch.zeros(batch_size, num_nodes, self.state_size, device=x.device)
        
#         states = []
#         state_history = []
        
#         t0 = 0 if self.mode == 'forwards' else win_size - 1
#         tn = win_size if self.mode == 'forwards' else -1
#         step = 1 if self.mode == 'forwards' else -1
        
#         for t in range(t0, tn, step):
#             x_ = self.input_encoder(x[:, t])
#             state_out = self.iterate_layers(state=state, x=x_, edge_index=edge_index)
            
#             # Multi-scale temporal skip connections
#             state = state_out + state

#             # TODO: Tune coefficients (anche a mano, se non si vedono improvements, va bene anche rimuovere la history)
#             state_history.append(state)
            
#             if len(state_history) >= 2:
#                 state = state + 0.1 * state_history[-2]
#             if len(state_history) >= 4:
#                 state = state + 0.05 * state_history[-4]
            
#             if self.mode == 'forwards':
#                 states.append(state)
#             else:
#                 states.insert(0, state)
        
#         return torch.stack(states, dim=1)


# class EnhancedBiDirectionalSTGNN(nn.Module):
#     """
#     Enhanced Bidirectional Spatiotemporal Graph Neural Network.
#     Processes sequences bidirectionally with attention-based spatial aggregation.
#     """
#     def __init__(self, input_size, hidden_size, n_stations, output_dist: str, 
#                  n_layers=2, dropout_p=0.1, num_heads=4, **kwargs):
#         super().__init__()
        
#         # Input encoding
#         self.encoder = nn.Sequential(
#             nn.Linear(input_size, hidden_size),
#             nn.LayerNorm(hidden_size),
#             nn.SiLU()
#         )
        
#         # Station embeddings
#         self.station_embeddings = NodeEmbedding(n_stations, hidden_size)
        
#         # Bidirectional Graph RNNs
#         self.forward_model = EnhancedLayeredGraphRNN(
#             input_size=hidden_size, 
#             hidden_size=hidden_size, 
#             n_layers=n_layers, 
#             mode='forwards', 
#             dropout_p=dropout_p,
#             num_heads=num_heads
#         )
        
#         self.backward_model = EnhancedLayeredGraphRNN(
#             input_size=hidden_size, 
#             hidden_size=hidden_size, 
#             n_layers=n_layers, 
#             mode='backwards', 
#             dropout_p=dropout_p,
#             num_heads=num_heads
#         )
        
#         # Output distribution
#         self.output_distr = dist_to_layer[output_dist](input_size=hidden_size)
        
#         # Skip connection
#         self.skip_conn = nn.Linear(input_size, 2 * hidden_size * n_layers)
        
#         # Hierarchical readout (3 layers)
#         self.readout = nn.Sequential(
#             nn.Linear(2 * hidden_size * n_layers, hidden_size * 2),
#             nn.LayerNorm(hidden_size * 2),
#             nn.SiLU(),
#             nn.Dropout(p=dropout_p),
#             nn.Linear(hidden_size * 2, hidden_size),
#             nn.LayerNorm(hidden_size),
#             nn.SiLU(),
#             nn.Dropout(p=dropout_p),
#             nn.Linear(hidden_size, hidden_size)
#         )
        
#     def forward(self, x, edge_index):
#         x0 = x
        
#         # Encode and add embeddings
#         x = self.encoder(x)
#         x = x + self.station_embeddings()
        
#         # Bidirectional processing
#         states_forwards = self.forward_model(x, edge_index)
#         states_backwards = self.backward_model(x, edge_index)
        
#         # Concatenate states
#         states = torch.concatenate([states_forwards, states_backwards], dim=-1)
        
#         # Skip connection
#         states = states + self.skip_conn(x0)
        
#         # Hierarchical readout
#         output = self.readout(states)
        
#         return self.output_distr(output)



# """
# STGNN2
# """


# class LearnedGraph(nn.Module):
#     """
#     Learns a static adjacency matrix based on node embeddings.
#     """
#     def __init__(self, n_stations, emb_size=10):
#         super().__init__()
#         self.source_embeddings = NodeEmbedding(n_stations, emb_size)
#         self.target_embeddings = NodeEmbedding(n_stations, emb_size)

#     def forward(self):
#         # Calculate A = Softmax(ReLU(E1 @ E2.T))
#         source = self.source_embeddings()
#         target = self.target_embeddings()
#         logits = F.relu(torch.matmul(source, target.transpose(0, 1)))
#         adj = F.softmax(logits, dim=1)
#         return adj

# class DualSpatialBlock(nn.Module):
#     """
#     Combines Dynamic Attention (GAT) on physical graph 
#     AND Static Learned Correlations (WaveNet-style) on latent graph.
#     """
#     def __init__(self, in_channels, out_channels, heads=2, dropout=0.2):
#         super().__init__()
        
#         # Branch 1: Dynamic Attention on Physical Graph
#         self.gat = GATv2Conv(in_channels, out_channels // heads, 
#                              heads=heads, 
#                              dropout=dropout, 
#                              add_self_loops=True)
        
#         # Branch 2: Static Convolution on Learned Graph
#         self.dense_lin = nn.Linear(in_channels, out_channels)

#         # Fusion
#         self.fusion = nn.Linear(out_channels * 2, out_channels)
#         self.act = nn.ReLU()

#     def forward(self, x, edge_index, learned_adj):
#         """
#         x: [Batch, Nodes, Features]
#         edge_index: [2, Num_Edges] Physical graph connectivity
#         learned_adj: [Nodes, Nodes] Dense learned matrix
#         """
#         B, N, C = x.shape
        
#         # --- 1. GAT Branch (Sparse, Dynamic) ---
#         # The standard GATv2Conv expects (Total_Nodes, Features).
#         # We must flatten the batch and repeat the edge_index for each batch element.
        
#         x_flat = x.view(-1, C) # Shape: [B*N, C]
        
#         # Vectorized expansion of edge_index for the whole batch
#         src, dst = edge_index
#         # Create offsets: [0, N, 2N, ... ]
#         offsets = torch.arange(B, device=x.device) * N
        
#         # Broadcast offsets to edges: [B, 1] + [1, E] -> [B, E] -> flatten
#         src_batch = (src.unsqueeze(0) + offsets.unsqueeze(1)).flatten()
#         dst_batch = (dst.unsqueeze(0) + offsets.unsqueeze(1)).flatten()
        
#         edge_index_batch = torch.stack([src_batch, dst_batch], dim=0)
        
#         # Apply GAT on the "giant" batched graph
#         gat_out_flat = self.gat(x_flat, edge_index_batch)
        
#         # Reshape back: [B*N, C] -> [B, N, C]
#         gat_out = gat_out_flat.view(B, N, -1)
        
#         # --- 2. Learned Graph Branch (Dense, Static) ---
#         # Operation: X' = A_learned @ X @ W
#         # x: [B, N, In], adj: [N, N] -> [B, N, In]
#         # We use einsum to multiply the adjacency for every batch element
#         dense_agg = torch.einsum('nm, bmc -> bnc', learned_adj, x)
#         dense_out = self.dense_lin(dense_agg)
        
#         # --- 3. Fusion ---
#         combined = torch.cat([gat_out, dense_out], dim=-1)
#         return self.act(self.fusion(combined))

# class AttentionTCN_GNN(nn.Module):
#     def __init__(self, num_layers, input_size, output_dist, hidden_channels, 
#                  n_stations, kernel_size=3, dropout_p=0.2, causal_conv=False, 
#                  learned_adj_emb_size=10, gat_heads=2, **kwargs):
#         super().__init__()
        
#         if isinstance(hidden_channels, int):
#             hidden_channels = [hidden_channels]*num_layers
            
#         self.rearrange_for_gnn = rearrange 
        
#         # 1. Embeddings and Encoder
#         self.station_embeddings = NodeEmbedding(n_stations, hidden_channels[0])
#         self.encoder = nn.Linear(input_size, hidden_channels[0])
        
#         # 2. The Learned Graph Generator (WaveNet style)
#         self.learned_graph_gen = LearnedGraph(n_stations, learned_adj_emb_size)

#         # 3. Layers
#         self.tcn_layers = nn.ModuleList()
#         self.spatial_layers = nn.ModuleList()
#         self.norm_layers = nn.ModuleList()
        
#         for l in range(num_layers):
#             dilation = 2**l
#             in_size = hidden_channels[0] if l == 0 else hidden_channels[l-1]
#             out_size = hidden_channels[l]
            
#             # TCN (Temporal)
#             self.tcn_layers.append(
#                 TCNLayer(in_channels=in_size, out_channels=out_size, 
#                          kernel_size=kernel_size, dilation=dilation, 
#                          dropout_p=dropout_p, causal_conv=causal_conv)
#             )
            
#             # Dual Spatial (Dynamic Attention + Static Learned)
#             self.spatial_layers.append(
#                 DualSpatialBlock(in_channels=out_size, out_channels=out_size, 
#                                  heads=gat_heads, dropout=dropout_p)
#             )
            
#             self.norm_layers.append(nn.BatchNorm1d(num_features=out_size))
            
#         # 4. Output Head
#         self.output_distr = dist_to_layer[output_dist](input_size=hidden_channels[-1])

#     def forward(self, x, edge_index):
#         # x input: [Batch, LeadTime, Stations, Features]
        
#         # Encoding
#         x = self.encoder(x)
#         x = x + self.station_embeddings()
        
#         # Pre-compute the learned adjacency (Static per batch)
#         learned_adj = self.learned_graph_gen()
        
#         # Rearrange for TCN: [B, T, N, F] -> [(B N), F, T]
#         b, t, n, f = x.shape
#         x = rearrange(x, 'b t n f -> (b n) f t')
        
#         skips = []
        
#         for tcn, spatial, norm in zip(self.tcn_layers, self.spatial_layers, self.norm_layers):
            
#             # 1. Temporal Convolution
#             x, skip = tcn(x)
            
#             # 2. Reshape for Spatial Interaction
#             # TCN out: [(B N), F, T] -> [B, T, N, F] -> [(B T), N, F]
#             # We fold Batch and Time together for the spatial pass
#             x_spatial = rearrange(x, '(b n) f t -> (b t) n f', b=b, n=n)
            
#             # 3. Spatial Convolution (Attention + Learned)
#             x_spatial = spatial(x_spatial, edge_index, learned_adj)
            
#             # 4. Reshape back for Norm and next TCN
#             # [(B T), N, F] -> [B, T, N, F] -> [(B N), F, T]
#             x = rearrange(x_spatial, '(b t) n f -> (b n) f t', b=b)
            
#             # 5. Norm
#             x = norm(x)
            
#             skips.append(skip)
            
#         # Sum skip connections
#         skips_stack = torch.stack(skips, dim=-1) 
#         result = skips_stack.sum(dim=-1) + x 
        
#         # Final formatting: [(B N), F, T] -> [B, T, N, F]
#         output = rearrange(result, '(b n) f t -> b t n f', b=b)
        
#         return self.output_distr(output)






###########################
###### OUR VERSIONS #######
###########################

"""
Model0: Baseline (TCN)
Refactored: Fixed skip connection logic.
"""

class Model0TCN(nn.Module):
    """
    TCN-only baseline model (Model0).
    """

    def __init__(self, input_size, hidden_channels, n_stations, output_dist: str, num_layers, kernel_size, dropout_p, **kwargs):
        super().__init__()

        self.encoder = nn.Linear(input_size, hidden_channels)

        self.rearrange_for_tcn = Rearrange('b t n c -> (b n) c t')                
        self.rearrange_from_tcn = Rearrange('(b n) c t -> b t n c', n=n_stations)

        tcn_layers = []
        for l in range(num_layers):
            dilation = 2**l
            in_channels = hidden_channels

            tcn_layers.append(
                TCNLayer(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout_p=dropout_p,
                    causal_conv=False 
                )
            )

        self.tcn_layers = nn.ModuleList(tcn_layers)
        self.output_distr = dist_to_layer[output_dist](input_size=hidden_channels)

    def forward(self, x, edge_index=None): 
        x = self.encoder(x)                       
        x = self.rearrange_for_tcn(x)             

        skips = []
        for tcn_layer in self.tcn_layers:
            x, skip = tcn_layer(x) 
            skips.append(skip)

        # Standard TCN/WaveNet: Sum of skip connections forms the output
        skips_stack = torch.stack(skips, dim=-1)  
        skips_sum = skips_stack.sum(dim=-1)       
        
        output = self.rearrange_from_tcn(skips_sum) 

        return self.output_distr(output)
   

"""
STGNN1: EnhancedBiDirectionalSTGNN
Implemented GAT-GRU Cell for temporal gating.
"""

class EnhancedLayeredGraphRNN(nn.Module):
    """
    Enhanced Graph RNN with GATv2 for spatial mixing and GRU for temporal gating.
    Fixes the vanishing gradient problem of simple residual connections.
    """
    def __init__(self, input_size, hidden_size, n_layers=2, dropout_p=0.1, 
                 mode: Literal['forwards', 'backwards'] = 'forwards', num_heads=2, **kwargs):
        super().__init__(**kwargs)
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.mode = mode
        
        self.input_encoder = nn.Linear(input_size, hidden_size)
        
        self.gat_layers = nn.ModuleList()
        self.gru_cells = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for _ in range(n_layers):
            # Spatial: GATv2
            # concat=False keeps output dim = hidden_size (averages heads or projects them)
            # This is crucial for keeping dimensions consistent with GRU
            self.gat_layers.append(
                GATv2Conv(hidden_size, hidden_size, heads=num_heads, 
                          dropout=dropout_p, concat=False)
            )
            # Temporal: GRU Cell
            self.gru_cells.append(nn.GRUCell(hidden_size, hidden_size))
            self.layer_norms.append(nn.LayerNorm(hidden_size))

        self.dropout = nn.Dropout(p=dropout_p)
        self.state_size = hidden_size * n_layers
        
    def forward(self, x, edge_index):
        batch_size, win_size, num_nodes, num_feats = x.size()
        
        # Initialize states
        state = torch.zeros(batch_size, num_nodes, self.state_size, device=x.device)
        
        # Efficient Batch Graph Construction
        # Instead of repeating edge_index every step inside GAT, we prepare the batch logic here.
        src, dst = edge_index
        offsets = torch.arange(batch_size, device=x.device) * num_nodes
        src_batch = (src.unsqueeze(0) + offsets.unsqueeze(1)).flatten()
        dst_batch = (dst.unsqueeze(0) + offsets.unsqueeze(1)).flatten()
        batch_edge_index = torch.stack([src_batch, dst_batch], dim=0)
        
        states = []
        
        t0 = 0 if self.mode == 'forwards' else win_size - 1
        tn = win_size if self.mode == 'forwards' else -1 
        step = 1 if self.mode == 'forwards' else -1
        
        for t in range(t0, tn, step):
            # Encode input for this timestep
            x_curr = self.input_encoder(x[:, t]) # [B, N, H]
            
            # Flatten B and N for PyG compatibility: [B*N, H]
            current_input = x_curr.view(-1, self.hidden_size)
            
            # Split state into layers: [B, N, H*L] -> [B*N, L, H]
            prev_states = state.view(batch_size * num_nodes, self.n_layers, self.hidden_size)
            new_layer_states = []
            
            for l in range(self.n_layers):
                h_prev = prev_states[:, l, :] # [B*N, H]
                
                # 1. Spatial Aggregation (GAT)
                # GAT sees the current input of the layer
                x_spatial = self.gat_layers[l](current_input, batch_edge_index)
                x_spatial = self.dropout(x_spatial)
                
                # 2. Temporal Gating (GRU)
                # GRU updates state based on spatial input and previous state
                h_new = self.gru_cells[l](x_spatial, h_prev)
                
                # Norm & Skip
                # Standard GRU outputs are usually clean, but LayerNorm helps in deep stacks
                h_new = self.layer_norms[l](h_new)
                
                new_layer_states.append(h_new)
                current_input = h_new # Output of layer l is input to layer l+1
            
            # Reconstruct state for next timestep
            state_flat = torch.cat(new_layer_states, dim=-1) # [B*N, H*L]
            state = state_flat.view(batch_size, num_nodes, -1)
            
            if self.mode == 'forwards':
                states.append(state)
            else:
                states.insert(0, state)
        
        return torch.stack(states, dim=1)


class EnhancedBiDirectionalSTGNN(nn.Module):
    """
    Refactored STGNN1 container.
    """
    def __init__(self, input_size, hidden_size, n_stations, output_dist: str, 
                 n_layers=2, dropout_p=0.1, num_heads=4, **kwargs):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU()
        )
        
        self.station_embeddings = NodeEmbedding(n_stations, hidden_size)
        
        # Use the new GAT-GRU RNN
        self.forward_model = EnhancedLayeredGraphRNN(
            input_size=hidden_size, 
            hidden_size=hidden_size, 
            n_layers=n_layers, 
            mode='forwards', 
            dropout_p=dropout_p,
            num_heads=num_heads
        )
        
        self.backward_model = EnhancedLayeredGraphRNN(
            input_size=hidden_size, 
            hidden_size=hidden_size, 
            n_layers=n_layers, 
            mode='backwards', 
            dropout_p=dropout_p,
            num_heads=num_heads
        )
        
        self.output_distr = dist_to_layer[output_dist](input_size=hidden_size)
        
        self.skip_conn = nn.Linear(input_size, 2 * hidden_size * n_layers)
        
        # Hierarchical readout
        self.readout = nn.Sequential(
            nn.Linear(2 * hidden_size * n_layers, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.SiLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, x, edge_index):
        x0 = x
        x = self.encoder(x)
        x = x + self.station_embeddings()
        
        states_forwards = self.forward_model(x, edge_index)
        states_backwards = self.backward_model(x, edge_index)
        
        states = torch.concatenate([states_forwards, states_backwards], dim=-1)
        states = states + self.skip_conn(x0)
        
        output = self.readout(states)
        
        return self.output_distr(output)


"""
STGNN2: AttentionTCN_GNN
Refactored: Optimized "Giant Graph" batching and Logic.
"""

class LearnedGraph(nn.Module):
    """
    Learns a static adjacency matrix.
    Fixed: Removed ReLU before Softmax to allow small non-zero probabilities.
    """
    def __init__(self, n_stations, emb_size=10):
        super().__init__()
        self.source_embeddings = NodeEmbedding(n_stations, emb_size)
        self.target_embeddings = NodeEmbedding(n_stations, emb_size)

    def forward(self):
        source = self.source_embeddings()
        target = self.target_embeddings()
        logits = torch.matmul(source, target.transpose(0, 1))
        # ReLU removed here. Softmax(logits) handles negative values correctly.
        adj = F.softmax(logits, dim=1)
        return adj

class EfficientDualSpatialBlock(nn.Module):
    """
    Improved Spatial Block with Gating and Residuals.
    Architecture:
        1. GAT Branch (Physical Graph)
        2. Learned Graph Branch (Latent Correlations)
        3. Gated Fusion (tanh * sigmoid)
        4. Residual Connection (Input + Spatial)
    """
    def __init__(self, in_channels, out_channels, heads=2, dropout=0.2):
        super().__init__()
        
        # We need a projection if input/output dimensions differ (for the residual)
        self.residual_proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        
        # Branch 1: Dynamic Attention (PyG)
        self.gat = GATv2Conv(in_channels, out_channels // heads, 
                             heads=heads, 
                             dropout=dropout, 
                             add_self_loops=True,
                             concat=True)
        
        # Branch 2: Static Learned Graph
        self.dense_lin = nn.Linear(in_channels, out_channels)

        # Gated Fusion mechanism
        # We project combined features into 2 * out_channels (half for value, half for gate)
        self.fusion = nn.Linear(out_channels * 2, out_channels * 2)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x_flat, batch_edge_index, learned_adj, original_batch_shape):
        """
        x_flat: [B*T*N, C]
        """
        B_times_T, N = original_batch_shape
        
        # --- 0. Residual Path ---
        residual = self.residual_proj(x_flat)
        
        # --- 1. GAT Branch ---
        gat_out = self.gat(x_flat, batch_edge_index) # [B*T*N, C]
        
        # --- 2. Learned Graph Branch ---
        x_reshaped = x_flat.view(B_times_T, N, -1)
        dense_agg = torch.einsum('nm, bmc -> bnc', learned_adj, x_reshaped)
        dense_out = self.dense_lin(dense_agg)
        dense_out = dense_out.reshape(x_flat.shape[0], -1) 
        
        # --- 3. Gated Fusion ---
        combined = torch.cat([gat_out, dense_out], dim=-1) # [..., 2*C]
        
        # Project to 2x channels for gating
        fused = self.fusion(combined) 
        val, gate = torch.chunk(fused, 2, dim=-1)
        
        # Gating: Tanh for information, Sigmoid for selection
        spatial_signal = torch.tanh(val) * torch.sigmoid(gate)
        spatial_signal = self.dropout(spatial_signal)
        
        # --- 4. Add Residual ---
        # The result is: Original_Features + Processed_Neighbor_Info
        out = residual + spatial_signal
        
        return self.norm(out)

# class AttentionTCN_GNN(nn.Module):
#     def __init__(self, num_layers, input_size, output_dist, hidden_channels, 
#                  n_stations, kernel_size=3, dropout_p=0.2, causal_conv=False, 
#                  learned_adj_emb_size=10, gat_heads=2, **kwargs):
#         super().__init__()
        
#         if isinstance(hidden_channels, int):
#             hidden_channels = [hidden_channels]*num_layers
            
#         self.station_embeddings = NodeEmbedding(n_stations, hidden_channels[0])
#         self.encoder = nn.Linear(input_size, hidden_channels[0])
#         self.learned_graph_gen = LearnedGraph(n_stations, learned_adj_emb_size)

#         self.tcn_layers = nn.ModuleList()
#         self.spatial_layers = nn.ModuleList()
        
#         for l in range(num_layers):
#             dilation = 2**l
#             in_size = hidden_channels[0] if l == 0 else hidden_channels[l-1]
#             out_size = hidden_channels[l]
            
#             # TCN Layer (Temporal)
#             self.tcn_layers.append(
#                 TCNLayer(in_channels=in_size, out_channels=out_size, 
#                          kernel_size=kernel_size, dilation=dilation, 
#                          dropout_p=dropout_p, causal_conv=causal_conv)
#             )
            
#             # Spatial Layer (Spatio-Temporal Mixing)
#             self.spatial_layers.append(
#                 EfficientDualSpatialBlock(in_channels=out_size, out_channels=out_size, 
#                                           heads=gat_heads, dropout=dropout_p)
#             )
            
#             # Note: BatchNorm is removed here because LayerNorm is added 
#             # inside the Spatial Block, which is generally better for GNNs.
            
#         self.output_distr = dist_to_layer[output_dist](input_size=hidden_channels[-1])

#     def forward(self, x, edge_index):
#         # x: [B, T, N, F]
#         x = self.encoder(x)
#         x = x + self.station_embeddings()
        
#         learned_adj = self.learned_graph_gen()
        
#         B, T, N, C = x.shape
        
#         # Efficient Batch Edge Index Construction
#         src, dst = edge_index
#         offsets = torch.arange(B * T, device=x.device) * N
#         src_batch = (src.unsqueeze(0) + offsets.unsqueeze(1)).flatten()
#         dst_batch = (dst.unsqueeze(0) + offsets.unsqueeze(1)).flatten()
#         batch_edge_index = torch.stack([src_batch, dst_batch], dim=0)
        
#         # TCN requires: [(B N), C, T]
#         x = rearrange(x, 'b t n c -> (b n) c t')
        
#         skips = []
        
#         for tcn, spatial in zip(self.tcn_layers, self.spatial_layers):
            
#             # 1. Temporal Convolution
#             x, skip = tcn(x) # Out: [(B N), C, T]
            
#             # 2. Reshape for Spatial
#             # (b n) c t -> (b t) n c
#             x_spatial = rearrange(x, '(b n) c t -> (b t) n c', b=B, n=N)
#             x_spatial_flat = x_spatial.reshape(-1, x_spatial.shape[-1]) 
            
#             # 3. Spatial Convolution with Residual & Gating
#             out_spatial = spatial(x_spatial_flat, batch_edge_index, learned_adj, (B*T, N))
            
#             # 4. Reshape back
#             out_spatial = out_spatial.view(B, T, N, -1)
#             x = rearrange(out_spatial, 'b t n c -> (b n) c t')
            
#             # No external Norm here, it's inside the spatial block
#             skips.append(skip)
            
#         # Sum skip connections
#         output = torch.stack(skips, dim=-1).sum(dim=-1) 
        
#         # Final formatting
#         output = rearrange(output, '(b n) c t -> b t n c', b=B, n=N)
        
#         return self.output_distr(output)



from einops import repeat

class AttentionTCN_GNN(nn.Module):
    """
    Refined Architecture incorporating insights:
    1. Explicit Horizon (Lead Time) Embeddings.
    2. Lightweight GATv2 for local physical diffusion.
    3. Global Spatial Transformer for long-range dynamic correlations.
    4. Input-Injection Gating to preserve raw NWP signals.
    """
    def __init__(self, num_layers, input_size, output_dist, hidden_channels, 
                 n_stations, kernel_size=3, dropout_p=0.2, causal_conv=False, 
                 gat_heads=2, max_lead_time=96, **kwargs):
        super().__init__()
        
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]*num_layers
            
        self.hidden_dim = hidden_channels[0]
        
        # --- Insight 1: Embeddings ---
        # Station Embedding (Spatial Identity)
        self.station_emb = NodeEmbedding(n_stations, self.hidden_dim)
        # Horizon Embedding (Temporal Identity - Critical for error growth modeling)
        self.horizon_emb = nn.Embedding(max_lead_time + 1, self.hidden_dim)
        
        # Initial Projection
        self.encoder = nn.Linear(input_size, self.hidden_dim)

        # --- Backbone: Interleaved TCN + Local GNN ---
        self.tcn_layers = nn.ModuleList()
        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        for l in range(num_layers):
            dilation = 2**l
            in_size = hidden_channels[0] if l == 0 else hidden_channels[l-1]
            out_size = hidden_channels[l]
            
            # TCN: Extract temporal features per station
            self.tcn_layers.append(
                TCNLayer(in_channels=in_size, out_channels=out_size, 
                         kernel_size=kernel_size, dilation=dilation, 
                         dropout_p=dropout_p, causal_conv=causal_conv)
            )
            
            # GATv2: Process LOCAL physical neighbors (sparse graph)
            # We keep this lightweight compared to the previous DualBlock
            self.gat_layers.append(
                GATv2Conv(out_size, out_size, heads=gat_heads, 
                          concat=False, dropout=dropout_p, add_self_loops=True)
            )
            
            self.norm_layers.append(nn.LayerNorm(out_size))
            
        # --- Insight 2: Global Spatial Refinement ---
        # After local processing, we let ALL stations talk to EACH OTHER dynamically.
        # This replaces the static "LearnedGraph".
        self.global_spatial_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, 
            num_heads=4, 
            batch_first=True, 
            dropout=dropout_p
        )
        self.attn_norm = nn.LayerNorm(self.hidden_dim)

        # --- Insight 3: Gated Input Injection ---
        self.raw_skip_proj = nn.Linear(input_size, self.hidden_dim)
        self.gate_generator = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Output
        self.output_distr = dist_to_layer[output_dist](input_size=self.hidden_dim)

    def forward(self, x, edge_index):
        # x: [B, T, N, F]
        B, T, N, F = x.shape
        
        # 1. Embeddings & Encoding
        h = self.encoder(x) # [B, T, N, H]
        
        # Add Station Embedding
        h = h + self.station_emb() # Broadcasting over B and T
        
        # Add Horizon Embedding
        # Create indices [0, 1, ..., T-1] repeated for batch and stations
        horizon_ids = torch.arange(T, device=x.device)
        # We need to broadcast this to [B, T, N, H]
        t_emb = self.horizon_emb(horizon_ids).unsqueeze(0).unsqueeze(2) # [1, T, 1, H]
        h = h + t_emb

        # 2. Backbone Processing
        # Prepare for TCN: [(B N), H, T]
        h_tcn = rearrange(h, 'b t n c -> (b n) c t')
        
        # Prepare Batch Edge Index for GAT (Pre-computed once)
        src, dst = edge_index
        offsets = torch.arange(B * T, device=x.device) * N
        src_batch = (src.unsqueeze(0) + offsets.unsqueeze(1)).flatten()
        dst_batch = (dst.unsqueeze(0) + offsets.unsqueeze(1)).flatten()
        batch_edge_index = torch.stack([src_batch, dst_batch], dim=0)

        skips = []
        
        for tcn, gat, norm in zip(self.tcn_layers, self.gat_layers, self.norm_layers):
            
            # Temporal Step
            h_tcn, skip = tcn(h_tcn) # [(B N), H, T]
            
            # Reshape for Spatial Step: flatten B and T into one "batch" dim
            # [(B N), H, T] -> [(B T), N, H]
            h_spatial = rearrange(h_tcn, '(b n) c t -> (b t) n c', b=B, n=N)
            h_spatial_flat = h_spatial.reshape(-1, self.hidden_dim) # [(B T N), H]
            
            # Physical Graph Step (Local Diffusion)
            h_gat = gat(h_spatial_flat, batch_edge_index)
            
            # Residual + Norm
            h_gat = h_gat.view(B*T, N, -1)
            h_spatial = norm(h_spatial + h_gat) # Residual connection crucial here
            
            # Back to TCN shape
            h_tcn = rearrange(h_spatial, '(b t) n c -> (b n) c t', b=B, t=T)
            
            skips.append(skip)

        # Sum WaveNet skips
        h_deep = torch.stack(skips, dim=-1).sum(dim=-1) # [(B N), H, T]
        h_deep = rearrange(h_deep, '(b n) h t -> (b t) n h', b=B, n=N) # [(B T), N, H]

        # 3. Global Spatial Refinement (Transformer)
        # Allows distant stations to correct each other dynamically
        # Query=Key=Value=h_deep
        attn_out, _ = self.global_spatial_attn(h_deep, h_deep, h_deep)
        h_refined = self.attn_norm(h_deep + attn_out)

        # 4. Gated Input Injection
        # Reshape to [B, T, N, H]
        h_refined = rearrange(h_refined, '(b t) n h -> b t n h', b=B, t=T)
        
        # Compute Gate
        gate = torch.sigmoid(self.gate_generator(h_refined))
        
        # Skip projection of raw input
        raw_proj = self.raw_skip_proj(x)
        
        # Final fusion: Blend deep features with raw input features
        h_final = gate * h_refined + (1 - gate) * raw_proj

        return self.output_distr(h_final)