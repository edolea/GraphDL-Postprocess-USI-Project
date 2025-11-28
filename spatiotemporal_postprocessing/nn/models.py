from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn as nn
import torch
# import torch.nn.functional as F
from typing import Literal
from tsl.nn.layers import GatedGraphNetwork, NodeEmbedding, BatchNorm
from tsl.nn.models import GraphWaveNetModel
from spatiotemporal_postprocessing.nn.probabilistic_layers import dist_to_layer
from spatiotemporal_postprocessing.nn.prototypes import TCNLayer


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

###########################
###### OUR VERSIONS #######
###########################

"""
Model0: baseline
"""

class Model0TCN(nn.Module):
    """
    TCN-only baseline model (Model0) for statistical postprocessing.

    Architecture:
    - same as TCN_GNN by MeteoSwiss, but with no graph information
    (NOTE: this is a downgrade, the purpose is only to check if the graph realtions actually improve our modeling)

    Args:
        input_size: Number of input features per station
        hidden_channels: Hidden dimension size
        n_stations: Number of weather stations
        output_dist: Distribution type (e.g., 'LogNormal')
        num_layers: Number of TCN layers (default: 4)
        kernel_size: Convolution kernel size (default: 3)
        dropout_p: Dropout probability (default: 0.2)
    """

    def __init__(self, input_size, hidden_channels, n_stations, output_dist: str, num_layers, kernel_size, dropout_p, **kwargs):
        super().__init__()

        self.encoder = nn.Linear(input_size, hidden_channels)

        self.rearrange_for_tcn = Rearrange('b t n c -> (b n) c t')                # [batch, time, stations, features] to [batch*stations, features, time]
        self.rearrange_from_tcn = Rearrange('(b n) c t -> b t n c', n=n_stations) # [batch*stations, channels, time] back to [batch, time, stations, channels]

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
                    causal_conv=True  # Ensure no future information leakage (always true)
                )
            )

        self.tcn_layers = nn.ModuleList(tcn_layers)

        self.output_distr = dist_to_layer[output_dist](input_size=hidden_channels)

    def forward(self, x, edge_index=None): # Graph edge indices (accepted but ignored for compatibility)
        
        x = self.encoder(x)                       # [batch, time, stations, features] -> [batch, time, stations, hidden]
        x = self.rearrange_for_tcn(x)             # [batch, time, stations, hidden] -> [batch*stations, hidden, time]

        skips = []
        for tcn_layer in self.tcn_layers:
            x, skip = tcn_layer(x) # TCNLayer returns (residual_output, skip_output)
            skips.append(skip)

        skips_stack = torch.stack(skips, dim=-1)  # [batch*stations, hidden, time, num_layers]
        skips_sum = skips_stack.sum(dim=-1)       # [batch*stations, hidden, time]
        output = skips_sum + x                    # [batch*stations, hidden, time]
        output = self.rearrange_from_tcn(output)  # [batch*stations, hidden, time] -> [batch, time, stations, hidden]

        return self.output_distr(output)
   

"""
STGNN1
"""

class GraphAttentionLayer(nn.Module):
    """
    (Replaces the baseline's GatedGraphNetwork from MeteoSwiss)

    Graph Attention Layer for spatiotemporal message passing.
    Allows nodes to dynamically weight the importance of their neighbors.
    """
    def __init__(self, in_features, out_features, num_heads=4, dropout_p=0.1, concat=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        
        self.W = nn.Linear(in_features, out_features * num_heads, bias=False)
        self.a_src = nn.Parameter(torch.Tensor(num_heads, out_features))
        self.a_dst = nn.Parameter(torch.Tensor(num_heads, out_features))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout_p)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(1))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(1))
        
    def forward(self, x, edge_index):
        batch_size, num_nodes, _ = x.size()
        
        # Transform features
        h = self.W(x)
        h = h.view(batch_size, num_nodes, self.num_heads, self.out_features)
        h = h.permute(0, 2, 1, 3)
        
        # Compute attention scores
        src_idx, dst_idx = edge_index[0], edge_index[1]
        alpha_src = (h[:, :, src_idx] * self.a_src.unsqueeze(0).unsqueeze(2)).sum(dim=-1)
        alpha_dst = (h[:, :, dst_idx] * self.a_dst.unsqueeze(0).unsqueeze(2)).sum(dim=-1)
        alpha = self.leaky_relu(alpha_src + alpha_dst)
        
        # Softmax normalization per node
        alpha_max = torch.zeros(batch_size, self.num_heads, num_nodes, device=x.device)
        alpha_max.scatter_reduce_(2, dst_idx.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1), 
                                   alpha, reduce='amax', include_self=False)
        alpha_max = alpha_max[:, :, dst_idx]
        alpha = torch.exp(alpha - alpha_max)
        
        alpha_sum = torch.zeros(batch_size, self.num_heads, num_nodes, device=x.device)
        alpha_sum.scatter_add_(2, dst_idx.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1), alpha)
        alpha_sum = alpha_sum[:, :, dst_idx] + 1e-10
        alpha = alpha / alpha_sum
        alpha = self.dropout(alpha)
        
        # Aggregate with attention weights
        h_edges = h[:, :, src_idx] * alpha.unsqueeze(-1)
        out = torch.zeros(batch_size, self.num_heads, num_nodes, self.out_features, device=x.device)
        out.scatter_add_(2, dst_idx.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(
            batch_size, self.num_heads, -1, self.out_features), h_edges)
        
        if self.concat:
            out = out.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, -1)
        else:
            out = out.mean(dim=1)
            
        return out
    
# If too slow, try this attention version, should be faster (adn hopeful wont impact performace)
# class GraphAttentionLayer(nn.Module):
#     """Simplified attention: Single head instead of multi-head - ~3x faster ????"""
#     def __init__(self, in_features, out_features, dropout_p=0.1):
#         super().__init__()
#         self.W = nn.Linear(in_features, out_features, bias=False)
#         self.a = nn.Parameter(torch.Tensor(out_features, 1))
#         self.leaky_relu = nn.LeakyReLU(0.2)
#         self.dropout = nn.Dropout(dropout_p)
#         nn.init.xavier_uniform_(self.W.weight)
#         nn.init.xavier_uniform_(self.a)
        
#     def forward(self, x, edge_index):
#         batch_size, num_nodes, _ = x.size()
#         h = self.W(x)  # [batch, nodes, features]
        
#         # Simplified attention: just use source node features
#         src_idx, dst_idx = edge_index[0], edge_index[1]
#         alpha = self.leaky_relu((h[:, src_idx] * self.a.t()).sum(dim=-1))
        
#         # Softmax per target node
#         alpha_max = torch.zeros(batch_size, num_nodes, device=x.device)
#         alpha_max.scatter_reduce_(1, dst_idx.unsqueeze(0).expand(batch_size, -1), 
#                                    alpha, reduce='amax', include_self=False)
#         alpha = torch.exp(alpha - alpha_max[:, dst_idx])
        
#         alpha_sum = torch.zeros(batch_size, num_nodes, device=x.device)
#         alpha_sum.scatter_add_(1, dst_idx.unsqueeze(0).expand(batch_size, -1), alpha)
#         alpha = alpha / (alpha_sum[:, dst_idx] + 1e-10)
#         alpha = self.dropout(alpha)
        
#         # Aggregate
#         h_agg = torch.zeros(batch_size, num_nodes, h.size(-1), device=x.device)
#         h_agg.scatter_add_(1, dst_idx.unsqueeze(0).unsqueeze(-1).expand(
#             batch_size, -1, h.size(-1)), h[:, src_idx] * alpha.unsqueeze(-1))
        
#         return h_agg


class EnhancedLayeredGraphRNN(nn.Module):
    """
    (Replaces the baseline's LayeredGraphRNN from MeteoSwiss)

    Enhanced Graph RNN with attention mechanisms and multi-scale temporal aggregation.
    """
    def __init__(self, input_size, hidden_size, n_layers=2, dropout_p=0.1, 
                 mode: Literal['forwards', 'backwards'] = 'forwards', num_heads=4, **kwargs):
        super().__init__(**kwargs)
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.mode = mode
        
        # Input encoding with 2-layer MLP
        self.input_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Graph Attention Layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(
                in_features=hidden_size * 2,
                out_features=hidden_size,
                num_heads=num_heads, # TODO to remove if testing on single head attention 
                dropout_p=dropout_p,
                concat=True
            ) for _ in range(n_layers)
        ])

        # Projection layers for multi-head outputs
        self.head_projection = nn.ModuleList([
            nn.Linear(hidden_size * num_heads, hidden_size) 
            for _ in range(n_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(p=dropout_p)
        self.state_size = hidden_size * n_layers
        
    def iterate_layers(self, state, x, edge_index):
        output = []
        state_ = rearrange(state, "b n ... (h l) -> l b n ... h", l=self.n_layers)
        
        for l, (gat_layer, proj_layer, layer_norm) in enumerate(
            zip(self.gat_layers, self.head_projection, self.layer_norms)
        ):
            state_in_ = state_[l]
            
            # Concatenate hidden state with input
            input_ = torch.concatenate([state_in_, x], dim=-1)
            
            # Apply graph attention
            h = gat_layer(input_, edge_index)
            h = proj_layer(h)
            h = self.dropout(h)
            
            # Residual connection
            h = h + x
            h = layer_norm(h)
            
            output.append(h)
            x = h
        
        return torch.cat(output, dim=-1)
    
    def forward(self, x, edge_index):
        batch_size, win_size, num_nodes, num_feats = x.size()
        state = torch.zeros(batch_size, num_nodes, self.state_size, device=x.device)
        
        states = []
        state_history = []
        
        t0 = 0 if self.mode == 'forwards' else win_size - 1
        tn = win_size if self.mode == 'forwards' else -1
        step = 1 if self.mode == 'forwards' else -1
        
        for t in range(t0, tn, step):
            x_ = self.input_encoder(x[:, t])
            state_out = self.iterate_layers(state=state, x=x_, edge_index=edge_index)
            
            # Multi-scale temporal skip connections
            state = state_out + state

            # TODO: Tune coefficients (anche a mano, se non si vedono improvements, va bene anche rimuovere la history)
            state_history.append(state)
            
            if len(state_history) >= 2:
                state = state + 0.1 * state_history[-2]
            if len(state_history) >= 4:
                state = state + 0.05 * state_history[-4]
            
            if self.mode == 'forwards':
                states.append(state)
            else:
                states.insert(0, state)
        
        return torch.stack(states, dim=1)


class EnhancedBiDirectionalSTGNN(nn.Module):
    """
    Enhanced Bidirectional Spatiotemporal Graph Neural Network.
    Processes sequences bidirectionally with attention-based spatial aggregation.
    """
    def __init__(self, input_size, hidden_size, n_stations, output_dist: str, 
                 n_layers=2, dropout_p=0.1, num_heads=4, **kwargs):
        super().__init__()
        
        # Input encoding
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU()
        )
        
        # Station embeddings
        self.station_embeddings = NodeEmbedding(n_stations, hidden_size)
        
        # Bidirectional Graph RNNs
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
        
        # Output distribution
        self.output_distr = dist_to_layer[output_dist](input_size=hidden_size)
        
        # Skip connection
        self.skip_conn = nn.Linear(input_size, 2 * hidden_size * n_layers)
        
        # Hierarchical readout (3 layers)
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
        
        # Encode and add embeddings
        x = self.encoder(x)
        x = x + self.station_embeddings()
        
        # Bidirectional processing
        states_forwards = self.forward_model(x, edge_index)
        states_backwards = self.backward_model(x, edge_index)
        
        # Concatenate states
        states = torch.concatenate([states_forwards, states_backwards], dim=-1)
        
        # Skip connection
        states = states + self.skip_conn(x0)
        
        # Hierarchical readout
        output = self.readout(states)
        
        return self.output_distr(output)