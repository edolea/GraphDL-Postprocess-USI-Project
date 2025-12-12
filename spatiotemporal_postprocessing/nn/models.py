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


###########################
####### OUR MODELS ########
###########################


"""
Model0
"""

class Model0TCN(nn.Module):
    """
    TCN-only baseline model (Model0).
    """

    def __init__(self, input_size, hidden_channels, n_stations, output_dist: str, num_layers, kernel_size, dropout_p, **kwargs):
        super().__init__()

        self.encoder = nn.Linear(input_size, hidden_channels)

        self.rearrange_from_tcn = Rearrange('(b n) c t -> b t n c', n=n_stations)
        self.rearrange_for_tcn = Rearrange('b t n c -> (b n) c t')                
        
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
                    causal_conv=False # Always false for baseline 
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

        skips_stack = torch.stack(skips, dim=-2)  
        result = skips_stack.sum(dim=-2) + x     
        
        output = self.rearrange_from_tcn(result) 

        return self.output_distr(output)
   

"""
STGNN1
"""

class EnhancedLayeredGraphRNN(nn.Module):
    """
    add explanation
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

            self.gat_layers.append(
                GATv2Conv(hidden_size, hidden_size, heads=num_heads, 
                          dropout=dropout_p, concat=False)
            )

            self.gru_cells.append(nn.GRUCell(hidden_size, hidden_size))
            self.layer_norms.append(nn.LayerNorm(hidden_size))

        self.dropout = nn.Dropout(p=dropout_p)
        self.state_size = hidden_size * n_layers
        
    def forward(self, x, edge_index):
        batch_size, win_size, num_nodes, num_feats = x.size()
        
        # Initialize states
        state = torch.zeros(batch_size, num_nodes, self.state_size, device=x.device)
        
        # Efficient Batch Graph Construction
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
            
            x_curr = self.input_encoder(x[:, t]) # [B, N, H]
            
            current_input = x_curr.view(-1, self.hidden_size) # [B*N, H] (for PyG compatibilty)
            
            prev_states = state.view(batch_size * num_nodes, self.n_layers, self.hidden_size) # [B, N, H*L] -> [B*N, L, H]
            new_layer_states = []
            
            for l in range(self.n_layers):
                h_prev = prev_states[:, l, :] # [B*N, H]
                
                # 1. Spatial Aggregation (GAT)
                x_spatial = self.gat_layers[l](current_input, batch_edge_index)
                x_spatial = self.dropout(x_spatial)
                
                # 2. Temporal Gating (GRU)
                h_new = self.gru_cells[l](x_spatial, h_prev)
                
                # Norm & Skip
                h_new = self.layer_norms[l](h_new)
                
                new_layer_states.append(h_new)
                current_input = h_new
            
            # Reconstruct state
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
STGNN2
"""

class EnhancedTCN_GNN(nn.Module):
    def __init__(self, num_layers, input_size, output_dist, hidden_channels, 
                 n_stations, kernel_size=3, dropout_p=0.2, causal_conv=True, 
                 gat_heads=2, max_lead_time=96, **kwargs):
        super().__init__()
        
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]*num_layers
            
        self.hidden_dim = hidden_channels[0]
        
        # --- Insight 1: Embeddings ---
        self.station_emb = NodeEmbedding(n_stations, self.hidden_dim)
        self.horizon_emb = nn.Embedding(max_lead_time + 1, self.hidden_dim)
        
        self.encoder = nn.Linear(input_size, self.hidden_dim)

        # --- Backbone: Interleaved TCN + Local GNN ---
        self.tcn_layers = nn.ModuleList()
        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        for l in range(num_layers):
            dilation = 2**l
            in_size = hidden_channels[0] if l == 0 else hidden_channels[l-1]
            out_size = hidden_channels[l]
            
            self.tcn_layers.append(
                TCNLayer(in_channels=in_size, out_channels=out_size, 
                         kernel_size=kernel_size, dilation=dilation, 
                         dropout_p=dropout_p, causal_conv=causal_conv)
            )
            
            self.gat_layers.append(
                GATv2Conv(out_size, out_size, heads=gat_heads, 
                          concat=False, dropout=dropout_p, add_self_loops=True)
            )
            
            self.norm_layers.append(nn.LayerNorm(out_size))
            
        self.global_spatial_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, 
            num_heads=4, 
            batch_first=True, 
            dropout=dropout_p
        )
        self.attn_norm = nn.LayerNorm(self.hidden_dim)

        self.raw_skip_proj = nn.Linear(input_size, self.hidden_dim)
        self.gate_generator = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.output_distr = dist_to_layer[output_dist](input_size=self.hidden_dim)

    def forward(self, x, edge_index):

        B, T, N, F = x.shape
        
        h = self.encoder(x)
        
        h = h + self.station_emb()
        
        horizon_ids = torch.arange(T, device=x.device)
        t_emb = self.horizon_emb(horizon_ids).unsqueeze(0).unsqueeze(2)
        h = h + t_emb

        h_tcn = rearrange(h, 'b t n c -> (b n) c t')
        
        src, dst = edge_index
        offsets = torch.arange(B * T, device=x.device) * N
        src_batch = (src.unsqueeze(0) + offsets.unsqueeze(1)).flatten()
        dst_batch = (dst.unsqueeze(0) + offsets.unsqueeze(1)).flatten()
        batch_edge_index = torch.stack([src_batch, dst_batch], dim=0)

        skips = []
        
        for tcn, gat, norm in zip(self.tcn_layers, self.gat_layers, self.norm_layers):
            
            h_tcn, skip = tcn(h_tcn)
            
            h_spatial = rearrange(h_tcn, '(b n) c t -> (b t) n c', b=B, n=N)
            h_spatial_flat = h_spatial.reshape(-1, self.hidden_dim)
            
            h_gat = gat(h_spatial_flat, batch_edge_index)
            
            h_gat = h_gat.view(B*T, N, -1)
            h_spatial = norm(h_spatial + h_gat)
            
            h_tcn = rearrange(h_spatial, '(b t) n c -> (b n) c t', b=B, t=T)

            h_final_tcn = h_tcn.clone()
            
            skips.append(skip)

        h_deep = torch.stack(skips, dim=-1).sum(dim=-1) + h_final_tcn   # [(B N), H, T]
        h_deep = rearrange(h_deep, '(b n) h t -> (b t) n h', b=B, n=N)  # [(B T), N, H]

        attn_out, _ = self.global_spatial_attn(h_deep, h_deep, h_deep)
        h_refined = self.attn_norm(h_deep + attn_out)

        h_refined = rearrange(h_refined, '(b t) n h -> b t n h', b=B, t=T)
        
        gate = torch.sigmoid(self.gate_generator(h_refined))
        
        raw_proj = self.raw_skip_proj(x)
        h_final = gate * h_refined + (1 - gate) * raw_proj

        return self.output_distr(h_final)