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
    Temporal Convolutional Network (TCN) baseline model for weather postprocessing.
    
    This non-graph baseline processes each weather station independently using stacked 
    dilated temporal convolutions to capture multi-scale temporal patterns. It serves 
    as a control to quantify the performance gap when spatial dependencies between 
    stations are completely ignored.
    
    Architecture:
        - Input encoding: Linear projection to hidden dimension
        - Stacked TCN layers with exponentially growing dilation rates (2^l for layer l)
        - Each layer produces both a residual output and a skip connection
        - Final output combines all temporal scales: sum(skips) + final_layer_output
        - Non-causal convolutions: can look both forward and backward in forecast window
    
    Args:
        input_size (int): Number of input features per timestep
        hidden_channels (int): Hidden dimension size
        n_stations (int): Number of weather stations
        output_dist (str): Output distribution type (e.g., 'LogNormal')
        num_layers (int): Number of TCN layers
        kernel_size (int): Temporal kernel size for convolutions
        dropout_p (float): Dropout probability
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
    Enhanced Layered Graph RNN for spatiotemporal weather postprocessing.
    
    This class implements a single directional (forward or backward) spatiotemporal 
    processing stream that combines Graph Attention Networks (GATv2) for spatial 
    aggregation with GRU cells for temporal modeling.

    Architecture Flow (per timestep t):
        For each layer l in n_layers:
            1. Spatial Aggregation: GATv2 aggregates information from neighbors
            2. Temporal Gating: GRU updates hidden state using spatial features
            3. Normalization: LayerNorm for stable training
            4. Feed to Next Layer: Output becomes input to next spatial layer    
    
    Improvements wrt previous version:
        1. Layer-by-Layer Processing: True deep learning where each layer takes the 
           previous layer's output as input, enabling hierarchical spatial representations
           
           h^(l)_t = GRU(GAT^(l)(h^(l-1)_t, E), h^(l)_{t-1})
           
           where for l=0, h^(-1)_t denotes the encoded input features.
           
        2. GATv2 Attention Mechanism: Learns to weight neighbors dynamically using 
           multi-head attention, automatically focusing on meteorologically relevant 
           stations (e.g., upwind nodes during advection events):
           
           α_ij = softmax_j(a^T · LeakyReLU(W[h_i | h_j]))
           
        3. GRU Temporal Gates: Learnable gating mechanism for robust gradient flow:
           
           h_t = (1-z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
           
           The update gate z_t balances old vs. new memory, while the reset gate r_t 
           allows dropping irrelevant historical context.
    
    Args:
        input_size (int): Dimension of input features
        hidden_size (int): Hidden state dimension
        n_layers (int): Number of spatial layers (depth)
        dropout_p (float): Dropout probability
        mode (Literal['forwards', 'backwards']): Temporal processing direction
        num_heads (int): Number of attention heads in GATv2
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
        
        # ini states
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
                
                # Spatial Aggregation (GAT)
                x_spatial = self.gat_layers[l](current_input, batch_edge_index)
                x_spatial = self.dropout(x_spatial)
                
                # Temporal Gating (GRU)
                h_new = self.gru_cells[l](x_spatial, h_prev)
                
                # Norm e Skip
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
    Enhanced Bidirectional Spatiotemporal Graph Neural Network (STGNN1).
    
    This model processes weather forecasts by combining forward and backward temporal 
    processing with spatial graph attention, enabling the capture of both past and future 
    temporal context along with spatial dependencies between weather stations.
    
    Architecture Overview:
        1. Input Encoding: Linear projection + LayerNorm + SiLU activation
        2. Station Embeddings: Learnable embeddings added to capture station-specific biases
        3. Bidirectional Processing:
           - Forward stream: Processes temporal sequence from t=0 to t=T
           - Backward stream: Processes temporal sequence from t=T to t=0
        4. Skip Connection: Projects raw input to match bidirectional state dimension
        5. Hierarchical Readout: 3-layer MLP with normalization to combine information:
           - Layer 1: 2H·L → 2H (combine bidirectional multi-layer states)
           - Layer 2: 2H → H (compress to hidden dimension)
           - Layer 3: H → H (final refinement)
        6. Distribution Output: Maps to probabilistic distribution parameters (μ, σ)
    
    Info Flow:
        x → Encode → +StationEmb → [ForwardRNN, BackwardRNN]
          ↓                              ↓
        Skip ────────────────────────→ Concat → Readout → Output Distribution
    
    Key improvements over BiDirectionalSTGNN:
        1. Each EnhancedLayeredGraphRNN uses GATv2 instead of simple message passing
        2. GRU cells replace additive residuals for better gradient flow
        3. Hierarchical 3-layer readout (vs simple 2-layer) for better feature combination
        4. Enhanced normalization with LayerNorm throughout the architecture
    
    Args:
        input_size (int): Number of input features per timestep
        hidden_size (int): Hidden state dimension  
        n_stations (int): Number of weather stations
        output_dist (str): Output distribution type (e.g., 'LogNormal')
        n_layers (int): Number of spatial layers in each directional stream
        dropout_p (float): Dropout probability
        num_heads (int): Number of attention heads for GATv2
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
    """
    Enhanced Temporal-Convolutional Graph Neural Network (STGNN2).
    
    This model combines temporal convolutions (TCN) for multi-scale temporal feature 
    extraction with graph neural networks (GNN) for spatial aggregation, enhanced with 
    global attention and adaptive gating mechanisms.

    Architecture Flow:
        1. Encoding: Linear projection + station embeddings + horizon embeddings
        2. TCN-GNN Backbone (num_layers iterations):
           a. TCN Layer: Dilated causal convolutions with dilation = 2^l
           b. Spatial Refinement: GATv2 processes all timesteps simultaneously  
           c. Residual: Add GATv2 output to create refined spatial features
           d. Store Skip: Save for multi-scale combination
        3. Multi-Scale Combination: sum(skips) + final_layer
        4. Global Attention: Self-attention across all stations
        5. Adaptive Gating: Blend refined features with raw input
        6. Output: Probabilistic distribution parameters (μ, σ)    
    
    Key improvements over TCN_GNN prototype:
        1. Direction-Aware Spatial Aggregation: GATv2Conv with multi-head attention 
           learns to weight neighbors based on meteorological relevance. Unlike STGNN1's 
           per-timestep processing, GATv2 here refines the entire temporal sequence after 
           TCN processing, operating on all timesteps simultaneously.
           
        2. Global Spatial Attention: Multi-head self-attention layer (4 heads) allows 
           distant stations to exchange information directly:
           
           H_global = MultiHeadAttn(H_deep, H_deep, H_deep)
           
           This captures large-scale patterns like pressure gradients across Switzerland 
           without requiring multiple message-passing hops.
           
        3. Smart Ensemble Correction via GRU-Inspired Gating: Blends deep features with 
           raw predictions using learned gates:
           
           h_final = g ⊙ h_refined + (1-g) ⊙ h_raw
           where g = σ(W_g · h_refined)
           
           Unlike STGNN1's temporal GRU gates, this operates on feature space rather 
           than time, deciding how much to trust deep processing vs. raw predictions. 
           This prevents over-correction when ensemble forecasts are already well-calibrated.
           
        4. Forecast Horizon Embeddings: Learnable embeddings e_τ ∈ ℝ^H for each lead 
           time τ ∈ {1,...,96}, providing direct information about forecast horizon so 
           the model can learn how uncertainty should grow with time.
    
    Args:
        num_layers (int): Number of TCN-GNN layers
        input_size (int): Number of input features
        output_dist (str): Output distribution type (e.g., 'LogNormal')
        hidden_channels (Union[int, List[int]]): Hidden dimensions (per layer or constant)
        n_stations (int): Number of weather stations
        kernel_size (int): Temporal kernel size for TCN
        dropout_p (float): Dropout probability
        causal_conv (bool): Whether to use causal convolutions (typically True)
        gat_heads (int): Number of attention heads in GATv2
        max_lead_time (int): Maximum forecast lead time (for horizon embeddings)
    """
    def __init__(self, num_layers, input_size, output_dist, hidden_channels, 
                 n_stations, kernel_size=3, dropout_p=0.2, causal_conv=True, 
                 gat_heads=2, max_lead_time=96, **kwargs):
        super().__init__()
        
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]*num_layers
            
        self.hidden_dim = hidden_channels[0]
        
        # horizon Embeddings
        self.station_emb = NodeEmbedding(n_stations, self.hidden_dim)
        self.horizon_emb = nn.Embedding(max_lead_time + 1, self.hidden_dim) # 97
        
        self.encoder = nn.Linear(input_size, self.hidden_dim)

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

        h_deep = torch.stack(skips, dim=-1).sum(dim=-1) + h_final_tcn
        h_deep = rearrange(h_deep, '(b n) h t -> (b t) n h', b=B, n=N)

        attn_out, _ = self.global_spatial_attn(h_deep, h_deep, h_deep)
        h_refined = self.attn_norm(h_deep + attn_out)

        h_refined = rearrange(h_refined, '(b t) n h -> b t n h', b=B, t=T)
        
        gate = torch.sigmoid(self.gate_generator(h_refined))
        
        raw_proj = self.raw_skip_proj(x)
        h_final = gate * h_refined + (1 - gate) * raw_proj

        return self.output_distr(h_final)