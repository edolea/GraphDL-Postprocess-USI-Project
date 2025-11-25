from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn as nn
import torch
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
        super().__init__(**kwargs)
        
        
        
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


class Model0TCN(nn.Module):
    """
    TCN-only baseline model (Model0) for statistical postprocessing.

    Architecture:
    - No graph operations: Processes each station's time series independently
    - Temporal modeling only: Uses stacked TCN layers with causal convolutions
    - Station-independent: No message passing or spatial aggregation
    - Probabilistic output: Outputs LogNormal distribution parameters

    The model processes input shape [batch, time, stations, features] by:
    1. Reshaping to [batch*stations, features, time] for Conv1d processing
    2. Applying input encoder to project features to hidden dimension
    3. Processing through stacked TCN layers with increasing dilation rates
    4. Reshaping back to [batch, time, stations, hidden]
    5. Applying distribution layer to get probabilistic outputs

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

        # Input encoder: project features to hidden dimension
        self.encoder = nn.Linear(input_size, hidden_channels)

        # Rearrange layers for shape transformations
        # From [batch, time, stations, features] to [batch*stations, features, time]
        self.rearrange_for_tcn = Rearrange('b t n c -> (b n) c t')
        # From [batch*stations, channels, time] back to [batch, time, stations, channels]
        self.rearrange_from_tcn = Rearrange('(b n) c t -> b t n c', n=n_stations)

        # Build TCN layers with exponentially increasing dilation rates
        tcn_layers = []
        for layer_idx in range(num_layers):
            dilation = 2 ** layer_idx  # 1, 2, 4, 8, ...
            in_channels = hidden_channels  # All layers use same hidden dimension

            tcn_layers.append(
                TCNLayer(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout_p=dropout_p,
                    causal_conv=True  # Ensure no future information leakage
                )
            )

        self.tcn_layers = nn.ModuleList(tcn_layers)

        # Probabilistic output layer
        self.output_distr = dist_to_layer[output_dist](input_size=hidden_channels)

    def forward(self, x, edge_index=None):
        """
        Forward pass through the TCN baseline.

        Args:
            x: Input tensor of shape [batch, time, stations, features]
            edge_index: Graph edge indices (accepted but ignored for compatibility)

        Returns:
            Distribution object with parameters of shape [batch, time, stations, output_dim]
        """
        # Store input shape for reshaping
        batch_size, time_steps, num_stations, num_features = x.size()

        # Encode input features to hidden dimension
        # Shape: [batch, time, stations, features] -> [batch, time, stations, hidden]
        x = self.encoder(x)

        # Reshape for Conv1d: [batch, time, stations, hidden] -> [batch*stations, hidden, time]
        x = self.rearrange_for_tcn(x)

        # Apply TCN layers with skip connections
        skips = []
        for tcn_layer in self.tcn_layers:
            # TCNLayer returns (residual_output, skip_output)
            x, skip = tcn_layer(x)
            skips.append(skip)

        # Aggregate skip connections and final output
        # Stack and sum all skip connections
        skips_stack = torch.stack(skips, dim=-1)  # [batch*stations, hidden, time, num_layers]
        skips_sum = skips_stack.sum(dim=-1)  # [batch*stations, hidden, time]

        # Combine skip connections with final layer output
        output = skips_sum + x  # [batch*stations, hidden, time]

        # Reshape back to original structure
        # [batch*stations, hidden, time] -> [batch, time, stations, hidden]
        output = self.rearrange_from_tcn(output)

        # Apply distribution layer to get probabilistic outputs
        return self.output_distr(output)
   