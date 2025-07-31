import torch
import torch.nn as nn
import mne
import numpy as np
from torch.nn.modules import activation


class MHA(nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.1):
        super().__init__()
        # Note: batch_first=True so that the input shape remains (batch, seq, d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
    
    def forward(self, x):
        # Self-attention: key, query, and value are all x
        attn_output, attn_weights = self.mha(x, x, x)
        return attn_output, attn_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        # Note: typically no activation after the second linear layer
        x = self.linear2(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # Layer normalization before each sub-layer (pre-norm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mha = MHA(n_heads, d_model, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        # Self-attention sub-layer with residual connection
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.mha(x_norm)
        x = x + self.dropout1(attn_out)
        
        
        # Feed-forward sub-layer with residual connection
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        x = x + self.dropout2(ff_out)
        
        return x, attn_weights

# Optionally, stack multiple Transformer encoder blocks:
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        attn_weights_list = []
        for layer in self.layers:
            x, attn = layer(x)
            attn_weights_list.append(attn)
        return x, attn_weights_list

class ChannelCoordinateEmbedderSimple(nn.Module):
    def __init__(self, channels, embed_dim=200):
        super().__init__()
        montage = mne.channels.make_standard_montage("standard_1020")
        ch_pos = montage.get_positions()["ch_pos"]
        coords = [ch_pos[ch] for ch in channels if ch in ch_pos]
        coords_array = np.array(coords)
        self.register_buffer("coords", torch.tensor(coords_array, dtype=torch.float32))
        self.linear = nn.Linear(3, embed_dim, bias=True) #false
        for param in self.linear.parameters():
            param.requires_grad = True #false

    def forward(self, x):
        embeddings = self.linear(self.coords)
        embeddings = torch.nn.functional.tanh(embeddings)
        return x + embeddings

class spatial_transformer_encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, n_layers = 1, channel_names=None, spatial_embedder_value=True):
        super().__init__()
        self.channels = channel_names
        self.encoder = TransformerEncoder(num_layers=n_layers, d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
        self.spatial_embedder = ChannelCoordinateEmbedderSimple(channels=self.channels, embed_dim=d_model)
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(31 * d_model, d_model),
                                        nn.GELU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(d_model, 1)
                                    )
        self.spatial_embedder_value = spatial_embedder_value
    def forward(self, x):
        if self.spatial_embedder_value == True:
            x = self.spatial_embedder(x)
        x_mmd, attn_weights_list = self.encoder(x)
        x_out = self.classifier(x)
        return x_out, x_mmd, attn_weights_list


from mamba_ssm import Mamba
# class spatial_mamba(nn.Module):
#     def __init__(self, d_model, dropout=0.1, n_layers=1, channel_names=None):
#         super().__init__()
#         self.channels = channel_names
#         self.encoder = nn.ModuleList([Mamba(d_model=d_model, expand=2, d_state=64, d_conv = 2) for _ in range(n_layers)])
#         self.spatial_embedder = ChannelCoordinateEmbedderSimple(channels=self.channels, embed_dim=d_model)
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(31 * d_model, d_model),
#             nn.SiLU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_model, 1)
#         )

#     def forward(self, x):
#         x = self.spatial_embedder(x)
#         for layer in self.encoder:
#             x = layer(x)
#         x_out = self.classifier(x)
#         return x_out, x
    
import torch.nn as nn
from mamba_ssm import Mamba

class spatial_mamba(nn.Module):
    def __init__(self, d_model, dropout=0.1, n_layers=1, channel_names=None):
        super().__init__()
        self.channels = channel_names
        self.dropout = dropout
        # Spatial coordinate embedding
        self.spatial_embedder = ChannelCoordinateEmbedderSimple(
            channels=self.channels,
            embed_dim=d_model
        )
        # Encoder: Mamba block + pre-norm + post-dropout + FFN + post-dropout
        self.encoder = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(d_model),
                'mamba': Mamba(d_model=d_model, expand=2, d_state=16, d_conv=5),
                'drop1': nn.Dropout(dropout),
                'norm2': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, 2 * d_model),
                    nn.SiLU(),
                    nn.Linear(2 * d_model, d_model)
                ),
                'drop2': nn.Dropout(dropout)
            }) for _ in range(n_layers)]
        )
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(31 * d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        # Add spatial embeddings
        # x = self.spatial_embedder(x)
        # Process through Mamba + FFN blocks with residuals, layer norm, and dropout
        for block in self.encoder:
            # Mamba sub-layer
            res = x
            x = block['norm1'](x)
            x = block['mamba'](x)
            x = block['drop1'](x)
            x = res + x
            # Feed-forward sub-layer
            res = x
            x = block['norm2'](x)
            x = block['ffn'](x)
            x = block['drop2'](x)
            x = res + x
        # Classification output
        x_out = self.classifier(x)
        return x_out, x





# Example usage:
# data = torch.randn(32, 31, 5)  # Example input shape (batch_size, num_channels, seq_length)
# epochs   = mne.read_epochs("./1-epochs-epo.fif", verbose=False)
# channels = epochs.ch_names
# mamba_model = spatial_mamba(d_model=5, n_layers=2, channel_names=channels)
# output, mmd = mamba_model(data)
# print(f"Output shape: {output.shape}")
# print(f"Output shape: {mmd.shape}")

# Example usage:
# data = torch.randn(5, 31, 5)

# model = CNN_model(conv_out_1=32, conv_out_2 = 64, num_classes=1)
# output = model(data)
# print(output.shape)

# epochs = mne.read_epochs("asmr_epochs/1-epochs-epo.fif", verbose=0)
# x = torch.randn(32, 30, 2000)
# channels = epochs.ch_names
# spatial_transformer_encoder_model = spatial_transformer_encoder(d_model=2000, n_heads=4, d_ff=2000*2, dropout=0.1, n_layers=5, channel_names=channels)
# x_out, attn_weights = spatial_transformer_encoder_model(x)
# print(f"Embeddings shape: {x_out.shape}")

# from torchinfo import summary
# print(summary(spatial_transformer_encoder_model, input_size=(32, 30, 2000), col_names=["input_size", "output_size", "num_params", "trainable"]))