"""One-dimensional conditional denoising network and IC feature extractors for DDPM.

Keep this module next to DDPM.py, which imports ContextUnet.
The default configuration uses 24 history weeks and trajectories padded to 128 steps.
"""

import torch
import torch.nn as nn
import math

class ResidualConvBlock(nn.Module):
    """One-dimensional convolutional block with an optional residual connection."""
    def __init__(self, in_channels: int, out_channels: int, is_res: bool = False) -> None:
        super().__init__()

        # Check whether the residual branch can be added without channel projection.
        self.same_channels = in_channels == out_channels

        self.is_res = is_res

        # Apply two blocks of one-dimensional convolution, batch normalization, and GELU.
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.is_res:

            x1 = self.conv1(x)

            x2 = self.conv2(x1)

            if self.same_channels:
                out = x + x2
            else:

                # Project the residual branch when the channel counts differ.
                shortcut = nn.Conv1d(x.shape[1], x2.shape[1], kernel_size=1, stride=1, padding=0).to(x.device)
                out = shortcut(x) + x2

            # Rescale the summed output.
            return out / 1.414

        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

    # Accessors for the output channel count.
    def get_out_channels(self):
        return self.conv2[0].out_channels

    def set_out_channels(self, out_channels):
        self.conv1[0].out_channels = out_channels
        self.conv2[0].in_channels = out_channels
        self.conv2[0].out_channels = out_channels

class UnetUp(nn.Module):
    """Concatenate a skip connection and upsample with a one-dimensional transposed convolution."""
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()

        layers = [
            nn.ConvTranspose1d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):

        x = torch.cat((x, skip), 1)

        x = self.model(x)
        return x

class UnetDown(nn.Module):
    """Extract residual features and downsample with a stride-2 convolution."""
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        self.conv_block = nn.Sequential(
            ResidualConvBlock(in_channels, out_channels, is_res=True),
            ResidualConvBlock(out_channels, out_channels, is_res=True)
        )

        # The default downsampling path reduces sequence length from 128 to 64, 32, and 16.
        self.downsample = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        x = self.conv_block(x)
        x = self.downsample(x)
        return x

class EmbedFC(nn.Module):
    """Map diffusion times or condition vectors to a specified embedding dimension."""
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        """Build an embedding with two fully connected layers and a GELU activation."""
        self.input_dim = input_dim

        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):

        x = x.reshape(-1, self.input_dim)

        return self.model(x)
    
class PositionalEncoding1D(nn.Module):
    """Generate sinusoidal positional encodings and cache results for matching shapes."""
    def __init__(self, channels):
        super().__init__()
        self.channels = 2 * math.ceil(channels / 2)

        self.target_channels = channels
        
        self.inv_freq = 1 / torch.pow(10000, torch.arange(0, self.channels, 2).float() / self.channels)
        self.cached_penc = None
        self.cached_shape = None

    def get_emb(self, sin_inp):
        emb = torch.stack((torch.sin(sin_inp), torch.cos(sin_inp)), dim=-1)
        emb = emb.view(*emb.shape[:-2], -1)
        return emb   
    
    def forward(self, x):
        if len(x.shape) != 3:
            raise RuntimeError("Input must be three-dimensional: (batch, channels, sequence_length).")
            
        batch_size, _, x_len = x.shape

        # Reuse positional encodings for matching shapes and move them to the input device.
        current_shape = (batch_size, self.target_channels, x_len)
        if self.cached_penc is not None and self.cached_shape == current_shape:
            return self.cached_penc.to(x.device)

        # Compute sine and cosine encodings from sequence positions.
        pos_x = torch.arange(x_len, device=x.device).float()
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq.to(x.device))
        emb = self.get_emb(sin_inp_x)

        if emb.shape[1] > self.target_channels:
            emb = emb[:, :self.target_channels]
        elif emb.shape[1] < self.target_channels:

            padding = torch.zeros(emb.shape[0], self.target_channels - emb.shape[1], device=x.device)
            emb = torch.cat([emb, padding], dim=1)

        # Arrange encodings as (batch, target_channels, sequence_length).
        cached_penc = emb.unsqueeze(0).repeat(batch_size, 1, 1).permute(0, 2, 1)

        self.cached_penc = cached_penc
        self.cached_shape = current_shape
        
        return cached_penc

class AttentionBlock(nn.Module):
    """Auxiliary attention block; not used in the current ContextUnet forward path."""
    def __init__(self, units, groups=8):
        super().__init__()
        self.units = units
        self.groups = groups

        self.norm = nn.GroupNorm(groups, units)
        self.query = nn.Linear(units, units)
        self.key = nn.Linear(units, units)
        self.value = nn.Linear(units, units)
        self.proj = nn.Linear(units, units)

    def forward(self, x):
        scale = float(self.units) ** -0.5
        
        x = nn.GroupNorm(self.groups, x.shape[1]).to(x.device)(x)

        q = self.query.to(x.device)(x)
        k = self.key.to(x.device)(x)
        v = self.value.to(x.device)(x)

        attn_score = torch.einsum("blc, bLc->blL", q, k) * scale
        attn_score = torch.softmax(attn_score, dim=-1)

        proj = torch.einsum("blL,bLc->blc", attn_score, v)
        proj = self.proj.to(x.device)(proj)
        return (x + proj)

class ICFeatureExtractor(nn.Module):
    """Jointly process average, maximum, and minimum IC channels into fixed-length features."""
    def __init__(self, seq_length=100, out_features=16):
        super().__init__()

        # Process three IC channels with two convolutions and global average pooling.
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),

            nn.Conv1d(32, out_features, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(out_features),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
    
    def forward(self, x):
        """Convert IC curves of shape (samples, 3, sequence_length) into fixed-length features."""
        return self.feature_extractor(x).squeeze(-1)

class ICProcessor(nn.Module):
    """Organize incremental-capacity features by vehicle batch and history week."""
    def __init__(self, num_time_steps=24, ic_features=16):
        super().__init__()

        self.num_time_steps = num_time_steps
        self.ic_features = ic_features

        self.ic_extractor = ICFeatureExtractor(seq_length=100, out_features=ic_features)
        
    def forward(self, ic_tensor):

        ic_features = self.ic_extractor(ic_tensor)

        # Restore the batch dimension using num_time_steps IC curves per vehicle.
        total_samples = ic_features.shape[0]
        batch_size = total_samples // self.num_time_steps

        ic_features = ic_features.reshape(batch_size, self.num_time_steps, self.ic_features)
        
        return ic_features

class ContextUnet(nn.Module):
    """One-dimensional conditional denoising network combining point and IC features.

    in_channels: Number of capacity input channels; 1 in the main script.
    n_feat: Base convolution channel count; 64 in the main script.
    n_cfeat: Point features per history step; 16 in the main script.
    h_cfeat: History weeks used for conditioning; 24 in the main script.
    height: Padded capacity trajectory length; 128 in the main script.
    ic_features: IC features per history week; 16 in the main script.
    """
    def __init__(self, in_channels, n_feat=64, n_cfeat=16, h_cfeat=24, height=128, ic_features=16):  
        super().__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h_cfeat = h_cfeat
        self.height = height
        self.ic_features = ic_features

        # Combined point and IC feature dimension per history step; 32 by default.
        self.fused_cfeat = n_cfeat + ic_features  
        
        if height % 8 != 0:
            raise ValueError(f"height ({height}) must be a multiple of 8")

        self.ic_processor = ICProcessor(num_time_steps=h_cfeat, ic_features=ic_features) 
        
        # Learn attention weights along the history dimension.
        self.context_fusion = nn.Sequential(
            nn.Linear(self.fused_cfeat, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        self.final_context_dim = self.fused_cfeat  

        # Concatenate capacity inputs and positional encodings, then extract initial features.
        self.init_conv = ResidualConvBlock(2 * in_channels, n_feat, is_res=True)

        # Three downsampling stages increase channel counts and reduce sequence length.
        self.down1 = UnetDown(n_feat, 2 * n_feat)
        self.down2 = UnetDown(2 * n_feat, 4 * n_feat)
        self.down3 = UnetDown(4 * n_feat, 8 * n_feat)

        # Pool the bottleneck features, reducing their default length from 16 to 8.
        self.to_vec = nn.Sequential(
            nn.AvgPool1d(2),
            nn.GELU()
        )

        # Construct diffusion-time embeddings for four decoder scales.
        self.timeembed1 = EmbedFC(1, 8 * n_feat)
        self.timeembed2 = EmbedFC(1, 4 * n_feat)  
        self.timeembed3 = EmbedFC(1, 2 * n_feat)  
        self.timeembed4 = EmbedFC(1, n_feat)

        # Construct condition embeddings for four decoder scales.
        self.contextembed1 = EmbedFC(self.final_context_dim, 8 * n_feat)
        self.contextembed2 = EmbedFC(self.final_context_dim, 4 * n_feat)
        self.contextembed3 = EmbedFC(self.final_context_dim, 2 * n_feat)
        self.contextembed4 = EmbedFC(self.final_context_dim, n_feat)

        self.posemb = PositionalEncoding1D(in_channels)

        # Store sequence-length metadata for each scale.
        self.seq_len_1 = height // 16
        self.seq_len_2 = height // 8
        self.seq_len_3 = height // 4
        self.seq_len_4 = height // 2

        # Restore sequence length in four stages, using skip connections in the last three.
        self.up0 = nn.Sequential(
            nn.ConvTranspose1d(8 * n_feat, 8 * n_feat, 2, stride=2, padding=0),
            nn.GroupNorm(8, 8 * n_feat),
            nn.GELU(),
        )
        
        self.up1 = UnetUp(16 * n_feat, 4 * n_feat)
        self.up2 = UnetUp(8 * n_feat, 2 * n_feat)
        self.up3 = UnetUp(4 * n_feat, n_feat)

        # Predict noise with the same shape as the capacity input.
        self.out = nn.Sequential(
            nn.Conv1d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.GELU(),
            nn.Conv1d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, t, c=None, c_points=None, ic_tensor=None, apply_context_mask=None):
        """Predict diffusion noise.

        x: Noisy capacity trajectories, shaped (batch, in_channels, sequence_length).
        t: Normalized diffusion times, shaped (batch,) or (batch, 1).
        c: Pre-fused history conditions, shaped (batch, 24, 32) by default.
        c_points: Separate point features, shaped (batch, 24, 16) by default.
        ic_tensor: Three-channel IC curves as a four-dimensional batch or flattened 3D tensor.
        apply_context_mask: Retained interface argument; the main script masks conditions before calling.
        """ 
        batch_size = x.shape[0]          
        # Accept pre-fused conditions or separate point features and IC curves.
        if c is not None:
            fused_features = c
        elif c_points is not None:
            if ic_tensor is not None:

                if ic_tensor.dim() == 4:
                    ic_tensor = ic_tensor.view(-1, 3, 100)

                ic_features = self.ic_processor(ic_tensor)

                fused_features = torch.cat([c_points, ic_features], dim=-1)            
            else:

                zero_padding = torch.zeros(batch_size, c_points.shape[1], self.ic_features, 
                                         device=c_points.device, dtype=c_points.dtype)
                fused_features = torch.cat([c_points, zero_padding], dim=-1)
        else:
            raise ValueError("Either c or c_points must be provided.")

        # Aggregate history features into a condition vector using learned attention weights.
        attention_weights = self.context_fusion(fused_features)
        context_vector = (fused_features * attention_weights).sum(dim=1)

        # Capacity inputs have shape (batch, 1, 128) in the default configuration.
        posemb = self.posemb(x).to(x)
        x = torch.cat([x, posemb], dim=1)

        x = self.init_conv(x)

        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)

        hiddenvec = self.to_vec(down3)

        temb1 = self.timeembed1(t).view(-1, 8 * self.n_feat, 1)
        temb2 = self.timeembed2(t).view(-1, 4 * self.n_feat, 1)
        temb3 = self.timeembed3(t).view(-1, 2 * self.n_feat, 1)
        temb4 = self.timeembed4(t).view(-1, self.n_feat, 1)

        cemb1_base = self.contextembed1(context_vector).view(-1, 8 * self.n_feat, 1)
        cemb2_base = self.contextembed2(context_vector).view(-1, 4 * self.n_feat, 1)
        cemb3_base = self.contextembed3(context_vector).view(-1, 2 * self.n_feat, 1)
        cemb4_base = self.contextembed4(context_vector).view(-1, self.n_feat, 1)

        up0 = self.up0(hiddenvec)

        # Broadcast time and condition embeddings to modulate decoder features at each scale.
        cemb1 = cemb1_base.expand(-1, -1, up0.shape[2])
        temb1_expanded = temb1.expand(-1, -1, up0.shape[2])

        up1 = self.up1(cemb1 * up0 + temb1_expanded, down3)

        cemb2 = cemb2_base.expand(-1, -1, up1.shape[2])
        temb2_expanded = temb2.expand(-1, -1, up1.shape[2])

        up2 = self.up2(cemb2 * up1 + temb2_expanded, down2)

        cemb3 = cemb3_base.expand(-1, -1, up2.shape[2])
        temb3_expanded = temb3.expand(-1, -1, up2.shape[2])

        up3 = self.up3(cemb3 * up2 + temb3_expanded, down1)

        cemb4 = cemb4_base.expand(-1, -1, up3.shape[2])
        temb4_expanded = temb4.expand(-1, -1, up3.shape[2])

        # Combine shallow features with decoder outputs to produce the final noise estimate.
        up3_modulated = cemb4 * up3 + temb4_expanded
        out = self.out(torch.cat((up3_modulated, x), dim=1))
        
        return out
