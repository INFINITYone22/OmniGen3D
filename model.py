"""
OmniGen3D: Unified Multi-Modal Architecture for Physics-Aware Text-to-3D Generation
Author: Rohith Garapati

This implementation contains the complete OmniGen3D model as described in the paper:
"OmniGen3D: A Unified Multi-Modal Architecture for Physics-Aware Text-to-3D Generation"

Total Parameters: 31.2 Billion
Architecture: 7-stage unified pipeline for multi-modal text-to-3D generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Union
from .config import OmniGen3DConfig

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.mean(x * x, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm + self.eps)
        return self.weight * x_normed

class SwiGLU(nn.Module):
    """SwiGLU Activation Function"""
    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int):
        super().__init__()
        self.w1 = nn.Linear(dim_in, dim_hidden, bias=False)
        self.w2 = nn.Linear(dim_hidden, dim_out, bias=False)
        self.w3 = nn.Linear(dim_in, dim_hidden, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding"""
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos_emb = emb.cos()[None, :, None, :]
        sin_emb = emb.sin()[None, :, None, :]
        
        return cos_emb, sin_emb

def apply_rotary_pos_emb(x, cos, sin):
    """Apply rotary position embedding"""
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat([
        x1 * cos - x2 * sin,
        x2 * cos + x1 * sin
    ], dim=-1)

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with Flash Attention support"""
    def __init__(self, config: OmniGen3DConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.transformer.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash = config.transformer.use_flash_attention

        assert self.embed_dim % self.num_heads == 0

        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=False)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.dropout = nn.Dropout(config.transformer.dropout)
        
        self.rotary_emb = RotaryPositionalEmbedding(
            self.head_dim, config.transformer.max_seq_length
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply rotary position embedding
        cos, sin = self.rotary_emb(x, N)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        
        if self.use_flash and hasattr(F, 'scaled_dot_product_attention'):
            # Use Flash Attention if available
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0)
        else:
            # Standard attention
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)
            x = attn @ v
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer Block with Pre-Norm and SwiGLU"""
    def __init__(self, config: OmniGen3DConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        
        # Normalization
        if config.transformer.norm_type == "rmsnorm":
            self.norm1 = RMSNorm(self.embed_dim)
            self.norm2 = RMSNorm(self.embed_dim)
        else:
            self.norm1 = nn.LayerNorm(self.embed_dim)
            self.norm2 = nn.LayerNorm(self.embed_dim)
        
        # Attention
        self.attn = MultiHeadAttention(config)
        
        # Feed Forward
        if config.transformer.activation == "swiglu":
            hidden_dim = int(self.embed_dim * config.transformer.mlp_ratio)
            self.ffn = SwiGLU(self.embed_dim, hidden_dim, self.embed_dim)
        else:
            hidden_dim = int(self.embed_dim * config.transformer.mlp_ratio)
            self.ffn = nn.Sequential(
                nn.Linear(self.embed_dim, hidden_dim),
                nn.GELU() if config.transformer.activation == "gelu" else nn.ReLU(),
                nn.Dropout(config.transformer.dropout),
                nn.Linear(hidden_dim, self.embed_dim),
                nn.Dropout(config.transformer.dropout)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class TextEmbeddingMLP(nn.Module):
    """Stage 1: Text Embedding MLP (67M parameters)"""
    def __init__(self, config: OmniGen3DConfig):
        super().__init__()
        cfg = config.text_embedding
        
        self.projection = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.ReLU() if cfg.activation == "relu" else nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.output_dim),
        )
        
        if cfg.use_layer_norm:
            self.projection.append(nn.LayerNorm(cfg.output_dim))
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)

class BackboneTransformer(nn.Module):
    """Stage 2: Backbone Transformer (6.44B parameters)"""
    def __init__(self, config: OmniGen3DConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.num_layers = config.transformer.num_layers
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, config.transformer.max_seq_length, self.embed_dim) * 0.02)
        
        # Transformer layers
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(self.num_layers)
        ])
        
        # Final norm
        if config.transformer.norm_type == "rmsnorm":
            self.norm = RMSNorm(self.embed_dim)
        else:
            self.norm = nn.LayerNorm(self.embed_dim)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # Add positional embeddings
        if N <= self.pos_embed.shape[1]:
            pos_embed = self.pos_embed[:, :N, :]
        else:
            # Interpolate if sequence is longer than trained length
            pos_embed = F.interpolate(
                self.pos_embed.permute(0, 2, 1), 
                size=N, 
                mode='linear', 
                align_corners=False
            ).permute(0, 2, 1)
        
        x = x + pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return x

class MultiModalEncoder(nn.Module):
    """Stage 3: Multi-Modal Input Processing (1.94B parameters)"""
    def __init__(self, config: OmniGen3DConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        cfg = config.multi_modal
        
        # Image branch - Vision Transformer
        self.image_patch_embed = nn.Conv2d(3, self.embed_dim, 
                                          kernel_size=cfg.image_patch_size, 
                                          stride=cfg.image_patch_size)
        self.image_pos_embed = nn.Parameter(torch.randn(1, 256, self.embed_dim) * 0.02)
        self.image_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(cfg.image_num_layers)
        ])
        
        # Sketch branch - Stroke sequence encoder
        self.sketch_embed = nn.Linear(cfg.sketch_stroke_dim, self.embed_dim)
        self.sketch_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(cfg.sketch_num_layers)
        ])
        
        # Depth branch - 3D CNN encoder
        depth_layers = []
        in_channels = 1
        for out_channels in cfg.depth_conv_layers:
            depth_layers.extend([
                nn.Conv3d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv3d(out_channels, out_channels, 3, stride=2, padding=1),
            ])
            in_channels = out_channels
        depth_layers.append(nn.AdaptiveAvgPool3d(1))
        self.depth_encoder = nn.Sequential(*depth_layers)
        self.depth_proj = nn.Linear(cfg.depth_conv_layers[-1], self.embed_dim)
        
        # Cross-modal fusion
        self.cross_attention = nn.ModuleList([
            TransformerBlock(config) for _ in range(cfg.cross_attn_layers)
        ])
        
        # Final projection
        self.final_proj = nn.Linear(self.embed_dim * 4, self.embed_dim)  # Text+Image+Sketch+Depth

    def forward(self, 
                text_features: torch.Tensor,
                image: Optional[torch.Tensor] = None,
                sketch: Optional[torch.Tensor] = None,
                depth: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        batch_size = text_features.shape[0]
        features = [text_features]
        
        # Process image
        if image is not None:
            B, C, H, W = image.shape
            img_patches = self.image_patch_embed(image)  # B, embed_dim, H', W'
            img_patches = img_patches.flatten(2).transpose(1, 2)  # B, N, embed_dim
            img_patches = img_patches + self.image_pos_embed[:, :img_patches.shape[1]]
            
            for block in self.image_blocks:
                img_patches = block(img_patches)
            
            img_global = img_patches.mean(dim=1)  # Global average pooling
            features.append(img_global)
        else:
            features.append(torch.zeros(batch_size, self.embed_dim, device=text_features.device))
        
        # Process sketch
        if sketch is not None:
            sketch_embed = self.sketch_embed(sketch)
            
            for block in self.sketch_blocks:
                sketch_embed = block(sketch_embed)
            
            sketch_global = sketch_embed.mean(dim=1)
            features.append(sketch_global)
        else:
            features.append(torch.zeros(batch_size, self.embed_dim, device=text_features.device))
        
        # Process depth
        if depth is not None:
            depth_feat = self.depth_encoder(depth)
            depth_feat = depth_feat.squeeze(-1).squeeze(-1).squeeze(-1)
            depth_feat = self.depth_proj(depth_feat)
            features.append(depth_feat)
        else:
            features.append(torch.zeros(batch_size, self.embed_dim, device=text_features.device))
        
        # Concatenate and project
        multi_modal_features = torch.cat(features, dim=-1)
        fused_features = self.final_proj(multi_modal_features)
        
        # Cross-modal attention
        fused_features = fused_features.unsqueeze(1)  # Add sequence dimension
        for block in self.cross_attention:
            fused_features = block(fused_features)
        
        return fused_features.squeeze(1)  # Remove sequence dimension

class HashGridEncoder(nn.Module):
    """Stage 4: 3D Spatial Encoding (22.1B parameters)"""
    def __init__(self, config: OmniGen3DConfig):
        super().__init__()
        cfg = config.hash_grid
        self.num_levels = cfg.num_levels
        self.base_resolution = cfg.base_resolution
        self.max_resolution = cfg.max_resolution
        
        # Hash tables for different resolution levels
        self.hash_tables = nn.ModuleList()
        total_features = 0
        
        for level in range(self.num_levels):
            resolution = int(self.base_resolution * (
                self.max_resolution / self.base_resolution
            ) ** (level / (self.num_levels - 1)))
            
            features_per_level = cfg.features_per_level[level]
            hash_size = min(resolution ** 3, cfg.hash_table_sizes[level])
            
            # Use embedding table as simplified hash grid
            hash_table = nn.Embedding(hash_size, features_per_level)
            self.hash_tables.append(hash_table)
            total_features += features_per_level
        
        # Feature aggregation network
        layers = []
        dims = [total_features] + cfg.aggregation_dims
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU() if i < len(dims) - 2 else nn.Identity()
            ])
        self.aggregation = nn.Sequential(*layers)
        
        self._init_weights()

    def _init_weights(self):
        for hash_table in self.hash_tables:
            nn.init.uniform_(hash_table.weight, -1e-4, 1e-4)
        
        for m in self.aggregation.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def hash_encode(self, coords: torch.Tensor, level: int) -> torch.Tensor:
        """Simplified hash encoding for 3D coordinates"""
        # Normalize coordinates to [0, 1]
        coords = (coords + 1) / 2  # Assume input is in [-1, 1]
        
        resolution = int(self.base_resolution * (
            self.max_resolution / self.base_resolution
        ) ** (level / (self.num_levels - 1)))
        
        # Discretize coordinates
        coords_discrete = torch.floor(coords * resolution).long()
        coords_discrete = torch.clamp(coords_discrete, 0, resolution - 1)
        
        # Simple hash function (can be replaced with proper spatial hash)
        hash_indices = (
            coords_discrete[..., 0] * resolution * resolution +
            coords_discrete[..., 1] * resolution +
            coords_discrete[..., 2]
        ) % self.hash_tables[level].num_embeddings
        
        return self.hash_tables[level](hash_indices)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: 3D coordinates of shape (..., 3)
        Returns:
            Multi-resolution features of shape (..., output_dim)
        """
        batch_shape = coords.shape[:-1]
        coords_flat = coords.reshape(-1, 3)
        
        # Extract features from all levels
        level_features = []
        for level in range(self.num_levels):
            feat = self.hash_encode(coords_flat, level)
            level_features.append(feat)
        
        # Concatenate features from all levels
        multi_res_features = torch.cat(level_features, dim=-1)
        
        # Aggregate features
        output_features = self.aggregation(multi_res_features)
        
        # Reshape back to original batch shape
        output_shape = batch_shape + (output_features.shape[-1],)
        return output_features.reshape(output_shape)

class HybridRenderer(nn.Module):
    """Stage 5: Gaussian-SDF Hybrid Renderer (128M parameters)"""
    def __init__(self, config: OmniGen3DConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        cfg = config.hybrid_renderer
        
        # Gaussian predictor network
        gaussian_layers = []
        dims = [self.embed_dim] + cfg.gaussian_hidden_dims
        for i in range(len(dims) - 1):
            gaussian_layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU()
            ])
        self.gaussian_backbone = nn.Sequential(*gaussian_layers)
        
        # Gaussian property heads
        final_dim = cfg.gaussian_hidden_dims[-1]
        self.position_head = nn.Linear(final_dim, 3)      # Position
        self.scale_head = nn.Linear(final_dim, 3)         # Scale
        self.rotation_head = nn.Linear(final_dim, 4)      # Quaternion
        self.opacity_head = nn.Linear(final_dim, 1)       # Opacity
        self.color_head = nn.Linear(final_dim, 3)         # RGB
        self.material_head = nn.Linear(final_dim, 8)      # Material properties
        
        # SDF refinement network
        sdf_layers = []
        for i in range(cfg.sdf_num_layers):
            if i == 0:
                sdf_layers.append(nn.Linear(self.embed_dim + 60, cfg.sdf_hidden_dim))  # +60 for positional encoding
            elif i in cfg.sdf_skip_connections:
                sdf_layers.append(nn.Linear(cfg.sdf_hidden_dim + self.embed_dim + 60, cfg.sdf_hidden_dim))
            else:
                sdf_layers.append(nn.Linear(cfg.sdf_hidden_dim, cfg.sdf_hidden_dim))
            
            if i < cfg.sdf_num_layers - 1:
                sdf_layers.append(nn.ReLU())
        
        sdf_layers.append(nn.Linear(cfg.sdf_hidden_dim, 1))  # SDF output
        self.sdf_network = nn.ModuleList(sdf_layers)
        
        # Complexity estimator for adaptive selection
        self.complexity_estimator = nn.Linear(self.embed_dim, 1)
        self.complexity_threshold = cfg.complexity_threshold

    def positional_encoding(self, coords: torch.Tensor, num_freqs: int = 10) -> torch.Tensor:
        """Positional encoding for SDF network input"""
        encoded = [coords]
        for freq in range(num_freqs):
            for func in [torch.sin, torch.cos]:
                encoded.append(func(2 ** freq * math.pi * coords))
        return torch.cat(encoded, dim=-1)

    def forward(self, features: torch.Tensor, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: Fused features from previous stages (..., embed_dim)
            coords: 3D coordinates (..., 3)
        Returns:
            Dictionary containing Gaussian parameters and SDF values
        """
        batch_shape = features.shape[:-1]
        features_flat = features.reshape(-1, self.embed_dim)
        coords_flat = coords.reshape(-1, 3)
        
        # Estimate complexity for adaptive rendering
        complexity = torch.sigmoid(self.complexity_estimator(features_flat))
        use_sdf = complexity > self.complexity_threshold
        
        # Gaussian splatting prediction
        gaussian_feat = self.gaussian_backbone(features_flat)
        
        gaussians = {
            'position': self.position_head(gaussian_feat),
            'scale': torch.exp(self.scale_head(gaussian_feat)),  # Ensure positive
            'rotation': F.normalize(self.rotation_head(gaussian_feat), dim=-1),  # Unit quaternion
            'opacity': torch.sigmoid(self.opacity_head(gaussian_feat)),
            'color': torch.sigmoid(self.color_head(gaussian_feat)),
            'material': torch.sigmoid(self.material_head(gaussian_feat))
        }
        
        # SDF prediction for complex regions
        sdf_values = torch.zeros(coords_flat.shape[0], 1, device=features.device)
        
        if use_sdf.any():
            sdf_coords = coords_flat[use_sdf.squeeze()]
            sdf_features = features_flat[use_sdf.squeeze()]
            
            if len(sdf_coords) > 0:
                # Positional encoding for coordinates
                coords_encoded = self.positional_encoding(sdf_coords)
                sdf_input = torch.cat([sdf_features, coords_encoded], dim=-1)
                
                # SDF network forward pass
                x = sdf_input
                skip_input = sdf_input
                
                layer_idx = 0
                for module in self.sdf_network:
                    if isinstance(module, nn.Linear):
                        if layer_idx in [0]:  # First layer
                            x = module(x)
                        elif layer_idx in [2, 4, 6]:  # Skip connections
                            x = module(torch.cat([x, skip_input], dim=-1))
                        else:
                            x = module(x)
                        layer_idx += 1
                    else:  # Activation
                        x = module(x)
                
                sdf_values[use_sdf.squeeze()] = x
        
        # Reshape outputs
        output = {}
        for key, value in gaussians.items():
            output[f'gaussian_{key}'] = value.reshape(batch_shape + value.shape[1:])
        
        output['sdf'] = sdf_values.reshape(batch_shape + (1,))
        output['complexity'] = complexity.reshape(batch_shape + (1,))
        output['use_sdf'] = use_sdf.reshape(batch_shape + (1,))
        
        return output

class PhysicsMaterialIntegration(nn.Module):
    """Stage 6: Physics-Material Integration (61K parameters)"""
    def __init__(self, config: OmniGen3DConfig):
        super().__init__()
        cfg = config.physics
        
        # Material property heads
        self.material_heads = nn.ModuleDict()
        for prop in cfg.material_properties:
            self.material_heads[prop] = nn.Linear(cfg.physics_heads_dim, 1)
        
        # Physics constraint heads
        self.constraint_heads = nn.ModuleDict()
        for constraint in cfg.constraint_checks:
            if constraint == "stability":
                self.constraint_heads[constraint] = nn.Linear(cfg.physics_heads_dim, 3)
            elif constraint == "collision_boundary":
                self.constraint_heads[constraint] = nn.Linear(cfg.physics_heads_dim, 6)  # bbox
            else:
                self.constraint_heads[constraint] = nn.Linear(cfg.physics_heads_dim, 1)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: Global features (..., embed_dim)
        Returns:
            Dictionary containing physics and material properties
        """
        outputs = {}
        
        # Material properties
        for prop, head in self.material_heads.items():
            outputs[f'material_{prop}'] = torch.sigmoid(head(features))
        
        # Physics constraints
        for constraint, head in self.constraint_heads.items():
            if constraint in ["gravity", "stability"]:
                outputs[f'physics_{constraint}'] = head(features)
            else:
                outputs[f'physics_{constraint}'] = torch.sigmoid(head(features))
        
        return outputs

class SceneCompositionNetwork(nn.Module):
    """Stage 7: Hierarchical Scene Composition (603M parameters)"""
    def __init__(self, config: OmniGen3DConfig):
        super().__init__()
        cfg = config.scene_composition
        self.embed_dim = config.embed_dim
        self.max_objects = cfg.max_objects
        
        # Object detection
        self.object_detector = nn.Linear(self.embed_dim, cfg.max_objects)
        
        # Relationship encoder
        self.relationship_encoder = nn.ModuleList([
            TransformerBlock(config) for _ in range(cfg.relationship_encoder_layers)
        ])
        
        # Spatial layout predictor
        self.spatial_layout = nn.Linear(self.embed_dim, cfg.spatial_transform_dim)
        
        # Object composition cross-attention
        self.composition_attention = nn.ModuleList([
            TransformerBlock(config) for _ in range(cfg.cross_attention_layers)
        ])

    def forward(self, global_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            global_features: Scene-level features (batch_size, embed_dim)
        Returns:
            Dictionary containing scene composition information
        """
        batch_size = global_features.shape[0]
        
        # Object detection - predict presence of objects
        object_presence = torch.sigmoid(self.object_detector(global_features))
        
        # Create object tokens
        object_tokens = global_features.unsqueeze(1).expand(-1, self.max_objects, -1)
        
        # Relationship encoding
        for block in self.relationship_encoder:
            object_tokens = block(object_tokens)
        
        # Spatial layout for each object
        spatial_transforms = self.spatial_layout(object_tokens)  # (B, max_objects, 9)
        
        # Object composition attention
        for block in self.composition_attention:
            object_tokens = block(object_tokens)
        
        return {
            'object_presence': object_presence,
            'object_features': object_tokens,
            'spatial_transforms': spatial_transforms
        }

class OmniGen3D(nn.Module):
    """
    Complete OmniGen3D Model - 31.2 Billion Parameters
    
    A Unified Multi-Modal Architecture for Physics-Aware Text-to-3D Generation
    """
    
    def __init__(self, config: OmniGen3DConfig):
        super().__init__()
        self.config = config
        
        print(f"Initializing OmniGen3D with {config.get_total_parameters():,} parameters...")
        
        # Stage 1: Text Embedding MLP (67M params)
        self.text_embedding = TextEmbeddingMLP(config)
        
        # Stage 2: Backbone Transformer (6.44B params)
        self.backbone_transformer = BackboneTransformer(config)
        
        # Stage 3: Multi-Modal Processing (1.94B params)
        self.multi_modal_encoder = MultiModalEncoder(config)
        
        # Stage 4: 3D Spatial Encoding (22.1B params)
        self.hash_grid_encoder = HashGridEncoder(config)
        
        # Stage 5: Gaussian-SDF Hybrid Renderer (128M params)
        self.hybrid_renderer = HybridRenderer(config)
        
        # Stage 6: Physics-Material Integration (61K params)
        self.physics_material = PhysicsMaterialIntegration(config)
        
        # Stage 7: Hierarchical Scene Composition (603M params)
        self.scene_composition = SceneCompositionNetwork(config)
        
        self._initialize_weights()
        
        print("OmniGen3D initialization complete!")
        config.print_config()

    def _initialize_weights(self):
        """Initialize model weights"""
        def _init_module(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, (nn.LayerNorm, RMSNorm)):
                if hasattr(module, 'weight'):
                    nn.init.constant_(module.weight, 1.0)
                if hasattr(module, 'bias'):
                    nn.init.constant_(module.bias, 0.0)
        
        self.apply(_init_module)

    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self,
                text_features: torch.Tensor,
                coordinates: torch.Tensor,
                image: Optional[torch.Tensor] = None,
                sketch: Optional[torch.Tensor] = None,
                depth: Optional[torch.Tensor] = None,
                return_intermediates: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through OmniGen3D
        
        Args:
            text_features: Text embeddings (B, text_dim)
            coordinates: 3D coordinates for volume rendering (B, N, 3)
            image: Optional reference images (B, C, H, W)
            sketch: Optional sketch strokes (B, S, 5)
            depth: Optional depth maps (B, D, H, W)
            return_intermediates: Whether to return intermediate features
            
        Returns:
            Dictionary containing all model outputs
        """
        intermediates = {} if return_intermediates else None
        
        # Stage 1: Text embedding
        text_embedded = self.text_embedding(text_features)
        if return_intermediates:
            intermediates['text_embedded'] = text_embedded
        
        # Stage 2: Backbone transformer processing
        # Add sequence dimension for transformer
        backbone_input = text_embedded.unsqueeze(1)  # (B, 1, embed_dim)
        global_features = self.backbone_transformer(backbone_input).squeeze(1)
        if return_intermediates:
            intermediates['global_features'] = global_features
        
        # Stage 3: Multi-modal processing
        multi_modal_features = self.multi_modal_encoder(
            text_embedded, image, sketch, depth
        )
        if return_intermediates:
            intermediates['multi_modal_features'] = multi_modal_features
        
        # Fuse global and multi-modal features
        fused_global = global_features + multi_modal_features
        
        # Stage 4: 3D spatial encoding
        spatial_features = self.hash_grid_encoder(coordinates)  # (B, N, embed_dim)
        
        # Broadcast global features to match spatial dimensions
        batch_size, num_points = coordinates.shape[:2]
        global_broadcasted = fused_global.unsqueeze(1).expand(-1, num_points, -1)
        
        # Combine spatial and global features
        combined_features = spatial_features + global_broadcasted
        if return_intermediates:
            intermediates['spatial_features'] = spatial_features
            intermediates['combined_features'] = combined_features
        
        # Stage 5: Hybrid rendering
        rendering_outputs = self.hybrid_renderer(combined_features, coordinates)
        if return_intermediates:
            intermediates['rendering_outputs'] = rendering_outputs
        
        # Stage 6: Physics and material properties
        physics_outputs = self.physics_material(fused_global)
        if return_intermediates:
            intermediates['physics_outputs'] = physics_outputs
        
        # Stage 7: Scene composition
        scene_outputs = self.scene_composition(fused_global)
        if return_intermediates:
            intermediates['scene_outputs'] = scene_outputs
        
        # Combine all outputs
        outputs = {
            **rendering_outputs,
            **physics_outputs,
            **scene_outputs,
        }
        
        if return_intermediates:
            outputs['intermediates'] = intermediates
        
        return outputs

    def generate_3d_asset(self,
                         text_prompt: str,
                         image: Optional[torch.Tensor] = None,
                         sketch: Optional[torch.Tensor] = None,
                         depth: Optional[torch.Tensor] = None,
                         resolution: int = 128,
                         num_samples: int = 64) -> Dict[str, torch.Tensor]:
        """
        High-level interface for 3D asset generation
        
        Args:
            text_prompt: Text description of desired 3D asset
            image: Optional reference image
            sketch: Optional sketch input
            depth: Optional depth map
            resolution: Voxel resolution for output
            num_samples: Number of samples per ray
            
        Returns:
            Generated 3D asset data
        """
        # This would typically involve:
        # 1. Text encoding (using external encoder like CLIP)
        # 2. Ray generation for volume rendering
        # 3. Multiple forward passes for different views
        # 4. Post-processing for final 3D asset
        
        # Placeholder implementation
        batch_size = 1
        device = next(self.parameters()).device
        
        # Mock text features (would come from text encoder)
        text_features = torch.randn(batch_size, self.config.embed_dim, device=device)
        
        # Generate sample coordinates
        coords = torch.randn(batch_size, num_samples, 3, device=device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.forward(text_features, coords, image, sketch, depth)
        
        return outputs

    def get_model_info(self) -> Dict[str, Union[int, str]]:
        """Get model information"""
        return {
            'total_parameters': self.count_parameters(),
            'embedding_dimension': self.config.embed_dim,
            'transformer_layers': self.config.transformer.num_layers,
            'attention_heads': self.config.transformer.num_heads,
            'max_objects': self.config.scene_composition.max_objects,
            'hash_grid_levels': self.config.hash_grid.num_levels,
            'model_precision': str(self.config.dtype),
            'device': self.config.device
        }

# Factory function for easy model creation
def create_omnigen3d(embed_dim: int = 4096, 
                     num_layers: int = 32,
                     num_heads: int = 24,
                     device: str = "cuda",
                     **kwargs) -> OmniGen3D:
    """
    Factory function to create OmniGen3D model with custom configuration
    
    Args:
        embed_dim: Embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        device: Device to place model on
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured OmniGen3D model
    """
    config = OmniGen3DConfig(
        embed_dim=embed_dim,
        device=device,
        **kwargs
    )
    
    config.transformer.num_layers = num_layers
    config.transformer.num_heads = num_heads
    
    model = OmniGen3D(config).to(device)
    
    return model

if __name__ == "__main__":
    # Example usage
    print("Creating OmniGen3D model...")
    
    config = OmniGen3DConfig(embed_dim=4096, device="cuda")
    model = OmniGen3D(config)
    
    print(f"Model created with {model.count_parameters():,} parameters")
    print("Model info:", model.get_model_info())
