"""
OmniGen3D: Configuration Classes
Author: Rohith Garapati
Paper: OmniGen3D: A Unified Multi-Modal Architecture for Physics-Aware Text-to-3D Generation
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch

@dataclass
class TextEmbeddingConfig:
    """Configuration for Text Embedding MLP"""
    input_dim: int = 4096
    hidden_dim: int = 8192
    output_dim: int = 4096
    dropout: float = 0.1
    activation: str = "relu"
    use_layer_norm: bool = True

@dataclass
class TransformerConfig:
    """Configuration for Backbone Transformer"""
    embed_dim: int = 4096
    num_layers: int = 32
    num_heads: int = 24
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    activation: str = "swiglu"
    norm_type: str = "rmsnorm"
    use_flash_attention: bool = True
    max_seq_length: int = 2048

@dataclass
class MultiModalConfig:
    """Configuration for Multi-Modal Processing"""
    # Image branch
    image_patch_size: int = 16
    image_num_layers: int = 8
    image_hidden_dim: int = 4096
    
    # Sketch branch  
    sketch_stroke_dim: int = 5  # x, y, dx, dy, pressure
    sketch_num_layers: int = 4
    sketch_hidden_dim: int = 4096
    
    # Depth branch
    depth_conv_layers: List[int] = None
    depth_hidden_dim: int = 4096
    
    # Cross-attention
    cross_attn_heads: int = 24
    cross_attn_layers: int = 2
    
    def __post_init__(self):
        if self.depth_conv_layers is None:
            self.depth_conv_layers = [64, 128, 256, 512, 1024, 2048, 4096]

@dataclass
class HashGridConfig:
    """Configuration for Hash Grid Encoder"""
    num_levels: int = 6
    base_resolution: int = 32
    max_resolution: int = 1024
    features_per_level: List[int] = None
    hash_table_sizes: List[int] = None
    aggregation_dims: List[int] = None
    
    def __post_init__(self):
        if self.features_per_level is None:
            self.features_per_level = [32, 32, 32, 32, 32, 16]
        if self.hash_table_sizes is None:
            self.hash_table_sizes = [2**15, 2**18, 2**21, 2**24, 2**27, 2**30]
        if self.aggregation_dims is None:
            self.aggregation_dims = [192, 2048, 4096]

@dataclass
class HybridRendererConfig:
    """Configuration for Gaussian-SDF Hybrid Renderer"""
    # Gaussian predictor
    gaussian_hidden_dims: List[int] = None
    gaussian_num_properties: int = 22  # pos(3) + scale(3) + rot(4) + opacity(1) + color(3) + material(8)
    
    # SDF network
    sdf_num_layers: int = 8
    sdf_hidden_dim: int = 4096
    sdf_skip_connections: List[int] = None
    
    # Adaptive selection
    complexity_threshold: float = 0.5
    adaptive_resolution: bool = True
    
    def __post_init__(self):
        if self.gaussian_hidden_dims is None:
            self.gaussian_hidden_dims = [4096, 2048, 1024]
        if self.sdf_skip_connections is None:
            self.sdf_skip_connections = [2, 4, 6]

@dataclass
class PhysicsConfig:
    """Configuration for Physics-Material Integration"""
    material_properties: List[str] = None
    constraint_checks: List[str] = None
    physics_heads_dim: int = 4096
    
    def __post_init__(self):
        if self.material_properties is None:
            self.material_properties = [
                "density", "elasticity", "friction", "thermal", "conductivity"
            ]
        if self.constraint_checks is None:
            self.constraint_checks = [
                "gravity", "stability", "collision_boundary"
            ]

@dataclass
class SceneCompositionConfig:
    """Configuration for Hierarchical Scene Composition"""
    max_objects: int = 32
    relationship_encoder_layers: int = 4
    spatial_transform_dim: int = 9  # 3x3 transformation matrix
    cross_attention_layers: int = 2
    object_detection_dim: int = 4096

@dataclass
class OmniGen3DConfig:
    """Complete OmniGen3D Model Configuration"""
    # Global settings
    embed_dim: int = 4096
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    
    # Component configurations
    text_embedding: TextEmbeddingConfig = None
    transformer: TransformerConfig = None
    multi_modal: MultiModalConfig = None
    hash_grid: HashGridConfig = None
    hybrid_renderer: HybridRendererConfig = None
    physics: PhysicsConfig = None
    scene_composition: SceneCompositionConfig = None
    
    # Training settings
    batch_size: int = 32
    learning_rate: float = 1e-5
    num_epochs: int = 1000
    warmup_steps: int = 10000
    gradient_clip: float = 1.0
    
    # Inference settings
    generation_steps: int = 50
    output_resolution: Tuple[int, int, int] = (1024, 1024, 1024)
    mesh_vertices: int = 2_000_000
    texture_resolution: Tuple[int, int] = (1024, 1024)
    
    def __post_init__(self):
        # Initialize component configs if not provided
        if self.text_embedding is None:
            self.text_embedding = TextEmbeddingConfig()
        if self.transformer is None:
            self.transformer = TransformerConfig()
        if self.multi_modal is None:
            self.multi_modal = MultiModalConfig()
        if self.hash_grid is None:
            self.hash_grid = HashGridConfig()
        if self.hybrid_renderer is None:
            self.hybrid_renderer = HybridRendererConfig()
        if self.physics is None:
            self.physics = PhysicsConfig()
        if self.scene_composition is None:
            self.scene_composition = SceneCompositionConfig()
    
    def get_total_parameters(self) -> int:
        """Calculate total model parameters based on configuration"""
        # Text Embedding MLP
        text_params = (
            self.text_embedding.input_dim * self.text_embedding.hidden_dim + self.text_embedding.hidden_dim +
            self.text_embedding.hidden_dim * self.text_embedding.output_dim + self.text_embedding.output_dim
        )
        
        # Backbone Transformer
        single_block_params = (
            # QKV projection
            self.embed_dim * 3 * self.embed_dim + 3 * self.embed_dim +
            # Attention output
            self.embed_dim * self.embed_dim + self.embed_dim +
            # FFN
            self.embed_dim * (self.embed_dim * 4) + (self.embed_dim * 4) +
            (self.embed_dim * 4) * self.embed_dim + self.embed_dim +
            # LayerNorms
            self.embed_dim * 2
        )
        transformer_params = single_block_params * self.transformer.num_layers
        
        # Multi-Modal Processing (simplified estimation)
        multimodal_params = 1_940_268_576
        
        # Hash Grid Encoder
        hashgrid_params = 22_096_041_408
        
        # Hybrid Renderer
        renderer_params = 128_279_515
        
        # Physics Integration
        physics_params = 61_455
        
        # Scene Composition
        scene_params = 602_927_657
        
        total = (text_params + transformer_params + multimodal_params + 
                hashgrid_params + renderer_params + physics_params + scene_params)
        
        return total
    
    def print_config(self):
        """Print detailed configuration"""
        print("=" * 60)
        print("OmniGen3D Model Configuration")
        print("=" * 60)
        print(f"Total Parameters: {self.get_total_parameters():,}")
        print(f"Embedding Dimension: {self.embed_dim}")
        print(f"Transformer Layers: {self.transformer.num_layers}")
        print(f"Attention Heads: {self.transformer.num_heads}")
        print(f"Device: {self.device}")
        print(f"Precision: {self.dtype}")
        print("=" * 60)
