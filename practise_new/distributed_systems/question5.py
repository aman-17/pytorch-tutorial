"""
Question 5: Memory-Efficient Training with ZeRO and Activation Checkpointing
Scenario: Train a 70B parameter model with limited GPU memory using advanced memory optimization techniques.

Key concepts:
1. ZeRO optimizer states partitioning (stages 1-3)
2. Activation checkpointing and recomputation
3. CPU offloading for optimizer states and parameters
4. Memory-aware dynamic batching
5. Parameter offloading and streaming
6. Gradient compression and accumulation

Tasks:
- Implement ZeRO optimizer state partitioning (Stages 1-3)
- Design selective activation checkpointing strategy
- Optimize CPU-GPU memory transfers for large models
- Create memory-aware dynamic batching system
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torch.distributed as dist
from torch.optim import AdamW
from typing import List, Dict, Optional, Tuple, Iterator, Any
import gc
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not available. CPU memory monitoring will be limited.")
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
import numpy as np

class MemoryProfiler:
    """Profile and monitor memory usage"""
    
    def __init__(self):
        self.memory_snapshots = []
        self.peak_memory = 0
        self.allocation_history = []
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics"""
        stats = {}
        
        # GPU memory statistics
        if torch.cuda.is_available():
            stats.update({
                'gpu_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'gpu_cached_gb': torch.cuda.memory_reserved() / (1024**3),
                'gpu_max_allocated_gb': torch.cuda.max_memory_allocated() / (1024**3),
                'gpu_max_cached_gb': torch.cuda.max_memory_reserved() / (1024**3),
            })
        
        # CPU memory statistics
        if HAS_PSUTIL:
            process = psutil.Process()
            memory_info = process.memory_info()
            stats.update({
                'cpu_rss_gb': memory_info.rss / (1024**3),  # Resident set size
                'cpu_vms_gb': memory_info.vms / (1024**3),  # Virtual memory size
                'cpu_percent': process.memory_percent(),     # Percentage of system memory
                'system_memory_gb': psutil.virtual_memory().total / (1024**3),
                'system_available_gb': psutil.virtual_memory().available / (1024**3),
            })
        else:
            # Fallback CPU memory stats
            stats.update({
                'cpu_rss_gb': 0.0,
                'cpu_vms_gb': 0.0,
                'cpu_percent': 0.0,
                'system_memory_gb': 0.0,
                'system_available_gb': 0.0,
            })
        
        # Calculate efficiency metrics
        if torch.cuda.is_available():
            gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            stats['gpu_utilization'] = stats['gpu_allocated_gb'] / gpu_total
            stats['gpu_efficiency'] = stats['gpu_allocated_gb'] / stats['gpu_cached_gb'] if stats['gpu_cached_gb'] > 0 else 0
        
        return stats
    
    @contextmanager
    def profile_memory_usage(self, operation_name: str):
        """Context manager to profile memory usage of operations"""
        # Record memory before operation
        start_stats = self.get_memory_stats()
        start_time = time.time()
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        try:
            yield
        finally:
            # Record memory after operation
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_stats = self.get_memory_stats()
            end_time = time.time()
            
            # Calculate deltas
            memory_delta = {}
            for key in start_stats:
                if key in end_stats:
                    memory_delta[f'delta_{key}'] = end_stats[key] - start_stats[key]
            
            # Record the profiling result
            profile_result = {
                'operation': operation_name,
                'duration_sec': end_time - start_time,
                'start_memory': start_stats,
                'end_memory': end_stats,
                'memory_delta': memory_delta,
                'peak_gpu_memory_gb': torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
            }
            
            self.memory_snapshots.append(profile_result)
            
            # Update peak memory tracking
            current_peak = profile_result['peak_gpu_memory_gb']
            if current_peak > self.peak_memory:
                self.peak_memory = current_peak
            
            print(f"Memory profile [{operation_name}]: "
                  f"Duration: {profile_result['duration_sec']:.2f}s, "
                  f"Peak GPU: {current_peak:.2f}GB, "
                  f"Delta GPU: {memory_delta.get('delta_gpu_allocated_gb', 0):.2f}GB")
    
    def optimize_memory_layout(self, tensors: List[torch.Tensor]):
        """Optimize tensor memory layout for efficiency"""
        # TODO: Implement memory layout optimization
        # 1. Analyze tensor access patterns
        # 2. Coalesce small tensors
        # 3. Align memory for optimal access
        # 4. Reduce memory fragmentation
        pass

class ZeROOptimizer:
    """ZeRO-style optimizer with state partitioning"""
    
    def __init__(self, params, base_optimizer_class=AdamW, partition_size: Optional[int] = None,
                 cpu_offload: bool = False, overlap_comm: bool = True):
        self.params = list(params)
        self.base_optimizer_class = base_optimizer_class
        self.cpu_offload = cpu_offload
        self.overlap_comm = overlap_comm
        
        # Distributed setup
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.partition_size = partition_size or (len(self.params) // self.world_size)
        
        # ZeRO state
        self.param_to_rank = {}
        self.rank_to_params = defaultdict(list)
        self.optimizer_states = {}
        self.gathered_params = {}
        
        # Setup partitioning
        self.setup_partitioning()
    
    def setup_partitioning(self):
        """Setup parameter partitioning across ranks"""
        # Assign parameters to ranks using round-robin
        for i, param in enumerate(self.params):
            assigned_rank = i % self.world_size
            self.param_to_rank[param] = assigned_rank
            self.rank_to_params[assigned_rank].append(param)
        
        # Create local optimizer for parameters assigned to this rank
        local_params = self.rank_to_params[self.rank]
        if local_params:
            self.local_optimizer = self.base_optimizer_class(local_params)
        else:
            self.local_optimizer = None
        
        print(f"Rank {self.rank}: Assigned {len(local_params)} parameters")
    
    def zero_stage1(self):
        """
        ZeRO Stage 1: Partition optimizer states across ranks
        Each rank stores optimizer states for 1/N of parameters
        """
        # Each rank only maintains optimizer states for its assigned parameters
        # This is already handled in setup_partitioning by creating local_optimizer
        
        # Register hooks to gather optimizer states when needed
        def optimizer_state_hook(param):
            if param in self.param_to_rank:
                owner_rank = self.param_to_rank[param]
                if owner_rank != self.rank:
                    # Need to gather state from the owning rank
                    # In practice, this would involve communication
                    pass
        
        # Apply hooks to all parameters
        for param in self.params:
            param.register_hook(optimizer_state_hook)
        
        print(f"ZeRO Stage 1 initialized on rank {self.rank}")
    
    def zero_stage2(self):
        """
        ZeRO Stage 2: Partition gradients across ranks
        Gradients are reduced-scattered during backward pass
        """
        self.gradient_hooks = []
        
        def gradient_reduce_scatter_hook(param):
            def hook_fn(grad):
                if grad is not None and dist.is_initialized():
                    # Perform reduce-scatter on gradient
                    # Each rank gets its portion of the gradient
                    owner_rank = self.param_to_rank.get(param, self.rank)
                    
                    if owner_rank == self.rank:
                        # This rank owns this parameter, reduce gradient from all ranks
                        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                        grad.div_(self.world_size)
                    else:
                        # This rank doesn't own this parameter, zero out gradient
                        grad.zero_()
                
                return grad
            return hook_fn
        
        # Register gradient hooks
        for param in self.params:
            if param.requires_grad:
                hook = param.register_hook(gradient_reduce_scatter_hook(param))
                self.gradient_hooks.append(hook)
        
        print(f"ZeRO Stage 2 initialized on rank {self.rank}")
    
    def zero_stage3(self):
        """
        ZeRO Stage 3: Partition model parameters across ranks
        Parameters are gathered on-demand during forward/backward
        """
        # Store original parameters and replace with placeholders
        self.original_params = {}
        
        for param in self.params:
            owner_rank = self.param_to_rank[param]
            self.original_params[param] = param.data.clone()
            
            if owner_rank != self.rank:
                # This rank doesn't own this parameter, create empty placeholder
                param.data = torch.empty(0, dtype=param.dtype, device=param.device)
        
        print(f"ZeRO Stage 3 initialized on rank {self.rank}")
        print(f"Memory saved by partitioning parameters")
    
    def gather_parameters(self, param_list: List[nn.Parameter]) -> Dict[nn.Parameter, torch.Tensor]:
        """Gather partitioned parameters for computation"""
        gathered_params = {}
        
        for param in param_list:
            owner_rank = self.param_to_rank.get(param, self.rank)
            
            if owner_rank == self.rank:
                # This rank owns the parameter
                gathered_params[param] = param.data
            else:
                # Need to gather from owner rank
                if dist.is_initialized():
                    # Create tensor to receive the parameter
                    gathered_param = torch.zeros_like(self.original_params[param])
                    
                    # Broadcast from owner rank to all ranks
                    dist.broadcast(gathered_param, src=owner_rank)
                    gathered_params[param] = gathered_param
                    
                    # Temporarily assign to parameter for computation
                    param.data = gathered_param
                else:
                    # Single GPU case
                    gathered_params[param] = self.original_params[param]
                    param.data = self.original_params[param]
        
        return gathered_params
    
    def release_parameters(self, param_list: List[nn.Parameter]):
        """Release gathered parameters to save memory"""
        for param in param_list:
            owner_rank = self.param_to_rank.get(param, self.rank)
            
            if owner_rank != self.rank:
                # This rank doesn't own this parameter, release gathered data
                param.data = torch.empty(0, dtype=param.dtype, device=param.device)
        
        # Force garbage collection to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def offload_to_cpu(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """Offload tensors to CPU memory"""
        # TODO: Implement CPU offloading
        # 1. Transfer tensors to CPU asynchronously
        # 2. Manage CPU memory efficiently
        # 3. Implement streaming for large tensors
        # 4. Track offloaded tensors
        pass
    
    def prefetch_from_cpu(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """Prefetch tensors from CPU to GPU"""
        # TODO: Implement CPU prefetching
        # 1. Predict which tensors will be needed
        # 2. Transfer to GPU asynchronously
        # 3. Overlap transfer with computation
        # 4. Manage GPU memory for prefetched tensors
        pass
    
    def step(self):
        """Perform optimizer step with ZeRO"""
        # TODO: Implement ZeRO optimizer step
        # 1. Gather required optimizer states
        # 2. Perform local optimization
        # 3. Update parameters and states
        # 4. Handle communication efficiently
        pass
    
    def state_dict(self) -> Dict[str, Any]:
        """Get complete optimizer state dict"""
        # TODO: Implement state dict gathering
        # 1. Collect states from all ranks
        # 2. Reconstruct complete state dict
        # 3. Handle large state efficiently
        pass
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state dict"""
        # TODO: Implement state dict loading
        # 1. Partition state dict across ranks
        # 2. Load local portion of states
        # 3. Ensure consistency across ranks
        pass

class ActivationCheckpointing:
    """Advanced activation checkpointing strategies"""
    
    def __init__(self, memory_budget: float = 0.8):
        self.memory_budget = memory_budget  # Fraction of GPU memory to use
        self.checkpoint_config = {}
        self.memory_profiler = MemoryProfiler()
    
    def selective_checkpoint(self, model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """
        Implement selective activation checkpointing
        Choose which layers to checkpoint based on memory/compute trade-off
        """
        # Get checkpointing strategy
        checkpoint_config = self.compute_checkpointing_strategy(model, inputs)
        
        x = inputs
        for i, layer in enumerate(model.layers):
            layer_name = f"layer_{i}"
            
            if checkpoint_config.get(layer_name, False):
                # Use gradient checkpointing for this layer
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                # Regular forward pass
                x = layer(x)
        
        return x
    
    def memory_aware_checkpointing(self, layers: List[nn.Module], 
                                 input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Memory-aware checkpointing that adapts to available memory
        """
        x = input_tensor
        memory_stats = self.memory_profiler.get_memory_stats()
        current_utilization = memory_stats.get('gpu_utilization', 0.0)
        
        # Dynamic checkpointing based on memory pressure
        checkpoint_threshold = 0.8  # Checkpoint if GPU utilization > 80%
        
        for i, layer in enumerate(layers):
            if current_utilization > checkpoint_threshold:
                # High memory pressure - use checkpointing
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                # Normal forward pass
                x = layer(x)
            
            # Update memory stats periodically
            if i % 5 == 0:  # Check every 5 layers
                memory_stats = self.memory_profiler.get_memory_stats()
                current_utilization = memory_stats.get('gpu_utilization', 0.0)
        
        return x
    
    def attention_checkpointing(self, attention_layer: nn.Module, 
                              query: torch.Tensor, key: torch.Tensor, 
                              value: torch.Tensor) -> torch.Tensor:
        """
        Specialized checkpointing for attention mechanisms
        """
        # TODO: Implement attention checkpointing
        # 1. Checkpoint attention score computation
        # 2. Recompute attention weights during backward
        # 3. Optimize for different attention patterns
        # 4. Handle different attention implementations
        pass
    
    def gradient_checkpointing_with_rng(self, function, *args):
        """Gradient checkpointing that preserves RNG state"""
        # TODO: Implement RNG-aware checkpointing
        # 1. Save RNG state before forward pass
        # 2. Restore RNG state during recomputation
        # 3. Ensure dropout patterns are consistent
        # 4. Handle multiple RNG streams
        pass
    
    def compute_checkpointing_strategy(self, model: nn.Module, 
                                     sample_input: torch.Tensor) -> Dict[str, bool]:
        """
        Compute optimal checkpointing strategy for a model
        """
        strategy = {}
        
        # Profile memory usage per layer
        layer_memory_usage = {}
        
        with self.memory_profiler.profile_memory_usage("layer_profiling"):
            x = sample_input
            for i, layer in enumerate(model.layers):
                layer_name = f"layer_{i}"
                
                # Measure memory before layer
                memory_before = self.memory_profiler.get_memory_stats()['gpu_allocated_gb']
                
                # Forward pass through layer
                x = layer(x)
                
                # Measure memory after layer
                memory_after = self.memory_profiler.get_memory_stats()['gpu_allocated_gb']
                
                layer_memory_usage[layer_name] = memory_after - memory_before
        
        # Simple heuristic: checkpoint layers with high memory usage
        memory_threshold = np.percentile(list(layer_memory_usage.values()), 75)  # Top 25% by memory
        
        for layer_name, memory_usage in layer_memory_usage.items():
            strategy[layer_name] = memory_usage > memory_threshold
        
        # Ensure we don't checkpoint too many layers (would hurt performance)
        total_checkpointed = sum(strategy.values())
        max_checkpoints = len(model.layers) // 2  # At most 50% of layers
        
        if total_checkpointed > max_checkpoints:
            # Keep only the most memory-intensive layers
            sorted_layers = sorted(layer_memory_usage.items(), key=lambda x: x[1], reverse=True)
            strategy = {name: False for name in layer_memory_usage.keys()}
            for name, _ in sorted_layers[:max_checkpoints]:
                strategy[name] = True
        
        return strategy

class MemoryOptimizedTransformer(nn.Module):
    """Transformer with memory optimizations"""
    
    def __init__(self, config: Dict[str, Any], use_checkpointing: bool = True,
                 cpu_offloading: bool = False):
        super().__init__()
        self.config = config
        self.use_checkpointing = use_checkpointing
        self.cpu_offloading = cpu_offloading
        
        # Model components
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config['n_layers'])
        ])
        self.ln_f = nn.LayerNorm(config['d_model'])
        self.output_head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
        
        # Memory optimization components
        self.checkpointer = ActivationCheckpointing()
        self.memory_manager = None
        
    def forward(self, input_ids: torch.Tensor, use_checkpointing: Optional[bool] = None) -> torch.Tensor:
        """Forward pass with memory optimizations"""
        # TODO: Implement memory-optimized forward pass
        # 1. Apply embedding with potential CPU offloading
        # 2. Process layers with selective checkpointing
        # 3. Manage activation memory efficiently
        # 4. Apply output projection
        pass
    
    def forward_with_offloading(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass with CPU offloading"""
        # TODO: Implement CPU offloading forward pass
        # 1. Stream parameters from CPU as needed
        # 2. Offload intermediate activations to CPU
        # 3. Manage data transfers efficiently
        # 4. Overlap computation with transfers
        pass

class TransformerBlock(nn.Module):
    """Memory-optimized transformer block"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        d_model = config['d_model']
        n_heads = config['n_heads']
        
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
    
    def forward(self, x: torch.Tensor, use_checkpoint: bool = True) -> torch.Tensor:
        """Forward with optional checkpointing"""
        if use_checkpoint:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attention(x, x, x)
        x = residual + attn_out
        
        # Feed-forward with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x

class DynamicBatchManager:
    """Memory-aware dynamic batching"""
    
    def __init__(self, max_memory_gb: float = 24.0, safety_margin: float = 0.1):
        self.max_memory = max_memory_gb * 1024**3  # Convert to bytes
        self.safety_margin = safety_margin
        self.memory_profiler = MemoryProfiler()
        
        # Adaptive batching state
        self.current_batch_size = 1
        self.batch_size_history = []
        self.memory_usage_history = []
    
    def estimate_memory_usage(self, batch_size: int, seq_len: int, 
                            model_config: Dict[str, Any]) -> float:
        """Estimate memory usage for given batch size"""
        # TODO: Implement memory estimation
        # 1. Calculate activation memory requirements
        # 2. Include gradient memory
        # 3. Account for optimizer states
        # 4. Add safety margins
        pass
    
    def adapt_batch_size(self, current_memory: float, target_memory: float) -> int:
        """Adapt batch size based on memory usage"""
        # TODO: Implement adaptive batching
        # 1. Calculate optimal batch size for target memory
        # 2. Apply gradual changes to avoid instability
        # 3. Consider training dynamics
        # 4. Update batch size history
        pass
    
    def create_dynamic_dataloader(self, dataset, model_config: Dict[str, Any]) -> Iterator:
        """Create dataloader with dynamic batching"""
        # TODO: Implement dynamic dataloader
        # 1. Monitor memory usage during training
        # 2. Adjust batch size based on available memory
        # 3. Handle variable sequence lengths
        # 4. Maintain training stability
        pass

class MemoryEfficientTrainer:
    """Trainer with comprehensive memory optimizations"""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        
        # Memory optimization components
        self.zero_optimizer = None
        self.checkpointer = ActivationCheckpointing()
        self.batch_manager = DynamicBatchManager(config.get('max_memory_gb', 24))
        self.memory_profiler = MemoryProfiler()
        
        # Training state
        self.current_step = 0
        self.memory_budget = config.get('memory_budget', 0.9)
        
    def setup_zero_optimizer(self, stage: int = 2):
        """Setup ZeRO optimizer"""
        # TODO: Implement ZeRO optimizer setup
        # 1. Choose appropriate ZeRO stage
        # 2. Configure CPU offloading if needed
        # 3. Set up parameter partitioning
        # 4. Initialize optimizer with ZeRO
        pass
    
    def train_step_memory_efficient(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Memory-efficient training step"""
        # TODO: Implement memory-efficient training step
        # 1. Monitor memory before step
        # 2. Apply dynamic batching if needed
        # 3. Use gradient checkpointing selectively
        # 4. Manage optimizer states efficiently
        # 5. Track memory usage throughout step
        pass
    
    def optimize_memory_during_training(self):
        """Continuously optimize memory usage during training"""
        # TODO: Implement continuous memory optimization
        # 1. Monitor memory trends
        # 2. Adjust checkpointing strategies
        # 3. Trigger garbage collection when needed
        # 4. Offload unused parameters
        pass
    
    def handle_oom_recovery(self):
        """Recover from out-of-memory errors"""
        # TODO: Implement OOM recovery
        # 1. Detect OOM conditions
        # 2. Reduce batch size
        # 3. Increase checkpointing
        # 4. Retry training step
        pass

def benchmark_memory_techniques():
    """Benchmark different memory optimization techniques"""
    # TODO: Implement memory technique benchmarking
    # 1. Test ZeRO stages 1, 2, 3
    # 2. Compare checkpointing strategies
    # 3. Measure CPU offloading effectiveness
    # 4. Analyze memory vs speed trade-offs
    pass

def profile_large_model_training():
    """Profile memory usage for large model training"""
    # TODO: Implement large model profiling
    # 1. Create large model (70B+ parameters)
    # 2. Profile memory usage with different optimizations
    # 3. Identify memory bottlenecks
    # 4. Generate optimization recommendations
    pass

def main():
    """Main function to test memory-efficient training"""
    # TODO: Implement main testing function
    # 1. Setup large model for testing
    # 2. Test ZeRO optimizations
    # 3. Benchmark activation checkpointing
    # 4. Test dynamic batching
    # 5. Profile memory usage patterns
    
    # Large model configuration (70B parameters)
    config = {
        'vocab_size': 50000,
        'd_model': 8192,
        'n_layers': 80,
        'n_heads': 64,
        'max_seq_len': 2048,
        'max_memory_gb': 24.0
    }
    
    print("TODO: Implement memory-efficient training testing")
    print(f"Testing with model config: {config}")
    print("Optimization techniques:")
    print("- ZeRO Stage 1: Optimizer state partitioning")
    print("- ZeRO Stage 2: Gradient partitioning")
    print("- ZeRO Stage 3: Parameter partitioning")
    print("- Activation checkpointing")
    print("- CPU offloading")
    print("- Dynamic batching")

if __name__ == "__main__":
    main()