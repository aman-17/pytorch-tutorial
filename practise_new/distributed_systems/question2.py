"""
Question 2: Model Parallel Pipeline Training
Scenario: Implement pipeline parallelism for a large transformer that doesn't fit on a single GPU.

Key concepts:
1. Pipeline parallelism vs data parallelism
2. Model partitioning across devices
3. Micro-batching and pipeline scheduling
4. Gradient synchronization across pipeline stages
5. Memory optimization and bubble time reduction

Tasks:
- Partition transformer layers across multiple GPUs
- Implement micro-batching strategy
- Handle gradient synchronization across pipeline stages
- Optimize memory usage and bubble time
"""

import torch
import torch.nn as nn

# from torch.distributed.pipelining import pipeline, ScheduleGPipe, SplitPoint  # Not available in all PyTorch versions
import torch.distributed as dist
from typing import List, Dict, Any
import time

class EmbeddingLayer(nn.Module):
    """Combined embedding and positional embedding layer"""
    def __init__(self, token_embedding, pos_embedding):
        super().__init__()
        self.token_embedding = token_embedding
        self.pos_embedding = pos_embedding
    
    def forward(self, input_ids):
        seq_len = input_ids.shape[1]
        tok_embeds = self.token_embedding(input_ids)
        pos_ids = torch.arange(0, seq_len, device=input_ids.device, dtype=torch.long)
        pos_embeds = self.pos_embedding(pos_ids).unsqueeze(0)
        return tok_embeds + pos_embeds

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        shortcut = x
        x=self.norm1(x)
        x, _ = self.attention(x)
        x = self.dropout(x)
        x = shortcut + x
        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        return shortcut + x

class PipelineTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        # TODO: Design model for pipeline partitioning
        # 1. Embedding layer (typically on first device)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # 2. Transformer layers (to be distributed)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        
        # 3. Output head (typically on last device)
        self.ln_f = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, input_ids):
        # TODO: Implement forward pass for pipeline
        # Note: This will be modified when creating pipeline partitions
        batch_size, seq_len = input_ids.shape
        tok_embeds = self.embedding(input_ids)
        pos_ids = torch.arange(0, seq_len, device=input_ids.device, dtype=torch.long)
        pos_embeds = self.pos_embedding(pos_ids).unsqueeze(0)
        x = tok_embeds + pos_embeds
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.output_head(x)
        return logits

class PipelinePartitioner:
    """Handles model partitioning across devices"""
    
    def __init__(self, model, devices: List[str], balance_strategy='uniform'):
        self.model = model
        self.devices = devices
        self.balance_strategy = balance_strategy
        self.partitions = []
        self.world_size = len(devices)
    
    def create_partitions(self) -> List[nn.Sequential]:
        """
        Partition model layers across devices
        
        Returns:
            List of nn.Sequential modules, one per device
        """
        partitions = []
        total_layers = len(self.model.layers)
        layers_per_device = total_layers // self.world_size
        
        # First device gets embedding + some transformer layers
        first_partition = []
        first_partition.append(('embedding', self.model.embedding))
        first_partition.append(('pos_embedding', self.model.pos_embedding))
        
        # Add transformer layers to first device
        for i in range(layers_per_device):
            first_partition.append((f'layer_{i}', self.model.layers[i]))
        
        first_modules = [module for _, module in first_partition]
        partitions.append(nn.Sequential(*first_modules))
        
        # Middle devices get transformer layers only
        for device_idx in range(1, self.world_size - 1):
            start_layer = device_idx * layers_per_device
            end_layer = (device_idx + 1) * layers_per_device
            
            device_layers = []
            for i in range(start_layer, end_layer):
                device_layers.append((f'layer_{i}', self.model.layers[i]))
            
            device_modules = [module for _, module in device_layers]
            partitions.append(nn.Sequential(*device_modules))
        
        # Last device gets remaining layers + output head
        if self.world_size > 1:
            last_partition = []
            start_layer = (self.world_size - 1) * layers_per_device
            
            for i in range(start_layer, total_layers):
                last_partition.append((f'layer_{i}', self.model.layers[i]))
            
            last_partition.append(('ln_f', self.model.ln_f))
            last_partition.append(('output_head', self.model.output_head))
            
            last_modules = [module for _, module in last_partition]
        partitions.append(nn.Sequential(*last_modules))
        
        return partitions
    
    def balance_by_parameters(self) -> Dict[int, List[str]]:
        """Balance partitions by parameter count"""
        # TODO: Implement parameter-based balancing
        # 1. Count parameters in each layer
        # 2. Use greedy algorithm to balance load
        # 3. Consider memory overhead of activations
        pass
    
    def balance_by_computation(self) -> Dict[int, List[str]]:
        """Balance partitions by computation time"""
        # TODO: Implement computation-based balancing
        # 1. Profile individual layer forward/backward times
        # 2. Balance total computation per device
        # 3. Consider communication overhead
        pass

class PipelineScheduler:
    """Handles micro-batching and pipeline scheduling"""
    
    def __init__(self, n_microbatches, n_stages):
        self.n_microbatches = n_microbatches
        self.n_stages = n_stages
        self.schedule = []
    
    def create_schedule(self, schedule_type='1f1b'):
        """
        Create pipeline schedule
        
        Args:
            schedule_type: '1f1b' (1-forward-1-backward) or 'gpipe'
            
        Returns:
            List of (stage_id, microbatch_id, operation) tuples
        """
        schedule = []
        
        if schedule_type == 'gpipe':
            # GPipe: All forwards first, then all backwards
            # Forward phase
            for mb in range(self.n_microbatches):
                for stage in range(self.n_stages):
                    schedule.append((stage, mb, 'forward'))
            
            # Backward phase  
            for mb in range(self.n_microbatches - 1, -1, -1):
                for stage in range(self.n_stages - 1, -1, -1):
                    schedule.append((stage, mb, 'backward'))
        
        elif schedule_type == '1f1b':
            # 1F1B: Interleave forward and backward
            # Warmup phase: fill pipeline with forwards
            for mb in range(min(self.n_microbatches, self.n_stages)):
                for stage in range(mb + 1):
                    schedule.append((stage, mb, 'forward'))
            
            # Steady state: 1 forward, 1 backward per stage
            for mb in range(self.n_stages, self.n_microbatches):
                for stage in range(self.n_stages):
                    schedule.append((stage, mb, 'forward'))
                    if mb >= self.n_stages:
                        schedule.append((stage, mb - self.n_stages, 'backward'))
            
            # Cooldown: drain pipeline with backwards
            for mb in range(max(0, self.n_microbatches - self.n_stages), self.n_microbatches):
                for stage in range(self.n_stages - 1, -1, -1):
                    schedule.append((stage, mb, 'backward'))
        
        self.schedule = schedule
        return schedule
    
    def calculate_bubble_time(self) -> float:
        """Calculate pipeline bubble time"""
        # Bubble time = (n_stages - 1) / n_microbatches * 100%
        return (self.n_stages - 1) / self.n_microbatches

class MicroBatchManager:
    """Manages micro-batch creation and processing"""
    
    def __init__(self, batch_size, n_microbatches):
        self.batch_size = batch_size
        self.n_microbatches = n_microbatches
        self.microbatch_size = batch_size // n_microbatches
    
    def split_batch(self, batch):
        """Split batch into micro-batches"""
        microbatches = []
        
        if isinstance(batch, dict):
            # Handle dictionary batch (input_ids, labels, etc.)
            batch_size = list(batch.values())[0].shape[0]
            
            for i in range(self.n_microbatches):
                start_idx = i * self.microbatch_size
                end_idx = min((i + 1) * self.microbatch_size, batch_size)
                
                microbatch = {}
                for key, tensor in batch.items():
                    microbatch[key] = tensor[start_idx:end_idx]
                
                microbatches.append(microbatch)
        else:
            # Handle tensor batch
            batch_size = batch.shape[0]
            
            for i in range(self.n_microbatches):
                start_idx = i * self.microbatch_size
                end_idx = min((i + 1) * self.microbatch_size, batch_size)
                microbatches.append(batch[start_idx:end_idx])
        
        return microbatches
    
    def process_microbatch(self, microbatch, stage_id, partition):
        """Process single micro-batch through pipeline stage"""
        # Forward pass through stage
        if isinstance(microbatch, dict):
            # For first stage, use input_ids
            if stage_id == 0:
                output = partition(microbatch['input_ids'])
            else:
                # For other stages, microbatch is the tensor from previous stage
                output = partition(microbatch)
        else:
            output = partition(microbatch)
        
        return output

class PipelineTrainer:
    """Pipeline parallel trainer"""
    
    def __init__(self, model, devices, n_microbatches=8, schedule='1f1b'):
        self.model = model
        self.devices = devices
        self.n_microbatches = n_microbatches
        self.schedule_type = schedule
        
        # Initialize pipeline components
        self.partitioner = PipelinePartitioner(model, devices)
        self.scheduler = PipelineScheduler(n_microbatches, len(devices))
        self.pipeline_model = None
        self.partitions = None
        
    def setup_pipeline(self):
        """Setup pipeline parallel model"""
        # 1. Partition model across devices
        self.partitions = self.partitioner.create_partitions()
        
        # 2. Move partitions to respective devices
        for i, partition in enumerate(self.partitions):
            device = self.devices[i]
            partition = partition.to(device)
            self.partitions[i] = partition
            print(f"Partition {i} moved to {device}")
            
            # Print partition info
            param_count = sum(p.numel() for p in partition.parameters())
            print(f"  Parameters: {param_count:,}")
        
        print("Pipeline setup complete!")
    
    def create_pipeline_model(self, chunks=8):
        """Create pipeline model using manual pipeline implementation"""
        if self.partitions is None:
            self.setup_pipeline()
        
        # Create simple pipeline execution function
        def pipeline_forward(input_batch):
            # Split batch into micro-batches
            batch_manager = MicroBatchManager(input_batch.shape[0] if not isinstance(input_batch, dict) else list(input_batch.values())[0].shape[0], self.n_microbatches)
            microbatches = batch_manager.split_batch(input_batch)
            
            # Process each micro-batch through pipeline
            results = []
            for microbatch in microbatches:
                current_output = microbatch
                
                # Forward through each stage
                for stage_id, partition in enumerate(self.partitions):
                    device = self.devices[stage_id]
                    
                    # Move input to correct device
                    if isinstance(current_output, dict):
                        for key in current_output:
                            current_output[key] = current_output[key].to(device)
                    else:
                        current_output = current_output.to(device)
                    
                    # Process through partition
                    current_output = batch_manager.process_microbatch(current_output, stage_id, partition)
                
                results.append(current_output)
            
            # Concatenate results
            return torch.cat(results, dim=0)
        
        self.pipeline_model = pipeline_forward
        return self.pipeline_model
    
    def forward_backward_pipeline(self, batch):
        """Execute forward and backward passes through pipeline"""
        if self.pipeline_model is None:
            self.create_pipeline_model()
        
        # Execute forward pass
        output = self.pipeline_model(batch)
        
        # For demonstration, compute a simple loss
        if isinstance(batch, dict) and 'labels' in batch:
            target = batch['labels'].to(output.device)
            loss = torch.nn.functional.cross_entropy(output.view(-1, output.size(-1)), target.view(-1))
        else:
            # Dummy loss for demo
            loss = torch.mean(output ** 2)
        
        # Backward pass
        loss.backward()
        
        return loss.item()
    
    def optimize_memory_usage(self):
        """Optimize memory usage in pipeline"""
        # TODO: Implement memory optimizations
        # 1. Activation checkpointing
        # 2. Gradient accumulation timing
        # 3. Memory-efficient attention
        # 4. CPU offloading for activations
        pass
    
    def handle_load_balancing(self):
        """Handle load balancing across pipeline stages"""
        # TODO: Implement dynamic load balancing
        # 1. Monitor stage execution times
        # 2. Detect bottlenecks
        # 3. Suggest repartitioning strategies
        pass

class PipelineProfiler:
    """Profile pipeline performance and bottlenecks"""
    
    def __init__(self, pipeline_trainer):
        self.trainer = pipeline_trainer
        self.stage_times = {}
        self.communication_times = {}
        self.memory_usage = {}
    
    def profile_stage_times(self):
        """Profile execution time per pipeline stage"""
        # TODO: Implement stage timing
        # 1. Time forward/backward for each stage
        # 2. Identify bottleneck stages
        # 3. Calculate pipeline efficiency
        pass
    
    def profile_communication(self):
        """Profile inter-stage communication overhead"""
        # TODO: Implement communication profiling
        # 1. Time tensor transfers between stages
        # 2. Measure bandwidth utilization
        # 3. Identify communication bottlenecks
        pass
    
    def analyze_bubble_time(self):
        """Analyze and minimize bubble time"""
        # TODO: Implement bubble time analysis
        # 1. Calculate actual vs theoretical bubble time
        # 2. Suggest micro-batch size adjustments
        # 3. Recommend schedule optimizations
        pass

def create_sample_data(batch_size, seq_len, vocab_size):
    """Create sample data for testing"""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    return input_ids, labels

def compare_pipeline_strategies():
    """Compare different pipeline strategies"""
    # TODO: Implement comparison
    # 1. Compare 1F1B vs GPipe
    # 2. Test different micro-batch sizes
    # 3. Analyze throughput and memory usage
    # 4. Generate performance reports
    pass

def main():
    """Main function to test pipeline parallelism"""
    # Model configuration
    vocab_size = 50000
    d_model = 512  # Smaller for demo
    n_layers = 12  # 12 layers across 4 GPUs = 3 layers per GPU
    n_heads = 8
    seq_len = 256
    batch_size = 32
    
    # Pipeline configuration
    devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
    n_microbatches = 8
    
    print(f"Model: {n_layers} layers, {d_model} dimensions")
    print(f"Pipeline: {len(devices)} stages, {n_microbatches} micro-batches")
    
    # Create model
    model = PipelineTransformer(vocab_size, d_model, n_layers, n_heads)
    
    # Test partitioning
    partitioner = PipelinePartitioner(model, devices)
    partitions = partitioner.create_partitions()
    
    print(f"\nPartitions created: {len(partitions)}")
    for i, partition in enumerate(partitions):
        param_count = sum(p.numel() for p in partition.parameters())
        print(f"  Device {i}: {len(partition)} modules, {param_count:,} parameters")
    
    # Test scheduling
    scheduler = PipelineScheduler(n_microbatches, len(devices))
    
    # Compare scheduling strategies
    for schedule_type in ['gpipe', '1f1b']:
        schedule = scheduler.create_schedule(schedule_type)
        bubble_time = scheduler.calculate_bubble_time()
        
        print(f"\n{schedule_type.upper()} Schedule:")
        print(f"  Total operations: {len(schedule)}")
        print(f"  Bubble time: {bubble_time:.2%}")
        
        # Show first few operations
        print("  First 10 operations:")
        for i, (stage, mb, op) in enumerate(schedule[:10]):
            print(f"    {i+1}: Stage {stage}, Microbatch {mb}, {op}")
        
        if len(schedule) > 10:
            print(f"    ... and {len(schedule) - 10} more operations")

if __name__ == "__main__":
    main()