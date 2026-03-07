"""
PyTorch DataLoader Interview Questions - Practice Set
Focus: Data loading, DistributedSampler, custom samplers, optimization techniques

These questions cover common PyTorch interview topics around efficient data loading,
especially for distributed training scenarios.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from typing import Iterator, List, Optional
import math


"""
================================================================================
QUESTION 1: Implement a Custom DistributedSampler with Weighted Sampling
================================================================================

Scenario: You have an imbalanced dataset where you want to:
1. Use weighted sampling to balance classes
2. Support distributed training across multiple GPUs
3. Ensure each GPU gets different samples without overlap

Task: Implement a DistributedWeightedSampler that combines weighted random sampling
with distributed training requirements.

Key Concepts:
- DistributedSampler behavior (rank, world_size, sharding)
- Weighted sampling for class imbalance
- Reproducibility with seed
- Padding/truncation to ensure equal samples across ranks
"""

class DistributedWeightedSampler(Sampler):
    """
    Combines weighted sampling with distributed training support.

    Args:
        dataset: Dataset to sample from
        weights: Weight for each sample (higher = more likely to be sampled)
        num_samples: Number of samples to draw per epoch per rank
        rank: Process rank in distributed training
        world_size: Total number of processes
        replacement: Whether to sample with replacement
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        dataset: Dataset,
        weights: List[float],
        num_samples: Optional[int] = None,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        replacement: bool = True,
        seed: int = 0
    ):
        # TODO: Implement initialization
        # 1. Handle distributed setup (rank, world_size)
        # 2. Calculate num_samples per rank
        # 3. Store weights and seed
        # 4. Calculate total_size (padded to be divisible by world_size)

        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        if world_size is None:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.dataset = dataset
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.rank = rank
        self.world_size = world_size
        self.replacement = replacement
        self.seed = seed
        self.epoch = 0

        # Calculate number of samples per rank
        if num_samples is None:
            num_samples = len(self.dataset)

        # Total samples must be divisible by world_size
        self.total_size = math.ceil(num_samples / self.world_size) * self.world_size
        self.num_samples = self.total_size // self.world_size

    def __iter__(self) -> Iterator[int]:
        # TODO: Implement sampling logic
        # 1. Set random seed (epoch + seed for different samples each epoch)
        # 2. Generate weighted random indices
        # 3. Pad/truncate to total_size
        # 4. Subsample for current rank
        # 5. Return iterator

        # Set seed for reproducibility
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Generate weighted random samples
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            self.replacement,
            generator=g
        ).tolist()

        # Subsample for current rank
        indices = indices[self.rank:self.total_size:self.world_size]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for shuffling (call at start of each epoch)"""
        self.epoch = epoch


"""
================================================================================
QUESTION 2: Implement Variable-Length Sequence Batching with Padding
================================================================================

Scenario: You're training an NLP model where sequences have different lengths.
You need to:
1. Group sequences of similar length into batches (minimize padding)
2. Dynamically pad sequences to the max length in each batch
3. Create attention masks for padded sequences
4. Support distributed training

Task: Implement a custom collate function and sampler for efficient variable-length
sequence batching.
"""

class SequenceDataset(Dataset):
    """Dataset with variable-length sequences"""

    def __init__(self, sequences: List[List[int]], labels: List[int]):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.sequences[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class BucketBatchSampler(Sampler):
    """
    Groups sequences of similar length into batches to minimize padding.

    Args:
        data_source: Dataset with sequences
        batch_size: Batch size
        drop_last: Whether to drop incomplete batches
        sort_key: Function to get sequence length
    """

    def __init__(
        self,
        data_source: Dataset,
        batch_size: int,
        drop_last: bool = False,
        sort_key=lambda x: len(x['input_ids'])
    ):
        # TODO: Implement bucket batching
        # 1. Get lengths of all sequences
        # 2. Sort indices by length
        # 3. Create batches of similar-length sequences
        # 4. Optionally shuffle batches

        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sort_key = sort_key

        # Get lengths and sort indices
        self.indices_by_length = []
        for idx in range(len(data_source)):
            item = data_source[idx]
            length = self.sort_key(item)
            self.indices_by_length.append((idx, length))

        # Sort by length
        self.indices_by_length.sort(key=lambda x: x[1])

    def __iter__(self):
        # TODO: Generate batches
        # 1. Create batches from sorted indices
        # 2. Optionally shuffle batch order
        # 3. Yield batches

        batches = []
        batch = []

        for idx, length in self.indices_by_length:
            batch.append(idx)
            if len(batch) == self.batch_size:
                batches.append(batch)
                batch = []

        # Handle remaining samples
        if len(batch) > 0 and not self.drop_last:
            batches.append(batch)

        # Shuffle batch order
        np.random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return math.ceil(len(self.data_source) / self.batch_size)


def variable_length_collate_fn(batch: List[dict]) -> dict:
    """
    Collate function that pads sequences to max length in batch.

    Args:
        batch: List of samples from dataset

    Returns:
        Dictionary with padded tensors and attention masks
    """
    # TODO: Implement padding and masking
    # 1. Find max length in batch
    # 2. Pad all sequences to max length
    # 3. Create attention mask (1 for real tokens, 0 for padding)
    # 4. Stack into tensors

    # Find max length
    max_len = max(len(item['input_ids']) for item in batch)

    # Pad sequences
    input_ids = []
    attention_mask = []
    labels = []

    for item in batch:
        seq = item['input_ids']
        seq_len = len(seq)

        # Pad sequence
        padded_seq = torch.cat([
            seq,
            torch.zeros(max_len - seq_len, dtype=torch.long)
        ])
        input_ids.append(padded_seq)

        # Create attention mask
        mask = torch.cat([
            torch.ones(seq_len, dtype=torch.long),
            torch.zeros(max_len - seq_len, dtype=torch.long)
        ])
        attention_mask.append(mask)

        labels.append(item['label'])

    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'labels': torch.stack(labels)
    }


"""
================================================================================
QUESTION 3: Implement Multi-Worker DataLoader with Shared Memory Optimization
================================================================================

Scenario: You want to maximize data loading throughput by:
1. Using multiple worker processes
2. Optimizing memory usage with pin_memory
3. Implementing prefetching
4. Avoiding common pitfalls (worker init, RNG seeding)

Task: Implement a function that creates an optimized DataLoader configuration
and explain key parameters.
"""

class ImageDataset(Dataset):
    """Dummy image dataset for demonstration"""

    def __init__(self, size=10000, img_size=224):
        self.size = size
        self.img_size = img_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Simulate loading an image (normally would load from disk)
        image = torch.randn(3, self.img_size, self.img_size)
        label = idx % 10  # 10 classes
        return image, label


def worker_init_fn(worker_id):
    """
    Initialize each worker with different random seed.

    Important: Each worker should have different random state to avoid
    duplicate data augmentations.
    """
    # TODO: Implement worker initialization
    # 1. Set unique numpy seed per worker
    # 2. Set unique random seed per worker
    # 3. Optionally configure worker-specific resources

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    # random.seed(worker_seed)  # if using Python random module


def create_optimized_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1
) -> DataLoader:
    """
    Create an optimized DataLoader with best practices.

    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        prefetch_factor: Number of batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
        distributed: Whether to use DistributedSampler
        rank: Process rank (for distributed)
        world_size: Total processes (for distributed)

    Returns:
        Configured DataLoader
    """
    # TODO: Implement optimized DataLoader creation
    # 1. Set up sampler (DistributedSampler if distributed)
    # 2. Configure worker settings
    # 3. Enable pin_memory for GPU training
    # 4. Set prefetch_factor for better throughput
    # 5. Use persistent_workers to avoid worker respawn overhead

    sampler = None
    shuffle = True

    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        shuffle = False  # Sampler handles shuffling

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        drop_last=True  # Important for distributed training
    )

    return dataloader


"""
================================================================================
QUESTION 4: Implement Custom Sampler for Curriculum Learning
================================================================================

Scenario: You want to train a model using curriculum learning:
1. Start with easy examples (short sequences, simple patterns)
2. Gradually increase difficulty over epochs
3. Support both difficulty-based and random sampling

Task: Implement a CurriculumSampler that adjusts sampling based on training progress.
"""

class CurriculumSampler(Sampler):
    """
    Sampler that implements curriculum learning by gradually introducing
    harder examples as training progresses.

    Args:
        data_source: Dataset to sample from
        difficulty_scores: Difficulty score for each sample (higher = harder)
        num_epochs: Total number of training epochs
        curriculum_strategy: 'linear', 'exponential', or 'step'
    """

    def __init__(
        self,
        data_source: Dataset,
        difficulty_scores: List[float],
        num_epochs: int,
        curriculum_strategy: str = 'linear'
    ):
        # TODO: Implement initialization
        # 1. Sort samples by difficulty
        # 2. Initialize curriculum parameters
        # 3. Set up difficulty threshold schedule

        self.data_source = data_source
        self.difficulty_scores = np.array(difficulty_scores)
        self.num_epochs = num_epochs
        self.curriculum_strategy = curriculum_strategy
        self.current_epoch = 0

        # Sort indices by difficulty (easy to hard)
        self.sorted_indices = np.argsort(self.difficulty_scores)

    def set_epoch(self, epoch: int):
        """Update epoch for curriculum progression"""
        self.current_epoch = epoch

    def get_difficulty_threshold(self) -> float:
        """
        Calculate difficulty threshold based on current epoch.
        Returns value between 0 and 1 (percentage of dataset to use).
        """
        # TODO: Implement threshold calculation
        # 1. Linear: linearly increase from 0 to 1
        # 2. Exponential: slow start, fast end
        # 3. Step: discrete jumps in difficulty

        progress = self.current_epoch / self.num_epochs

        if self.curriculum_strategy == 'linear':
            threshold = progress
        elif self.curriculum_strategy == 'exponential':
            threshold = progress ** 2
        elif self.curriculum_strategy == 'step':
            # Discrete steps: 25%, 50%, 75%, 100%
            if progress < 0.25:
                threshold = 0.25
            elif progress < 0.5:
                threshold = 0.5
            elif progress < 0.75:
                threshold = 0.75
            else:
                threshold = 1.0
        else:
            threshold = 1.0

        return threshold

    def __iter__(self) -> Iterator[int]:
        # TODO: Implement curriculum sampling
        # 1. Calculate current difficulty threshold
        # 2. Select subset of samples below threshold
        # 3. Shuffle selected samples
        # 4. Return iterator

        threshold = self.get_difficulty_threshold()
        n_samples = int(len(self.data_source) * threshold)
        n_samples = max(n_samples, 1)  # At least 1 sample

        # Get indices of samples within difficulty threshold
        available_indices = self.sorted_indices[:n_samples]

        # Shuffle available indices
        np.random.shuffle(available_indices)

        return iter(available_indices.tolist())

    def __len__(self) -> int:
        threshold = self.get_difficulty_threshold()
        return int(len(self.data_source) * threshold)


"""
================================================================================
QUESTION 5: Implement Distributed Infinite DataLoader for Continuous Training
================================================================================

Scenario: For large-scale training, you want a DataLoader that:
1. Never stops (infinite iterations)
2. Properly handles distributed sampling
3. Supports mid-epoch checkpointing and resumption
4. Tracks exact position for reproducibility

Task: Implement an InfiniteDistributedDataLoader that can checkpoint/resume
at any point.
"""

class InfiniteDistributedSampler(Sampler):
    """
    Infinite sampler for distributed training that supports checkpointing.

    Args:
        dataset: Dataset to sample from
        rank: Process rank
        world_size: Total number of processes
        seed: Random seed
        shuffle: Whether to shuffle samples
    """

    def __init__(
        self,
        dataset: Dataset,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        seed: int = 0,
        shuffle: bool = True,
        start_iteration: int = 0
    ):
        # TODO: Implement infinite sampler initialization
        # 1. Handle distributed setup
        # 2. Calculate samples per rank
        # 3. Track current iteration for checkpointing

        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        if world_size is None:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

        # For checkpointing
        self.current_iteration = start_iteration

        # Calculate samples per rank
        self.num_samples = len(self.dataset)
        self.total_size = math.ceil(self.num_samples / self.world_size) * self.world_size
        self.num_samples_per_rank = self.total_size // self.world_size

    def __iter__(self) -> Iterator[int]:
        # TODO: Implement infinite iteration
        # 1. Generate infinite stream of indices
        # 2. Handle epoch boundaries
        # 3. Support resumption from checkpoint
        # 4. Ensure different samples each epoch

        while True:  # Infinite loop
            # Set seed for current epoch
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            if self.shuffle:
                # Shuffle indices
                indices = torch.randperm(self.num_samples, generator=g).tolist()
            else:
                indices = list(range(self.num_samples))

            # Pad to make divisible by world_size
            if len(indices) < self.total_size:
                indices += indices[:(self.total_size - len(indices))]

            # Subsample for current rank
            indices = indices[self.rank:self.total_size:self.world_size]
            assert len(indices) == self.num_samples_per_rank

            # Yield indices
            for idx in indices:
                yield idx
                self.current_iteration += 1

            # Move to next epoch
            self.epoch += 1

    def state_dict(self) -> dict:
        """Return state for checkpointing"""
        return {
            'epoch': self.epoch,
            'current_iteration': self.current_iteration,
            'seed': self.seed
        }

    def load_state_dict(self, state_dict: dict):
        """Restore state from checkpoint"""
        self.epoch = state_dict['epoch']
        self.current_iteration = state_dict['current_iteration']
        self.seed = state_dict['seed']


class CheckpointableDataLoader:
    """
    Wrapper around DataLoader that supports checkpointing.
    """

    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.iterator = None
        self.iterations_completed = 0

    def __iter__(self):
        if self.iterator is None:
            self.iterator = iter(self.dataloader)
        return self

    def __next__(self):
        if self.iterator is None:
            self.iterator = iter(self.dataloader)

        try:
            batch = next(self.iterator)
            self.iterations_completed += 1
            return batch
        except StopIteration:
            # For infinite dataloader, this shouldn't happen
            # But if it does, recreate iterator
            self.iterator = iter(self.dataloader)
            return next(self.iterator)

    def state_dict(self) -> dict:
        """Return state for checkpointing"""
        state = {
            'iterations_completed': self.iterations_completed
        }

        # Get sampler state if available
        if hasattr(self.dataloader.sampler, 'state_dict'):
            state['sampler_state'] = self.dataloader.sampler.state_dict()

        return state

    def load_state_dict(self, state_dict: dict):
        """Restore state from checkpoint"""
        self.iterations_completed = state_dict['iterations_completed']

        # Restore sampler state if available
        if 'sampler_state' in state_dict and hasattr(self.dataloader.sampler, 'load_state_dict'):
            self.dataloader.sampler.load_state_dict(state_dict['sampler_state'])

        # Fast-forward iterator to correct position
        self.iterator = iter(self.dataloader)
        # Note: In practice, you'd want to skip to the right position
        # This is a simplified version


"""
================================================================================
TEST CASES AND EXAMPLES
================================================================================
"""

def test_question1_distributed_weighted_sampler():
    """Test DistributedWeightedSampler"""
    print("\n" + "="*80)
    print("QUESTION 1: DistributedWeightedSampler Test")
    print("="*80)

    # Create dummy dataset
    dataset = list(range(100))

    # Create weights (favor higher indices)
    weights = [i + 1 for i in range(100)]

    # Test with 2 workers
    world_size = 2
    for rank in range(world_size):
        sampler = DistributedWeightedSampler(
            dataset,
            weights,
            num_samples=100,
            rank=rank,
            world_size=world_size,
            seed=42
        )

        samples = list(sampler)
        print(f"\nRank {rank}:")
        print(f"  Number of samples: {len(samples)}")
        print(f"  Sample mean: {np.mean(samples):.2f} (should be higher due to weighting)")
        print(f"  First 10 samples: {samples[:10]}")

    print("\n" + "="*80)


def test_question2_bucket_batching():
    """Test BucketBatchSampler and variable_length_collate_fn"""
    print("\n" + "="*80)
    print("QUESTION 2: Variable-Length Sequence Batching Test")
    print("="*80)

    # Create variable-length sequences
    sequences = [
        list(range(i * 5, i * 5 + np.random.randint(5, 25)))
        for i in range(50)
    ]
    labels = [i % 5 for i in range(50)]

    dataset = SequenceDataset(sequences, labels)

    # Create bucket batch sampler
    batch_sampler = BucketBatchSampler(dataset, batch_size=8)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=variable_length_collate_fn
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")

    # Test first batch
    batch = next(iter(dataloader))
    print(f"\nFirst batch:")
    print(f"  Input IDs shape: {batch['input_ids'].shape}")
    print(f"  Attention mask shape: {batch['attention_mask'].shape}")
    print(f"  Labels shape: {batch['labels'].shape}")
    print(f"  Sequence lengths in batch: {batch['attention_mask'].sum(dim=1).tolist()}")

    print("\n" + "="*80)


def test_question3_optimized_dataloader():
    """Test optimized DataLoader creation"""
    print("\n" + "="*80)
    print("QUESTION 3: Optimized DataLoader Test")
    print("="*80)

    dataset = ImageDataset(size=1000)

    dataloader = create_optimized_dataloader(
        dataset,
        batch_size=32,
        num_workers=4,
        distributed=False
    )

    print(f"DataLoader configuration:")
    print(f"  Batch size: {dataloader.batch_size}")
    print(f"  Num workers: {dataloader.num_workers}")
    print(f"  Pin memory: {dataloader.pin_memory}")
    print(f"  Prefetch factor: {dataloader.prefetch_factor}")
    print(f"  Persistent workers: {dataloader.persistent_workers}")

    # Time a few batches
    import time
    start_time = time.time()
    for i, (images, labels) in enumerate(dataloader):
        if i >= 5:
            break
    elapsed = time.time() - start_time

    print(f"\nTime to load 5 batches: {elapsed:.4f}s")
    print("\n" + "="*80)


def test_question4_curriculum_learning():
    """Test CurriculumSampler"""
    print("\n" + "="*80)
    print("QUESTION 4: Curriculum Learning Sampler Test")
    print("="*80)

    # Create dataset with difficulty scores
    dataset = list(range(100))
    difficulty_scores = list(range(100))  # 0 (easy) to 99 (hard)

    sampler = CurriculumSampler(
        dataset,
        difficulty_scores,
        num_epochs=10,
        curriculum_strategy='linear'
    )

    # Test progression over epochs
    for epoch in [0, 2, 5, 9]:
        sampler.set_epoch(epoch)
        samples = list(sampler)

        print(f"\nEpoch {epoch}:")
        print(f"  Number of samples: {len(samples)}")
        print(f"  Max difficulty: {max(samples)}")
        print(f"  Mean difficulty: {np.mean(samples):.2f}")

    print("\n" + "="*80)


def test_question5_infinite_dataloader():
    """Test InfiniteDistributedSampler"""
    print("\n" + "="*80)
    print("QUESTION 5: Infinite DataLoader Test")
    print("="*80)

    dataset = list(range(50))

    sampler = InfiniteDistributedSampler(
        dataset,
        rank=0,
        world_size=1,
        seed=42
    )

    # Create dataloader
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=8)
    checkpointable_loader = CheckpointableDataLoader(dataloader)

    # Iterate for a bit
    print("Iterating through infinite dataloader...")
    for i, batch in enumerate(checkpointable_loader):
        if i >= 10:
            break
        if i % 5 == 0:
            print(f"  Iteration {i}: batch size = {len(batch)}")

    # Test checkpointing
    state = checkpointable_loader.state_dict()
    print(f"\nCheckpoint state:")
    print(f"  Iterations completed: {state['iterations_completed']}")

    print("\n" + "="*80)


if __name__ == "__main__":
    """Run all tests"""

    print("\n" + "="*80)
    print("PyTorch DataLoader Interview Questions - Practice Set")
    print("="*80)

    # Run tests
    test_question1_distributed_weighted_sampler()
    test_question2_bucket_batching()
    test_question3_optimized_dataloader()
    test_question4_curriculum_learning()
    test_question5_infinite_dataloader()

    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)
