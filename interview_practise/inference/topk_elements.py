import torch
from collections import defaultdict

# Example data (ignoring unused N,C,G=12,6,4; using the small randn for illustration)
# In practice, replace with your actual tensors
groups = torch.arange(3)  # [0, 1, 2]
scores = torch.randn(3, 4)
k = 5

values = scores.max(dim=1).values
out = defaultdict(list)
unique_groups = torch.unique(groups)

for g in unique_groups:
    mask = (groups == g)
    group_values = values[mask]
    group_size = mask.sum().item()
    if group_size == 0:
        continue
    k_ = min(k, group_size)
    topk_val, _ = torch.topk(group_values, k_, largest=True, sorted=True)
    out[g.item()] = topk_val.tolist()

# Example output (will vary due to randn; run to see)
print(out)