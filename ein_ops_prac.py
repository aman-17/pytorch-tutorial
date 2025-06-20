import einops
import torch
import numpy as np
pred = torch.randn(size=(1, 3*85, 13, 13))
new_pred = einops.rearrange(pred, 'b (n c) h w -> b (h w) (n c)', n=3, c=85)
print(new_pred.shape)

w = 13
pred1 = torch.arange(w, dtype=torch.float32)
# new_pred1 = einops.rearrange(pred1, '1 1 w 1 -> 1 1 1 w')
new_pred1 = einops.rearrange(pred1, 'w -> 1 1 w 1')
new_pred1 = einops.rearrange(new_pred1, '1 1 w 1 -> 1 1 1 w 1')
print(new_pred1.shape)

p2 = np.random.randn(1, 3*85, 13, 13, 17)
# p2_new = np.einops.rearrange(p2, 'b (n c) h w -> b (n c) h', n=3, c=85)
p2_new = einops.rearrange(p2, 'b (n c) h w o -> b (h w) (n c) o 1', n=3, c=85, o=17)
print(p2_new.shape)
p2 = np.squeeze(p2,0)
print(p2.shape)

