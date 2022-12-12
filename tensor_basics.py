import torch
import numpy as np

# empty tensors
x_1d = torch.empty(3)
x_2d = x = torch.empty(2,3)
x_3d = x = torch.empty(2,2,3)
# print(x_1d,x_2d)

# random tensors
x_random = torch.rand(2,2)
y_random = torch.rand(2,2)
# print(x_random)
# element-wise addition
ele_add = x_random + y_random # ele_add = torch.add(x_random, y_random)
# print(ele_add)
# operations
print(x_random[:1])
print(x_random[1,1])

# convert to n-dim 
x_new = torch.rand(4,4)
con_dim = x_new.view(16)

# zeros
x_zeros = torch.zeros(2,2)

# ones
x_ones = torch.ones(2,2,dtype=torch.int)

# tensor from a list
tensor_list = torch.tensor([2.1, 3.4,0.7])

# tensor from numpy
a_np = np.ones(5)
tensor_from_np = torch.from_numpy(a_np)

