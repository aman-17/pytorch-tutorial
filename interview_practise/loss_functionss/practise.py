import torch

class HuberLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, y_test, y_pred, delta=1.0):
        loss = torch.where(torch.abs(y_test - y_pred) <= delta, 0.5 * (y_test - y_pred) ** 2, delta * (torch.abs(y_test - y_pred) - 0.5 * (delta**2)))
        return loss
    
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, y, x1, x2, m):
        d = torch.norm(x1-x2)
        loss = (1-y)*(d**2)+y*(torch.max(torch.tensor(0), m-d)**2)
        return loss
    
class MSE(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, n, y_test, y_pred):
        loss = 1/n * torch.sum((y_test - y_pred)**2)
        return loss

y_test = torch.tensor([3.0, -0.5, 2.0, 7.0])
y_pred = torch.tensor([2.5, 0.0, 2.1, 7.8])
m  = 4.0
y=2
x = ContrastiveLoss()
print(x(y, y_test, y_pred, m))

