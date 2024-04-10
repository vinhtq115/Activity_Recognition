import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

t = torch.tensor([1,2], device=device)