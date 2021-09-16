import torch
t = torch.LongTensor(4, 6,8).fill_(2)

x= t[:,:-1]

print(x.size())