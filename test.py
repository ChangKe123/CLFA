import torch
def avg_pooling(features, mask):
	return torch.sum(features*(mask.unsqueeze(2)),dim=1)/torch.sum(mask, dim=1,keepdim=True)
a=torch.tensor([[[3,2,1],[1,0,4]], [[3,2,1],[1,0,5]]], dtype=torch.float)
b=torch.tensor([[0,1],[1,0]])
print(avg_pooling(a,b))
