import torch 
import torch.nn as nn


criterion = nn.CrossEntropyLoss()


output = torch.rand([3,5])
label = torch.tensor([0,0,1])
#label = torch.randint(0,1,(3,))
loss = criterion(output, label)
print("loss:", label.dtype, loss)
