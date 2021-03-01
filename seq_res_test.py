import torch 
import fairscale
from torchsummary import summary

from seq_res_101 import resnet101


model = resnet101()
print(len(model))
model = fairscale.nn.Pipe(model, balance=[100, 100, 104], devices=[0, 1, 2], chunks=8)

print("Finished Training Step")
#summary(model, input_size=(3, 64, 64))
del model