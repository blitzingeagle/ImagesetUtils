
import torch
import torch.nn as nn
import torch.legacy.nn as lnn

from functools import reduce
from torch.autograd import Variable

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))


scale2_0x_model = LambdaBase(
	nn.Conv2d(3,16,(3, 3)),
	nn.ReLU(),
	nn.Conv2d(16,32,(3, 3)),
	nn.ReLU(),
	nn.Conv2d(32,64,(3, 3)),
	nn.ReLU(),
	nn.Conv2d(64,128,(3, 3)),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3)),
	nn.ReLU(),
	nn.Conv2d(128,256,(3, 3)),
	nn.ReLU(),
	nn.ConvTranspose2d(256,3,(4, 4),(2, 2),(3, 3),(0, 0)),
	Lambda(lambda x: x.view(x.size(0),-1)), # View,
)