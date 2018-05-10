
import numpy as np
import cv2

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
    None,
    nn.Conv2d(3, 16, (3, 3), padding=(1,1)),
    nn.LeakyReLU(0.1),
    nn.Conv2d(16, 32, (3, 3), padding=(1,1)),
    nn.LeakyReLU(0.1),
    nn.Conv2d(32, 64, (3, 3), padding=(1,1)),
    nn.LeakyReLU(0.1),
    nn.Conv2d(64, 128, (3, 3), padding=(1,1)),
    nn.LeakyReLU(0.1),
    nn.Conv2d(128, 128, (3, 3), padding=(1,1)),
    nn.LeakyReLU(0.1),
    nn.Conv2d(128, 256, (3, 3), padding=(1,1)),
    nn.LeakyReLU(0.1),
    nn.ConvTranspose2d(256, 3, (4, 4), (2, 2), (3, 3), (0, 0)),
    Lambda(lambda x: x),  # View,
)
scale2_0x_model._modules['12'].bias.data = torch.zeros([3], dtype=torch.float32)

img = cv2.imread("test_input.png")   # read image as color
# img = img.astype(float) / 256   # normalize values from byte to float
# img = np.array([[0.0, 0.0, 1.0] if x < 5000 else [0.0, 0.0, 0.0] for x in range(10000)]).reshape(100, 100, 3)
# img = np.array([[1.0, 1.0, 1.0] if x%100 < 50 else [0.0, 0.0, 0.0] for x in range(10000)]).reshape(100, 100, 3)
# img = np.array([[0.0, 0.0, 1.0] if x//100%2==0 else [0.0, 0.0, 0.0] for x in range(10000)]).reshape(100, 100, 3)
# img = np.array([1 if x%3==0 else 1 for x in range(1200)]).reshape(20, 20, 3)
cv2.imshow("input", img)

N = 1

torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda")
model = scale2_0x_model.to(device)

# load weights from file into model
weights = torch.load("scale2.0x_model.pth")
model.load_state_dict(weights)

model.eval()

# data = torch.ones(N, 1, 32, 32, dtype=torch.float)
shape = (N, 1 if len(img.shape) < 3 else img.shape[2]) + img.shape[:2]
data = torch.from_numpy(img.reshape(shape)).float().cuda()
print(type(data), data.shape)

print(type(model), model)

output = model(data)
print(output.shape, output)

def get_image(arr):
    arr = arr.flatten()
    ret = []
    for i in range(arr.size):
        if i % 6 < 2:
            ret.append([arr[i]])
    return np.array(ret)

img2 = output.cpu().detach().numpy()
# img2 = get_image(img2).reshape(196, 196, 1)
# print(img2.shape)
#img2 = img2.reshape((img.shape[2], 2*(img.shape[0]-2), 2*(img.shape[1]-2)))
img2 = img2[0,:,:,:]/255.
img2 = np.swapaxes(img2,0,2)
img2 = np.swapaxes(img2,0,1)
# print(img2.shape, img2)
cv2.imshow("output", img2)
cv2.waitKey()
cv2.destroyAllWindows()

def copy_param(m,n):
    if m.weight is not None:
        if type(m.weight).__name__ == "ndarray": n.weight.data.copy_(torch.from_numpy(np.reshape(m.weight, tuple(n.weight.shape))))
        else: n.weight.data.copy_(m.weight)
    if m.bias is not None:
        if type(m.bias).__name__ == "ndarray": n.bias.data.copy_(torch.from_numpy(m.bias))
        else: n.bias.data.copy_(m.bias)
    if hasattr(n,'running_mean'): n.running_mean.copy_(m.running_mean)
    if hasattr(n,'running_var'): n.running_var.copy_(m.running_var)

print("\n"*5)

seq_model = nn.Sequential(
    nn.Conv2d(3, 16, (3, 3), padding=(1, 1)),
    nn.ReLU(),
    nn.Conv2d(16, 32, (3, 3), padding=(1, 1)),
    nn.ReLU(),
    nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),
    nn.ReLU(),
    nn.Conv2d(64, 128, (3, 3), padding=(1, 1)),
    nn.ReLU(),
    nn.Conv2d(128, 128, (3, 3), padding=(1, 1)),
    nn.ReLU(),
    nn.Conv2d(128, 256, (3, 3), padding=(1, 1)),
    nn.ReLU(),
    nn.ConvTranspose2d(256, 3, (4, 4), (2, 2), (3, 3), (0, 0))
).to(device)

for key in model._modules.keys():
    if key in seq_model._modules and type(seq_model._modules[key]).__name__ != "ReLU":
        copy_param(model._modules[key], seq_model._modules[key])

seq_output = seq_model(data)
print(seq_output, seq_output.shape)


