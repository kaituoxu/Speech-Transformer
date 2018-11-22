# INSTALL
# $ pip install visdom
# START
# $ visdom
# or 
# $ python -m visdom.server
# open browser and visit http://localhost:8097

import torch
import visdom

vis = visdom.Visdom(env="model_1")
vis.text('Hello, world', win='text1')
vis.text('Hi, Kaituo', win='text1', append=True)
for i in range(10):
    vis.line(X=torch.FloatTensor([i]), Y=torch.FloatTensor([i**2]), win='loss', update='append' if i> 0 else None)


epochs = 20
loss_result = torch.Tensor(epochs)
for i in range(epochs):
	loss_result[i] = i ** 2
opts = dict(title='LAS', ylabel='loss', xlabel='epoch')
x_axis = torch.arange(1, epochs+1)
y_axis = loss_result[:epochs]
vis2 = visdom.Visdom(env="view_loss")
vis2.line(X=x_axis, Y=y_axis, opts=opts)


while True:
	continue
