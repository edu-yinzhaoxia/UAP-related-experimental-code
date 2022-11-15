import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch

sys.path.append(os.path.realpath('..'))

from attacks import uap_sgd
from utils import loader_imgnet, model_imgnet, evaluate

dir_data = 'F:/tup/org/train/'
dir_uap = './'

loader = loader_imgnet(dir_data, 10000, 8) # adjust batch size as appropriate

# load model
model = model_imgnet('resnet50')

# clean accuracy
_, _, _, _, outputs, labels = evaluate(model, loader )
print('Accuracy:', sum(outputs == labels) / len(labels))

nb_epoch = 10
eps = 10 / 255
beta = 12
step_decay = 0.7
uap, losses = uap_sgd(model, loader, nb_epoch, eps, beta, step_decay)

# visualize UAP
plt.imshow(np.transpose(((uap / eps) + 1) / 2, (1, 2, 0)))

# plot loss
plt.plot(losses)

# evaluate
_, _, _, _, outputs, labels = evaluate(model, loader, uap = uap)
print('Accuracy:', sum(outputs == labels) / len(labels))


