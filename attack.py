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

loader = loader_imgnet(dir_data, 50000, 1) # adjust batch size as appropriate

# load model
model = model_imgnet('resnet50')

# clean accuracy
_, _, _, _, outputs, labels = evaluate(model, loader )
print('Accuracy:', sum(outputs == labels) / len(labels))

nb_epoch = 10
eps = 10 / 255
y_target = 815
beta = 12
step_decay = 0.6
uap= uap_sgd(model, loader, nb_epoch, eps, beta, step_decay, y_target = y_target)

# visualize UAP
plt.axis('off')
plt.imshow(np.transpose(((uap / eps) + 1) / 2, (1, 2, 0)))
plt.savefig('uap.png', format='png', bbox_inches='tight')
# evaluate
_, _, _, _, outputs, labels = evaluate(model, loader, uap = uap)
print('Accuracy:', sum(outputs == labels) / len(labels))
print('Targeted success rate:', sum(outputs == y_target) / len(labels))