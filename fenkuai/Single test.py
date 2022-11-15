import os
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2 
from model import resnet50, vgg

import urllib.request

if os.path.exists("vgg16Net.pth") == False:
    urllib.request.urlretrieve(
        'https://download.pytorch.org/models/vgg16-397923af.pth', "vgg16Net.pth")
elif os.path.exists("vgg19Net.pth") == False:
    urllib.request.urlretrieve(
        'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth', "vgg19Net.pth")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor()#,
        #  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor()])

    # read class_indict
    json_path = './imagenet_class_index.json'
    assert os.path.exists(
        json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    model = resnet50().to(device)
    # load model weights
    weights_path = "./resnet50-pre.pth"
    assert os.path.exists(
        weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # # load image
    # count = 0
    # sum = 0
    # img_path = "n01440764_10029.png"
    # #img_path = "F:/coding/CW (2)/AE_PGD100/21/43.png"

    # assert os.path.exists(
    #     img_path), "file: '{}' dose not exist.".format(img_path)
    path = "sk1.png"
    img = cv2.imread(path)

    img = Image.fromarray(np.uint8(img))
 
    
    img = data_transform(img)

    img = torch.unsqueeze(img, dim=0)

    model.eval()
    with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device)).cpu())
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

    print_res = "pre class: {}   prob: {:.4}%".format(
        class_indict[str(predict_cla)][1], 100*predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)


if __name__ == '__main__':
    main()
