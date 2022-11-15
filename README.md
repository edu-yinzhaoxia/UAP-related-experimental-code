# Universal Adversarial Perturbations on PyTorch

1. 使用 uap-generation 函数 加载不同训练好的扰动模型 生成不同种类通用对抗扰动
2. 使用 AE-generation 函数 可以将生成的通用对抗扰动添加至干净样本中 生成通用对抗样本
3. 使用 Single test 函数 可以对单张对抗样本进行预测，可以更换不同模型和权重，进行不同网络的对抗攻击测试
4. 使用 Batch Test 函数 可以对批量对抗样本进行预测，可以更换不同模型和权重，进行不同网络的对抗攻击测试
5. 使用 extract 函数 可以提取两个图像之间的差值图像
6. 使用 Noise enhancement 函数 可以对图像添加 椒盐噪声 高斯噪声等噪声
7. 使用 Extraction section 函数 可以将从图像按照固定尺寸固定位置，提取出局部图像
8. 使用 Self-flipping 函数 可以将图像自翻转
9. 使用 psnr 函数 衡量两个数据集对应各自图像的PSNR或SSIM值


实验1. 
