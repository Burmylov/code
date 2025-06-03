from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import time

import torch
import torch.optim as optim
import requests
from torchvision import transforms, models

# -----------------------------
# 配置与模型加载
# -----------------------------

# 记录开始时间
start_time = time.time()


vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

# 冻结 VGG 模型参数，防止训练过程中更新
for param in vgg.parameters():
    param.requires_grad_(False)

# 设置设备：优先使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
vgg.to(device)

# -----------------------------
# 图像加载与预处理函数
# -----------------------------

def load_image(img_path, max_size=400, shape=None):
    ''' 加载图像并进行标准化处理 '''
    if "http" in img_path:
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(img_path).convert('RGB')

    # 调整尺寸以加快计算速度
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # 去掉 alpha 通道，并添加 batch 维度
    image = transform(image)[:3, :, :].unsqueeze(0)
    return image


# -----------------------------
# 图像显示与转换函数
# -----------------------------

def im_convert(tensor):
    """ 将张量转为可显示的图像 """
    image = tensor.cpu().clone().detach().numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = np.clip(image, 0, 1)
    return image


# -----------------------------
# 特征提取与 Gram 矩阵
# -----------------------------

def get_features(image, model, layers=None):
    """ 提取指定层的特征 """
    if layers is None:
        layers = {
            '0': 'conv1_1', #浅层
            '5': 'conv2_1',
            '10': 'conv3_1', #中层
            '19': 'conv4_1',
            '21': 'conv4_2',  #深层
            '28': 'conv5_1'
        }

    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features


def gram_matrix(tensor):
    """ 计算 Gram 矩阵 """
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram


# -----------------------------
# 加载内容与风格图像
# -----------------------------

content = load_image('imgs/content.png').to(device)  #要迁移的内容图像
style = load_image('imgs/starry_night.jpg', shape=content.shape[-2:]).to(device)  #风格图像

# 获取内容与风格图像的特征
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# 计算每个风格层的 Gram 矩阵
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# 初始化目标图像为内容图像的副本，并设置 requires_grad=True
target = content.clone().requires_grad_(True).to(device)

# -----------------------------
# 损失权重设置
# -----------------------------

style_weights = {
    'conv1_1': 1.,
    'conv2_1': 0.75,
    'conv3_1': 0.2,
    'conv4_1': 0.2,
    'conv5_1': 0.2
}

content_weight = 1     # alpha
style_weight = 1e6     # beta

# -----------------------------
# 优化器设置
# -----------------------------

optimizer = optim.Adam([target], lr=0.003)
steps = 2000  # 总迭代次数

# -----------------------------
# 主循环：风格迁移
# -----------------------------

try:
    print("开始风格迁移...")

    for ii in range(1, steps + 1):
        target_features = get_features(target, vgg)

        # 内容损失
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

        # 风格损失
        style_loss = 0
        for layer in style_weights:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            _, d, h, w = target_feature.shape
            style_gram = style_grams[layer]
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
            style_loss += layer_style_loss / (d * h * w)

        # 总损失
        total_loss = content_weight * content_loss + style_weight * style_loss

        # 更新目标图像
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 打印训练信息
        if ii % 50 == 0:
            print(f"Step [{ii}/{steps}], Loss: {total_loss.item():.4f}")



except KeyboardInterrupt:
    print("\n训练被用户中断。正在保存当前结果...")
except Exception as e:
    print(f"\n发生错误：{str(e)}")
finally:
    # 无论如何都尝试保存当前图像
    final_image = im_convert(target)
    output_path = 'output/imgs_output.jpg'
    plt.imsave(output_path, final_image)
    print(f"最终图像已保存至：{output_path}")

    end_time = time.time()
    print(f"总耗时：{(end_time - start_time):.2f} 秒")