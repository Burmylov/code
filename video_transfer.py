from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import time

import torch
import torch.optim as optim
import requests
from torchvision import transforms, models
import cv2

# -----------------------------
# 配置与模型加载
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
for param in vgg.parameters():
    param.requires_grad_(False)
vgg.to(device)


# -----------------------------
# 图像加载与预处理（保持原始尺寸）
# -----------------------------

def load_image(img, max_size=None):
    """ 加载图像并保持原始尺寸 """
    if isinstance(img, str):
        if "http" in img:
            response = requests.get(img)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(img).convert('RGB')
    elif isinstance(img, Image.Image):
        image = img.convert('RGB')
    else:
        raise TypeError("输入必须是路径字符串或PIL图像对象")

    # 仅当指定max_size时调整尺寸
    if max_size and max(image.size) > max_size:
        size = max_size
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])
    else:
        # 保持原始尺寸
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])

    image = transform(image)[:3, :, :].unsqueeze(0)
    return image


# -----------------------------
# 图像转换工具
# -----------------------------

def im_convert(tensor):
    """ 将张量转换为可显示的图像 """
    image = tensor.cpu().clone().detach().numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = np.clip(image, 0, 1)
    return image


# -----------------------------
# 特征提取与Gram矩阵（保持不变）
# -----------------------------

def get_features(image, model, layers=None):
    """ 提取指定层的特征 """
    if layers is None:
        layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1',
                  '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features


def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    return torch.mm(tensor, tensor.t())


# -----------------------------
# 单帧风格迁移（优化梯度问题）
# -----------------------------

def single_frame_style_transfer(content_tensor, style_tensor, steps=50):
    content_features = get_features(content_tensor, vgg)
    style_features = get_features(style_tensor, vgg)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    target = content_tensor.clone().requires_grad_(True).to(device)
    optimizer = optim.Adam([target], lr=0.003)

    style_weights = {'conv1_1': 1.0, 'conv2_1': 0.75, 'conv3_1': 0.2,
                     'conv4_1': 0.2, 'conv5_1': 0.2}
    content_weight = 1
    style_weight = 1e6

    for step in range(steps):
        optimizer.zero_grad()
        target_features = get_features(target, vgg)

        # 内容损失
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

        # 风格损失
        style_loss = 0
        for layer in style_weights:
            target_gram = gram_matrix(target_features[layer])
            style_gram = style_grams[layer]
            layer_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
            style_loss += layer_loss / (target_features[layer].shape[1] *
                                        target_features[layer].shape[2] *
                                        target_features[layer].shape[3])

        total_loss = content_weight * content_loss + style_weight * style_loss
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            target.data.clamp_(0, 1)

    return target.detach()


# -----------------------------
# 视频处理主函数（关键修复）
# -----------------------------

def video_style_transfer(video_path, output_path, style_path, frame_limit=30):
    start_time = time.time()

    # 加载风格图像（保持原始尺寸）
    style_image = load_image(style_path, max_size=None).to(device)

    # 视频输入设置
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError("无法打开输入视频")

    # 获取原始视频参数
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 使用更兼容的编码器
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # 或替换为 'XVID' 生成AVI
    out = cv2.VideoWriter(output_path, fourcc, fps, (orig_width, orig_height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (frame_limit > 0 and frame_count >= frame_limit):
            break

        # 转换为PIL格式（保持原始尺寸）
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb_frame)

        # 加载内容图像（保持原始尺寸）
        content_tensor = load_image(pil_frame, max_size=None).to(device)

        # 风格迁移
        styled_tensor = single_frame_style_transfer(content_tensor, style_image, steps=50)

        # 转换为OpenCV格式并调整尺寸
        styled_image = im_convert(styled_tensor)
        styled_image = (styled_image * 255).astype(np.uint8)

        # 关键修复：强制调整到原始视频尺寸
        styled_image = cv2.resize(styled_image, (orig_width, orig_height))
        bgr_output = cv2.cvtColor(styled_image, cv2.COLOR_RGB2BGR)

        # 写入视频
        out.write(bgr_output)
        frame_count += 1
        print(f"已处理 {frame_count} 帧 | 耗时: {time.time() - start_time:.2f}s")

    cap.release()
    out.release()
    print(f"处理完成！输出视频尺寸: {orig_width}x{orig_height} | 路径: {output_path}")


# -----------------------------
# 主程序
# -----------------------------
if __name__ == "__main__":
    # 使用示例
    video_style_transfer(
        video_path="video/input.mp4",
        output_path="output/video_output.mp4",
        style_path="imgs/starry_night.jpg"
    )