import cv2
import numpy as np
import torch
import torch.nn as nn
import json
import matplotlib.pyplot as plt

def imread_unicode(path, flags=cv2.IMREAD_GRAYSCALE):
    stream = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(stream, flags)
    return img

def load_image(path):
    img = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, 0)  # [1, 20, 20]
    return torch.tensor(img, dtype=torch.float32)

# 车牌图像增强预处理
def enhance_plate(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_close = np.ones((3, 3), np.uint8)
    binary_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = np.ones((3, 3), np.uint8)
    binary_open = cv2.morphologyEx(binary_close, cv2.MORPH_OPEN, kernel_open)
    kernel_dilate = np.ones((3, 3), np.uint8)
    binary_inv = cv2.dilate(binary_open, kernel_dilate, iterations=1)
    return binary_inv

# 网络结构
class MyLeNet(nn.Module):
    def __init__(self, num_classes=65):
        super(MyLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 50, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(50, 32, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.conv3 = nn.Conv2d(32, 120, kernel_size=3, stride=1)
        self.fc = nn.Linear(120 * 10 * 10, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 标签转换
match = {'yun': '云','cuan': '川','hei': '黑','zhe': '浙','ning': '宁','jin': '津','gan': '赣','hu': '沪','liao': '辽','jl': '吉','qing': '青','zang': '藏',
         'e1': '鄂','meng': '蒙','gan1': '甘','qiong': '琼','shan': '陕','min': '闽','su': '苏','xin': '新','wan': '皖','jing': '京','xiang': '湘','gui': '贵',
         'yu1': '渝','yu': '豫','ji': '冀','yue': '粤','gui1': '桂','sx': '晋','lu': '鲁'}

# 加载标签映射
with open('./output/label_map.json', 'r', encoding='utf-8') as f:
    LABEL_temp = json.load(f)
for key, value in LABEL_temp.items():
    if value in match.keys():
        LABEL_temp[key] = match[value]
print('Label:', LABEL_temp)

def segment_characters(binary_plate):
    result = []
    for col in range(binary_plate.shape[1]):
        result.append(0)
        for row in range(binary_plate.shape[0]):
            result[col] += binary_plate[row][col] / 255

    character_dict = {}
    num = 0
    i = 0
    while i < len(result):
        if result[i] == 0:
            i += 1
        else:
            index = i + 1
            while index < len(result) and result[index] != 0:
                index += 1
            character_dict[num] = [i, index-1]
            num += 1
            i = index
    return character_dict

def recognize_plate(model, LABEL_temp, license_plate_path):
    gray_plate = imread_unicode(license_plate_path, cv2.IMREAD_GRAYSCALE)
    binary_plate = enhance_plate(gray_plate)
    plt.imshow(binary_plate, cmap='gray')
    plt.show()

    character_dict = segment_characters(binary_plate)
    print("分割到的字符数：", len(character_dict))
    print("character_dict:", character_dict)

    lab = []
    for i in range(8):
        if i == 2:
            continue
        if i not in character_dict:
            print(f"Warning: character_dict does not contain index {i}, skipping.")
            continue
        width = character_dict[i][1] - character_dict[i][0]
        if width >= 170:
            start = character_dict[i][0] + (width - 170) // 2
            end = start + 170
            char_img = binary_plate[:, start:end]
        else:
            padding = (170 - width) // 2
            char_img = np.pad(binary_plate[:, character_dict[i][0]:character_dict[i][1]],
                                ((0, 0), (padding, padding)), 'constant', constant_values=(0, 0))
        ndarray = cv2.resize(char_img, (20, 20))
        cv2.imwrite(f'./{i}.png', ndarray)
        infer_img = load_image(f'./{i}.png').unsqueeze(0)
        with torch.no_grad():
            result = model(infer_img)
            pred = result.argmax(dim=1).item()
            lab.append(pred)
    print('\n车牌识别结果为：', end='')
    for i in range(len(lab)):
        print(LABEL_temp[str(lab[i])], end='')
    print()

if __name__ == '__main__':
    model = MyLeNet(num_classes=len(LABEL_temp))
    model.load_state_dict(torch.load('./output/mynet_final.pth', map_location='cpu'))
    model.eval()
    license_plate_path = './川D.png'
    recognize_plate(model, LABEL_temp, license_plate_path)