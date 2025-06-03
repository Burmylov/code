import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 解决中文路径图片读取
def imread_unicode(path, flags=cv2.IMREAD_GRAYSCALE):
    stream = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(stream, flags)
    return img

# 车牌图像增强预处理
def enhance_plate(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_close = np.ones((3, 3), np.uint8)
    binary_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = np.ones((3,3), np.uint8)
    binary_open = cv2.morphologyEx(binary_close, cv2.MORPH_OPEN, kernel_open)
    kernel_dilate = np.ones((3,3), np.uint8)
    binary_inv = cv2.dilate(binary_open, kernel_dilate, iterations=1)
    return binary_inv

# 数据准备部分
data_path = r'C:\Users\30446\Desktop\computer vision\chepaishibie\data'
character_folders = os.listdir(data_path)
label = 0
LABEL_temp = {}
if(os.path.exists('./train_data.list')):
    os.remove('./train_data.list')
if(os.path.exists('./test_data.list')):
    os.remove('./test_data.list')

for character_folder in character_folders:
    with open('./train_data.list', 'a') as f_train, open('./test_data.list', 'a') as f_test:
        if character_folder.startswith("."):
            continue
        print(character_folder + " " + str(label))
        LABEL_temp[str(label)] = character_folder
        character_imgs = os.listdir(os.path.join(data_path, character_folder))
        for i in range(len(character_imgs)):
            if i % 10 == 0:
                f_test.write(os.path.join(os.path.join(data_path, character_folder), character_imgs[i]) + "\t" + str(label) + '\n')
            else:
                f_train.write(os.path.join(os.path.join(data_path, character_folder), character_imgs[i]) + "\t" + str(label) + '\n')
    label += 1
print('图像列表已生成')

# 保存标签映射
import json
with open('./output/label_map.json', 'w', encoding='utf-8') as f:
    json.dump(LABEL_temp, f, ensure_ascii=False, indent=4)

# 自定义Dataset
class MyDataset(Dataset):
    def __init__(self, data_list_path):
        self.data = []
        with open(data_list_path, 'r', encoding='gbk') as f:
            lines = f.readlines()
            for line in lines:
                img_path, label = line.strip().split('\t')
                self.data.append((img_path, int(label)))

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = imread_unicode(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Image decode failed: {img_path}")
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, 0)
        return torch.tensor(img, dtype=torch.float32), label

    def __len__(self):
        return len(self.data)

# 加载数据集
train_dataset = MyDataset('./train_data.list')
test_dataset = MyDataset('./test_data.list')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 定义网络结构
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyLeNet(num_classes=len(LABEL_temp)).to(device)

# 训练模型
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), torch.tensor(labels, dtype=torch.long).to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

os.makedirs('./output', exist_ok=True)
torch.save(model.state_dict(), './output/mynet_final.pth')

# 测试模型准确率
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), torch.tensor(labels, dtype=torch.long).to(device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Test Accuracy: {correct / total:.4f}")