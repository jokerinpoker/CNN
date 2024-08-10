pip install torch torchvision torchaudio

import torch

print("CUDA Available: ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Device Name: ", torch.cuda.get_device_name(0))
    print("CUDA Device Count: ", torch.cuda.device_count())

    # 导入必要的库
import argparse
import copy
import json
import os
import random
from matplotlib.image import imread

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# 确保安装了torch和torchvision
# !pip install torch
# !pip install torchvision

# 定义常量
BATCH_SIZE = 2
TEST_BATCH_SIZE = 5
EPOCHS = 1
LEARNING_RATE = 0.01
NO_CUDA = False  # 如果使用GPU，则设置为False
DRY_RUN = False
SEED = random.randint(1, 1000)  # 随机种子，如果希望在相同数据上训练，请设置为固定值
LOG_INTERVAL = 100  # 每多少批次输出一次训练状态
SAVE_MODEL = False  # 是否保存模型

# 设置设备
torch.manual_seed(SEED)
use_cuda = not NO_CUDA and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# 数据路径
DATASET_PATH = "/courses/CS5330.202450/students/zhao.wei2/Final/Final"
TRAIN_IMAGES_DIR = os.path.join(DATASET_PATH, "images/train")
VAL_IMAGES_DIR = os.path.join(DATASET_PATH, "images/val")
TEST_IMAGES_DIR = os.path.join(DATASET_PATH, "images/test")
TRAIN_ANNOTATIONS = os.path.join(DATASET_PATH, "/courses/CS5330.202450/students/zhao.wei2/Final/Final/annotations/instances_train.json")
VAL_ANNOTATIONS = os.path.join(DATASET_PATH, "/courses/CS5330.202450/students/zhao.wei2/Final/Final/annotations/instances_val.json")

# 图像变换
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.2404, 0.2967, 0.3563], [0.0547, 0.0527, 0.0477])
])

print("Train Annotations Path Exists:", os.path.exists(TRAIN_ANNOTATIONS))
print("Validation Annotations Path Exists:", os.path.exists(VAL_ANNOTATIONS))
print("Train Images Path Exists:", os.path.exists(TRAIN_IMAGES_DIR))
print("Validation Images Path Exists:", os.path.exists(VAL_IMAGES_DIR))
print("Test Images Path Exists:", os.path.exists(TEST_IMAGES_DIR))


import cv2
import torch

class LookoutDataset(Dataset):
    def __init__(self, root_dir, transform=None, gt_json_path=''):
        """
        初始化数据集
        :param root_dir: 图像文件所在的根目录
        :param transform: 图像变换（如有）
        :param gt_json_path: 标注文件路径
        """
        self.root_dir = root_dir
        self.transform = transform
        self.gt_json_path = gt_json_path
        self.labels = self.load_annotations(gt_json_path)
        self.image_list = sorted(os.listdir(root_dir))
        self.image_ids = {img_name: idx for idx, img_name in enumerate(self.image_list)}

    def load_annotations(self, json_path):
        """
        加载标注数据
        :param json_path: 标注文件路径
        :return: 标注数据字典
        """
        try:
            with open(json_path, 'r') as f:
                annotations = json.load(f)
            return annotations
        except FileNotFoundError:
            print(f"Error: File {json_path} not found.")
            return {}
        except json.JSONDecodeError:
            print(f"Error: File {json_path} is not a valid JSON file.")
            return {}

    def __len__(self):
        """
        返回数据集大小
        :return: 数据集大小
        """
        return len(self.image_list)

    def __getitem__(self, idx):
        """
        获取索引对应的图像和标签
        :param idx: 图像索引
        :return: 图像和标签
        """
        img_name = self.image_list[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        # 使用 OpenCV 加载图像并调整大小
        image = cv2.imread(img_path)
        image = cv2.resize(image, (640, 640))  # 将图像调整为 640x640
        
        if self.transform:
            # OpenCV 读取的图像是 BGR 形式，需要转换为 RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transform(image)

         # 确保标签的默认值是0，避免空值，并转换为张量
        label = self.labels.get(img_name, [0, 0, 0, 0])  # 使用适当的默认值（例如：[0, 0, 0, 0]表示没有检测到物体）
        label = torch.tensor(label, dtype=torch.float32)  # 转换为张量

        sample = (image, label)
        
        return sample

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 第一个卷积层 + 批归一化 + 池化层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 第二个卷积层 + 批归一化 + 池化层
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 第三个卷积层 + 批归一化 + 池化层
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 第四个卷积层 + 批归一化 + 池化层
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Dropout层
        self.dropout = nn.Dropout(0.5)
        
        # 自动计算全连接层的输入维度
        self._calculate_fc_input_dim()

        # 全连接层
        self.fc1 = nn.Linear(self.fc_input_dim, 1024)
        self.bnfc1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.bnfc2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 4)

    def _calculate_fc_input_dim(self):
        # 使用一个虚拟的输入来推断卷积层输出的大小
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 640, 640)
            dummy_output = self._forward_conv(dummy_input)
            self.fc_input_dim = dummy_output.numel()

    def _forward_conv(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        return x

    def forward(self, x):
        # 卷积层和池化层
        x = self._forward_conv(x)
        
        # 扁平化
        x = torch.flatten(x, start_dim=1)
        
        # 全连接层
        x = self.dropout(F.relu(self.bnfc1(self.fc1(x))))
        x = F.relu(self.bnfc2(self.fc2(x)))
        output = torch.sigmoid(self.fc3(x))
        return output

def train(log_interval, model, device, train_loader, optimizer, criterion, epoch, dry_run):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).float()
        
        # 打印数据和标签的形状，确认它们被正确地移动到设备上
        if batch_idx == 0:
            print(f"Batch {batch_idx}: Data shape: {data.shape}, Target shape: {target.shape}")
            print(f"Data device: {data.device}, Target device: {target.device}")
        
        optimizer.zero_grad()
        output = model(data)
        
        # 打印模型输出的形状，确认它们与标签形状匹配
        if batch_idx == 0:
            print(f"Output shape: {output.shape}")
        
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            if dry_run:
                break

def test(model, device, test_loader, criterion):
    """
    评估神经网络模型
    :param model: 神经网络模型
    :param device: 设备（CPU或GPU）
    :param test_loader: 测试数据加载器
    :param criterion: 损失函数
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).float()  # 确保标签形状为 [batch_size, 4]
            output = model(data)
            test_loss += criterion(output, target).item()  # 累积批量损失
            pred = torch.round(output)  # 获取预测结果
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}\n')
    return test_loss
import json

def save_detections(detections, output_file):
    """
    将检测结果保存到文件
    :param detections: 检测结果
    :param output_file: 输出文件路径
    """
    with open(output_file, 'w') as f:
        json.dump(detections, f)

def notify_other_boats(detections):
    """
    通知其他船只检测到的危险物体信息
    :param detections: 检测结果
    """
    # 这里可以实现具体的通知逻辑，例如通过网络请求发送数据
    for detection in detections:
        print(f"Notify: Detected {detection['category']} at {detection['bbox']} with confidence {detection['confidence']}")

def detect_and_store(model, device, test_loader, output_file):
    model.eval()
    detections = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).float()
            output = model(data)
            for i, out in enumerate(output):
                pred = torch.round(out).item()
                if pred > 0.5:  # 假设阈值为0.5
                    bbox = target[i].tolist()
                    detection = {
                        'category': 'hazardous_object',
                        'bbox': bbox,
                        'confidence': pred
                    }
                    detections.append(detection)

    # 保存检测结果并通知其他船只
    save_detections(detections, output_file)
    notify_other_boats(detections)

# 更新main函数以调用新的检测和存储功能
def main():
    # 设置训练参数
    batch_size = BATCH_SIZE
    test_batch_size = TEST_BATCH_SIZE
    epochs = EPOCHS
    learning_rate = LEARNING_RATE
    no_cuda = NO_CUDA
    dry_run = DRY_RUN
    seed = SEED
    log_interval = LOG_INTERVAL
    save_model = SAVE_MODEL
    
    # 设置设备
    torch.manual_seed(seed)
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")  # 打印正在使用的设备


    # 设置数据加载参数
    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    val_kwargs = {'batch_size': test_batch_size}  # 初始化 val_kwargs
    
    if use_cuda:
        cuda_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        val_kwargs.update(cuda_kwargs)


    # 加载数据集
    train_dataset = LookoutDataset(root_dir=TRAIN_IMAGES_DIR, transform=TRANSFORM, gt_json_path=TRAIN_ANNOTATIONS)
    val_dataset = LookoutDataset(root_dir=VAL_IMAGES_DIR, transform=TRANSFORM, gt_json_path=VAL_ANNOTATIONS)

    train_loader = DataLoader(train_dataset, **train_kwargs)
    test_loader = DataLoader(val_dataset, **test_kwargs)

    # 初始化模型、优化器和损失函数
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # 原损失函数：criterion = nn.MSELoss()
    criterion = nn.BCEWithLogitsLoss()  # 或者 CrossEntropyLoss

    # 训练和评估模型
    best_acc = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(1, epochs + 1):
        train(log_interval, model, device, train_loader, optimizer, criterion, epoch, dry_run)
        acc = test(model, device, test_loader, criterion)
        if acc > best_acc:
            best_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    print(f"Best accuracy (val): {best_acc:.2f}%")

    # 保存模型
    if save_model:
        torch.save(model.state_dict(), "model.pth")

    # 导出为ONNX格式
    dummy_input = torch.randn(1, 3, 108, 192, device=device)
    input_names = ["input"]
    output_names = ["output"]
    torch.onnx.export(model, dummy_input, "model.onnx", input_names=input_names, output_names=output_names)

    # 检测并存储检测结果
    detect_and_store(model, device, test_loader, "detections.json")

if __name__ == '__main__':
    main()