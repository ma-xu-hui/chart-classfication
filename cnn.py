import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# 配置
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

TRAIN_DIR = "TRAIN"  # 替换为训练集路径
TEST_DIR = "TEST"  # 替换为测试集路径

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((448, 448)),  # 调整图像大小为 448x448
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])


# 自定义Dataset类
class TrainDataset(Dataset):
    def __init__(self, train_dir, transform=None):
        self.transform = transform
        self.image_files = []
        self.labels = []

        # 遍历子文件夹
        for label_name in os.listdir(os.path.join(train_dir, 'images')):
            label_path = os.path.join(train_dir, 'images', label_name)
            if os.path.isdir(label_path):
                for image_name in os.listdir(label_path):
                    self.image_files.append(os.path.join(label_path, image_name))
                    self.labels.append(label_name)

        # 映射标签到整数
        self.label_mapping = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}
        self.labels = [self.label_mapping[label] for label in self.labels]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.image_dir = os.path.join(test_dir, 'images')
        self.label_dir = os.path.join(test_dir, 'labels')
        self.transform = transform
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label_file = os.path.join(self.label_dir, os.path.splitext(image_file)[0] + ".json")
        with open(label_file, 'r') as f:
            label_data = json.load(f)
        chart_type = label_data["task1"]["output"]["chart_type"]

        label_mapping = {
            "area": 0, "heatmap": 1, "horizontal_bar": 2, "horizontal_interval": 3, "line": 4, "manhattan": 5,
            "map": 6, "pie": 7, "scatter": 8, "scatter-line": 9, "surface": 10, "venn": 11,
            "vertical_bar": 12, "vertical_box": 13, "vertical_interval": 14
        }
        label = label_mapping.get(chart_type, -1)  # 默认值为-1表示未知类别

        return image, label


# 创建Dataset和DataLoader
train_dataset = TrainDataset(TRAIN_DIR, transform=transform)
print(f"Number of training samples: {len(train_dataset)}")
test_dataset = TestDataset(TEST_DIR, transform=transform)
print(f"Number of testing samples: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# 定义CNN模型
class ChartClassifierCNN(nn.Module):
    def __init__(self, num_classes):
        super(ChartClassifierCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 28 * 28, 1024),  # 修改输入大小为 256 * 28 * 28
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


num_classes = len(train_dataset.label_mapping)  # 根据训练集标签数量动态调整
model = ChartClassifierCNN(num_classes).to(DEVICE)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# 训练模型
def train_model():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 每10个batch打印一次进度
            if (batch_idx + 1) % 50 == 0:
                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}] Completed, Average Loss: {avg_loss:.4f}")


# 测试模型
def test_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 打印每个batch的测试结果
            print(
                f"Test Batch [{batch_idx + 1}/{len(test_loader)}], Batch Accuracy: {(predicted == labels).sum().item() / labels.size(0) * 100:.2f}%")

    accuracy = 100 * correct / total
    print(f"Overall Test Accuracy: {accuracy:.2f}%")


# 执行训练和测试
train_model()
test_model()
