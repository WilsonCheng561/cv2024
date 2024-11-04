import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import pandas as pd
import torch.optim as optim

# 定义图像大小和批次大小
IMG_SIZE = 128
BATCH_SIZE = 32
NUM_CLASSES = 10  # 有10个手势类别

# 数据增强和预处理
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
validation_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# 数据目录
data_dir = './gesture_recongnition'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

# 加载数据集
# ImageFolder 自动使用文件夹名称作为标签
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=validation_transforms)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")


# 加载EfficientNetV2模型
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

# 模型超参数
NUM_CLASSES = 10  # 手势类别数量
EPOCHS = 100
PATIENCE = 3  # Early stopping 的 patience

# 使用预训练的 EfficientNetV2 模型
weights = EfficientNet_V2_S_Weights.DEFAULT
model = efficientnet_v2_s(weights=weights)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 使用 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 记录每个 epoch 的训练和验证指标
history = {
    "epoch": [],
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

# Early stopping 的初始条件
best_val_loss = float("inf")
best_val_acc = 0
early_stop_count = 0

# 创建 checkpoint 目录
os.makedirs("checkpoint", exist_ok=True)

# 训练循环
for epoch in range(EPOCHS):
    # 训练阶段
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / total
    train_acc = 100 * correct / total

    # 验证阶段
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss /= total
    val_acc = 100 * correct / total

    # 打印当前 epoch 的结果
    print(f"Epoch [{epoch + 1}/{EPOCHS}], "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # 保存每个 epoch 的训练结果到历史记录
    history["epoch"].append(epoch + 1)
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    # 每 10 个 epoch 保存一次模型
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"checkpoint/gesture_model_epoch_{epoch + 1}.pth")

    # Early stopping 检查
    if val_loss < best_val_loss or val_acc > best_val_acc:
        best_val_loss = val_loss
        best_val_acc = val_acc
        early_stop_count = 0  # 重置计数器
        # 保存最佳模型
        torch.save(model.state_dict(), "checkpoint/best_gesture_model.pth")
        print("Validation loss improved, best model saved.")
    else:
        early_stop_count += 1
        print(f"No improvement. Early stop count: {early_stop_count}/{PATIENCE}")

    if early_stop_count >= PATIENCE:
        print("Early stopping triggered.")
        break

# 将训练过程保存为 CSV 文件
history_df = pd.DataFrame(history)
history_df.to_csv("checkpoint/training_history.csv", index=False)
print("Training history saved to checkpoint/training_history.csv")