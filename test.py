import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# 定义图像大小和批量大小
IMG_SIZE = 128
BATCH_SIZE = 32

# 数据预处理
test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载测试集
test_dir = 'gesture_recongnition/test'
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 加载模型（定义模型结构和类别数）
NUM_CLASSES = 10  # 假设有10个类别
model = efficientnet_v2_s(weights=None)  # 使用与训练时相同的模型结构
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 加载最优权重
checkpoint_path = 'checkpoint/best_gesture_model.pth'
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# 评估模型
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 计算整体指标
accuracy = accuracy_score(all_labels, all_preds) * 100
precision = precision_score(all_labels, all_preds, average='macro') * 100
recall = recall_score(all_labels, all_preds, average='macro') * 100
print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Test Precision: {precision:.2f}%")
print(f"Test Recall: {recall:.2f}%")

# 分类报告
class_report = classification_report(all_labels, all_preds, target_names=test_dataset.classes, output_dict=True)

# 提取每个类别的精确率、召回率和 f1 分数
categories = list(class_report.keys())[:-3]  # 忽略 'accuracy', 'macro avg', 'weighted avg'
precision_per_class = [class_report[cat]['precision'] * 100 for cat in categories]
recall_per_class = [class_report[cat]['recall'] * 100 for cat in categories]
f1_per_class = [class_report[cat]['f1-score'] * 100 for cat in categories]

# 绘制每个类别的指标条形图
x = np.arange(len(categories))
width = 0.25

plt.figure(figsize=(12, 6))
plt.bar(x - width, precision_per_class, width, label='Precision')
plt.bar(x, recall_per_class, width, label='Recall')
plt.bar(x + width, f1_per_class, width, label='F1 Score')

plt.xlabel('Categories')
plt.ylabel('Percentage (%)')
plt.title('Precision, Recall, and F1 Score per Class')
plt.xticks(x, categories)
plt.legend()
plt.show()


# 读取训练历史记录
history_df = pd.read_csv("checkpoint/training_history.csv")

# 绘制损失曲线
plt.figure(figsize=(12, 5))
plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
plt.plot(history_df["epoch"], history_df["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Train and Validation Loss")
plt.show()

# 绘制准确率曲线
plt.figure(figsize=(12, 5))
plt.plot(history_df["epoch"], history_df["train_acc"], label="Train Accuracy")
plt.plot(history_df["epoch"], history_df["val_acc"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.title("Train and Validation Accuracy")
plt.show()