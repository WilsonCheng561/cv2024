import os
import shutil
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# 定义数据路径
data_dir = 'train_gesture_data'
train_dir = 'train'
val_dir = 'val'
test_dir = 'test'

# 创建保存子集的文件夹
for split in [train_dir, val_dir, test_dir]:
    os.makedirs(split, exist_ok=True)
    for class_folder in os.listdir(data_dir):
        os.makedirs(os.path.join(split, class_folder), exist_ok=True)

# 遍历每个类别的文件夹，将图片分配到 train、val 和 test 文件夹
for class_folder in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_folder)
    images = os.listdir(class_path)
    random.shuffle(images)

    # 分割为 8:1:1
    train_split = int(0.8 * len(images))
    val_split = int(0.9 * len(images))

    train_images = images[:train_split]
    val_images = images[train_split:val_split]
    test_images = images[val_split:]

    # 移动图像到对应文件夹
    for image in train_images:
        shutil.copy(os.path.join(class_path, image), os.path.join(train_dir, class_folder, image))
    for image in val_images:
        shutil.copy(os.path.join(class_path, image), os.path.join(val_dir, class_folder, image))
    for image in test_images:
        shutil.copy(os.path.join(class_path, image), os.path.join(test_dir, class_folder, image))

# 定义图像转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载数据集
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

# 创建 DataLoader
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Test images: {len(test_dataset)}")
