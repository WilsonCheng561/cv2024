import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import math
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from PIL import Image

# 加载训练好的手势识别模型
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

# 手势类别映射
gesture_map = {
    0: 'fist',
    1: 'single finger',
    2: '2',
    3: '3',
    4: '4',
    5: 'restart',
    6: '6',
    7: '7',
    8: '8',
    9: 'other'
}

# 图像大小
IMG_SIZE = 128

# 数据预处理
data_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)  # 10 个手势类别
model.load_state_dict(torch.load('checkpoint/best_gesture_model.pth', map_location=device))
model = model.to(device)
model.eval()

# 初始化摄像头
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

def preprocess_image(img):
    # 转换图像为 PIL 格式，再进行预处理
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_transformed = data_transforms(img_pil)
    return img_transformed

def recognize_gesture(img):
    # 识别手势并获取置信度
    processed_img = preprocess_image(img)
    input_data = processed_img.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_data)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        gesture_label = gesture_map[predicted.item()]
    return gesture_label, confidence.item() * 100

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (640, 360))  # 将窗口缩小为原来的一半

    # 定义手部区域（假设手部位于图像的中心区域）
    h, w, _ = img.shape
    hand_region = img[int(h/2 - 100):int(h/2 + 100), int(w/2 - 100):int(w/2 + 100)]

    # 如果手部区域大小不足，跳过本帧
    if hand_region.shape[0] == 0 or hand_region.shape[1] == 0:
        continue

    # 调用手势识别函数
    gesture, confidence = recognize_gesture(hand_region)

    # 判断是否显示手势类别或“Just kidding!”
    display_text = "Just kidding!"
    if gesture == "restart" and confidence > 95:
        display_text = f"Gesture: {gesture} ({confidence:.2f}%)"
    elif confidence > 85:
        display_text = f"Gesture: {gesture} ({confidence:.2f}%)"

    # 显示手势标签或“Just kidding!”
    cv2.putText(img, display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 显示手部区域
    cv2.rectangle(img, (int(w/2 - 100), int(h/2 - 100)), (int(w/2 + 100), int(h/2 + 100)), (255, 0, 0), 2)

    cv2.imshow('Hand Gesture Recognition', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()