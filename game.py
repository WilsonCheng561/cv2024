import pygame
import random
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import mediapipe as mp
from PIL import Image


# 初始化Pygame
pygame.init()
pygame.mixer.init()

# 定义窗口尺寸
WIDTH = 480
HEIGHT = 600
FPS = 50  # 控制刷新率
enemy_num = 8  # 屏幕上的敌人数量，可用于调整难度
bullet_num = 10

# 定义颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# 定义手势类别映射
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

# 加载手势识别模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = efficientnet_v2_s(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(gesture_map))
model.load_state_dict(torch.load('checkpoint/best_gesture_model.pth', map_location=device))
model = model.to(device)
model.eval()

# 初始化摄像头
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# 初始化Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# 创建Pygame窗口
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Fighter Game!")
clock = pygame.time.Clock()

# 定义玩家类
class Player(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((50, 40))
        self.image.fill(GREEN)
        self.rect = self.image.get_rect()
        self.rect.centerx = WIDTH / 2
        self.rect.bottom = HEIGHT - 10
        self.speedx = 0

    def update(self, move_command=None):
        self.speedx = 0
        if move_command is not None:
            self.rect.centerx = move_command
        else:
            keystate = pygame.key.get_pressed()
            if keystate[pygame.K_LEFT]:
                self.speedx = -8
            if keystate[pygame.K_RIGHT]:
                self.speedx = 8
            self.rect.x += self.speedx

        if self.rect.right > WIDTH:
            self.rect.right = WIDTH
        if self.rect.left < 0:
            self.rect.left = 0

    def shoot(self):
        for i in range(bullet_num):
            angle = math.pi / (bullet_num - 1) * i
            bullet = Bullet(self.rect.centerx, self.rect.top, angle)
            all_sprites.add(bullet)
            bullets.add(bullet)

# 定义敌人类
class Enemies(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((30, 40))
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.rect.x = random.randrange(WIDTH - self.rect.width)
        self.rect.y = random.randrange(-100, -40)
        self.speedy = random.randrange(1, 8)
        self.speedx = random.randrange(-3, 3)

    def update(self, move_command=None):
        self.rect.x += self.speedx
        self.rect.y += self.speedy
        if self.rect.top > HEIGHT + 10 or self.rect.left < -25 or self.rect.right > WIDTH + 20:
            self.rect.x = random.randrange(WIDTH - self.rect.width)
            self.rect.y = random.randrange(-100, -40)
            self.speedy = random.randrange(1, 8)

# 定义子弹类
class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y, angle):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((10, 20))
        self.image.fill(YELLOW)
        self.rect = self.image.get_rect()
        self.rect.bottom = y
        self.rect.centerx = x
        self.speedy = -10 * math.sin(angle)
        self.speedx = -10 * math.cos(angle)

    def update(self, move_command=None):
        self.rect.x += self.speedx
        self.rect.y += self.speedy
        if self.rect.top < 0 or self.rect.left < 0 or self.rect.right > WIDTH or self.rect.bottom > HEIGHT:
            self.kill()


# 初始化精灵组
all_sprites = pygame.sprite.Group()
enemies = pygame.sprite.Group()
bullets = pygame.sprite.Group()
player = Player()

all_sprites.add(player)
for i in range(enemy_num):
    e = Enemies()
    all_sprites.add(e)
    enemies.add(e)

# 游戏循环
running = True
move_command = None
gesture = 'none'
confidence = 0.0

while running:
    # 保持循环运行速率
    clock.tick(FPS)

    # 处理Pygame事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 摄像头读取帧
    success, img = cap.read()
    if not success:
        continue
    img = cv2.flip(img, 1)
    img_display = img.copy()

    # 定义手部区域
    h, w, _ = img.shape
    hand_region = img[int(h/2 - 100):int(h/2 + 100), int(w/2 - 100):int(w/2 + 100)]

    # 如果手部区域大小不足，跳过本帧
    if hand_region.shape[0] == 0 or hand_region.shape[1] == 0:
        continue

    # 手势识别
    img_pil = Image.fromarray(cv2.cvtColor(hand_region, cv2.COLOR_BGR2RGB))
    img_transformed = data_transforms(img_pil)
    input_data = img_transformed.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_data)
        probabilities = nn.functional.softmax(outputs, dim=1)
        confidence_value, predicted = torch.max(probabilities, 1)
        gesture = gesture_map[predicted.item()]
        confidence = confidence_value.item() * 100

    # 显示手势标签和置信度
    cv2.putText(img_display, f'Gesture: {gesture} ({confidence:.2f}%)', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.rectangle(img_display, (int(w/2 - 100), int(h/2 - 100)),
                  (int(w/2 + 100), int(h/2 + 100)), (255, 0, 0), 2)

    # 根据手势执行相应的操作
    if gesture == 'restart' and confidence > 90:
        # 重置游戏状态
        for sprite in all_sprites:
            sprite.kill()
        player = Player()
        all_sprites.add(player)
        enemies = pygame.sprite.Group()
        bullets = pygame.sprite.Group()
        for i in range(enemy_num):
            e = Enemies()
            all_sprites.add(e)
            enemies.add(e)
    elif gesture == 'fist' and confidence > 90:
        # 清除所有敌人
        for enemy in enemies:
            enemy.kill()
    elif gesture == 'single finger' and confidence > 80:
        # 使用Mediapipe获取手指位置，控制战机移动
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(img_rgb)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            index_finger_tip = hand_landmarks.landmark[8]
            x = int(index_finger_tip.x * WIDTH)
            move_command = x
            # 绘制手部关键点
            mp_drawing.draw_landmarks(img_display, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            move_command = None
    else:
        move_command = None

    # 更新游戏状态
    all_sprites.update(move_command)

    # 检测子弹与敌人的碰撞
    hits = pygame.sprite.groupcollide(enemies, bullets, True, True)
    for hit in hits:
        e = Enemies()
        all_sprites.add(e)
        enemies.add(e)

    # 检测敌人与玩家的碰撞
    hits = pygame.sprite.spritecollide(player, enemies, False)
    if hits:
        running = False  # 可以改为显示"Game Over"界面

    # 绘制游戏画面
    screen.fill(BLACK)
    all_sprites.draw(screen)
    pygame.display.flip()

    # 显示摄像头窗口
    cv2.imshow('Hand Gesture Recognition', img_display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 退出
cap.release()
cv2.destroyAllWindows()
pygame.quit()
