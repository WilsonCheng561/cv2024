import pygame
import random
import math
import os
from os import path

WIDTH = 480
HEIGHT = 600
FPS = 50 # Control refresh rate
enemy_num=8 # How many enemies in the screen, we can use this to adjust difficulty
bullet_num=10

# define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# create window
pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Fighter game!")
clock = pygame.time.Clock()

img_dir = path.join(path.dirname(__file__), 'image')
enermy_images = []
enermy_list =['a-01.png','a-01b.png',
              'b-03.png','b-03b.png',
              'c-03.png','c-03b.png',
              'c-04.png','c-04b.png',
                                    ]
for img in enermy_list:
    enermy_images.append(pygame.image.load(path.join(img_dir, img)).convert_alpha())

background = pygame.image.load(path.join(img_dir, "map01-01.jpg")).convert_alpha()
background_rect = background.get_rect()

player_img = pygame.image.load(path.join(img_dir, "p-01a.png")).convert_alpha()
bullet_img = pygame.image.load(path.join(img_dir, "bullet-04-y.png")).convert_alpha()


class Player(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image=player_img
        self.original_width, self.original_height = self.image.get_size()
        self.image = pygame.transform.scale(player_img, (self.original_width // 2, self.original_height // 2))
        self.rect = self.image.get_rect()
        self.rect.centerx = WIDTH / 2
        self.rect.bottom = HEIGHT - 10
        self.speedx = 0
        self.energy = 100

    def update(self):
        self.speedx = 0

        # TODO If handgesture==move, maybe we need pass the handgesture variable into the update function
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
            angle=math.pi/(bullet_num-1)*i
            bullet = Bullet(self.rect.centerx, self.rect.top, angle)
            all_sprites.add(bullet)
            bullets.add(bullet)
        

class Enemies(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = random.choice(enermy_images)
        self.original_width, self.original_height = self.image.get_size()
        self.image= pygame.transform.scale(self.image, (self.original_width // 2, self.original_height // 2))
        self.image = self.image.copy()
        self.rect = self.image.get_rect()
        self.rect.x = random.randrange(WIDTH - self.rect.width)
        self.rect.y = random.randrange(-100, -40)
        self.speedy = random.randrange(1, 8)
        self.speedx = random.randrange(-3, 3)

    def update(self):
        self.rect.x += self.speedx
        self.rect.y += self.speedy
        if self.rect.top > HEIGHT + 10 or self.rect.left < -25 or self.rect.right > WIDTH + 20:
            self.rect.x = random.randrange(WIDTH - self.rect.width)
            self.rect.y = random.randrange(-100, -40)
            self.speedy = random.randrange(1, 8)

class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y ,angle):
        pygame.sprite.Sprite.__init__(self)
        self.image = bullet_img
        self.original_width, self.original_height = self.image.get_size()
        self.image = pygame.transform.scale(bullet_img,(self.original_width // 2, self.original_height // 2))
        self.rect = self.image.get_rect()
        self.rect.bottom = y
        self.rect.centerx = x
        self.speedy = -10 * math.sin(angle)
        self.speedx = -10 * math.cos(angle)
        

    def update(self):
        self.rect.x += self.speedx
        self.rect.y += self.speedy
        # kill if it moves off the top of the screen
        if self.rect.bottom < 0:
            self.kill()


all_sprites = pygame.sprite.Group()
enemies = pygame.sprite.Group()
bullets = pygame.sprite.Group()
player = Player()


font_name = pygame.font.match_font('arial')


def draw_text(surf, text, size, x, y):
    font = pygame.font.Font(font_name, size)
    text_surface = font.render(text, True, WHITE)
    text_rect = text_surface.get_rect()
    text_rect.midtop = (x, y)
    surf.blit(text_surface, text_rect)

def draw_energy_bar(surf, x, y, pct):
    if pct < 0:
        pct = 0
    BAR_LENGTH = 100
    BAR_HEIGHT = 10
    fill = (pct / 100) * BAR_LENGTH
    outline_rect = pygame.Rect(x, y, BAR_LENGTH, BAR_HEIGHT)
    fill_rect = pygame.Rect(x, y, fill, BAR_HEIGHT)
    pygame.draw.rect(surf, GREEN, fill_rect)
    pygame.draw.rect(surf, WHITE, outline_rect, 2)

def newenermy():
    e = Enemies()
    all_sprites.add(e)
    enemies.add(e)

# TODO the pygmae allows muti-players, Maybe we can develop muti-players mode if time available
all_sprites.add(player)
for i in range(enemy_num):
    e = Enemies()
    all_sprites.add(e)
    enemies.add(e)
score = 0
# Game loop
running=True


while running:

    # Make sure that the while loop does not exceed FPS times per second
    clock.tick(FPS)
    # TODO A function doing hand gesture classification

    for event in pygame.event.get():
        # check for closing window
        if event.type == pygame.QUIT:
            running = False

        # TODO If handgesture==shoot
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                player.shoot()

    hits = pygame.sprite.groupcollide(enemies, bullets, True, True)
    for hit in hits:
        e = Enemies()
        all_sprites.add(e)
        enemies.add(e)
    for enermy in hits:
        match enermy.original_width:
            case x if x<55:
                score += 3
            case x if 55 < x < 90:
                score += 2
            case x if x>90:
                score += 1

    # check if an enemy hit the player
    hits = pygame.sprite.spritecollide(player, enemies, True)
    for hit in hits:
        player.energy -= 25
        newenermy()
        if player.energy <= 0:
            running = False
    # TODO we can display "GameOver", but not kill the loop
    # if hits:
    #     running = False


    # Update
    all_sprites.update()

    # display
    screen.fill(BLACK)
    screen.blit(background, background_rect)
    all_sprites.draw(screen)
    draw_text(screen, str(score), 18, WIDTH / 2, 10)
    draw_energy_bar(screen, 5, 5, player.energy)
    pygame.display.flip()

pygame.quit()
