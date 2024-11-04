import pygame
import random
import math

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

class Player(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((50, 40))
        self.image.fill(GREEN)
        self.rect = self.image.get_rect()
        self.rect.centerx = WIDTH / 2
        self.rect.bottom = HEIGHT - 10
        self.speedx = 0

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
        self.image = pygame.Surface((30, 40))
        self.image.fill(RED)
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
        self.image = pygame.Surface((10, 20))
        self.image.fill(YELLOW)
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

# TODO the pygmae allows muti-players, Maybe we can develop muti-players mode if time available
all_sprites.add(player)
for i in range(enemy_num):
    e=Enemies()
    all_sprites.add(e)
    enemies.add(e)

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

    # check if a bullet hit the enemy
    hits = pygame.sprite.groupcollide(enemies, bullets, True, True)
    for hit in hits:
        e = Enemies()
        all_sprites.add(e)
        enemies.add(e)

    # check if an enemy hit the player
    hits = pygame.sprite.spritecollide(player, enemies, False)

    # TODO we can display "GameOver", but not kill the loop
    # if hits:
    #     running = False


    # Update
    all_sprites.update()

    # display
    screen.fill(BLACK)
    all_sprites.draw(screen)
    pygame.display.flip()

pygame.quit()
