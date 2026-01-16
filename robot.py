import pygame
from constants import *

# ROBOT (PLAYER)
class Robot(pygame.Rect):
  def __init__(self):
    pygame.Rect.__init__(self, ROBOT_X, ROBOT_Y, ROBOT_WIDTH, ROBOT_LENGTH)
    self.image = ROBOT_IMAGE_SIDE
    self.velocity_x = 0
    self.velocity_y = 0
    self.direction = 'up'
    self.loaded = False

  def update_image(self):
    if self.direction == 'right' or self.direction == 'left':
      self.image = ROBOT_IMAGE_SIDE_BOX if self.loaded else ROBOT_IMAGE_SIDE
    elif self.direction == 'up' or self.direction =='down':
      self.image = ROBOT_IMAGE_VERTICAL_BOX if self.loaded else ROBOT_IMAGE_VERTICAL

  def handle_inputs(self):
    keys = pygame.key.get_pressed()
    moving_up = keys[pygame.K_UP] or keys[pygame.K_w]
    moving_down = keys[pygame.K_DOWN] or keys[pygame.K_s]
    moving_left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    moving_right = keys[pygame.K_RIGHT] or keys[pygame.K_d]

    if moving_up:
      self.velocity_y = -ROBOT_VELOCITY_Y
      self.direction = 'up'
    elif moving_down:
      self.velocity_y = ROBOT_VELOCITY_Y
      self.direction = 'down'
    elif moving_right:
      self.velocity_x = ROBOT_VELOCITY_X
      self.direction = 'right'
    elif moving_left:
      self.velocity_x = -ROBOT_VELOCITY_X
      self.direction = 'left'
    self.update_image()


  def pickup_box(self, shelves):

    if self.loaded:
      return

    margin = 30

    for shelf in shelves:
      if not shelf.has_box:
        continue

      # visual dimensions for interaction
      shelf_visual_bottom = shelf.y + SHELF_IMAGE_HEIGHT
      shelf_visual_top = shelf.y
      shelf_right = shelf.x + TILE_SIZE
      shelf_left = shelf.x

      dx = abs(self.centerx - (shelf.x + TILE_SIZE // 2))           # horizontal center of robot - hor center of shelf
      dy = abs(self.centery - (shelf.y + SHELF_IMAGE_HEIGHT // 2))  # vertical center of robot - ver center of shelf

      in_front = False

      if self.direction == 'up':
        if (self.top >= shelf_visual_bottom - margin and self.top <= shelf_visual_bottom + margin and dx < TILE_SIZE // 2):
          in_front = True
      elif self.direction == 'down':
        if (self.bottom >= shelf_visual_top + margin and dx < TILE_SIZE // 2):
          in_front = True
      elif self.direction == 'left':
        if (shelf_right - margin <= self.left <= shelf_right + margin and self.bottom > shelf.y and self.top < shelf.y + SHELF_IMAGE_HEIGHT):
            in_front = True

      elif self.direction == 'right':
        if (shelf_left - margin <= self.right <= shelf_left + margin and self.bottom > shelf.y and self.top < shelf.y + SHELF_IMAGE_HEIGHT):
            in_front = True

      if in_front:
        self.loaded = True
        shelf.has_box = False
        shelf.image = shelf.empty_image
        self.update_image()
        print("Box picked from shelf!")
        break

  def drop_box(self, dropoff_platforms):
    if not self.loaded:
      return False

    margin = 15

    for platform in dropoff_platforms:
      dx = abs(self.centerx - (platform.centerx + TILE_SIZE * 2))                 # horiz center of robot - horiz center of platform
      dy = abs(self.centery - (platform.centery + CHARGE_STATION_HEIGHT // 2))    # verti center of robot - verti center of platform

      hitbox = platform.hitbox
      in_position = False

      if self.direction == 'down':
          if (platform.top - margin <= self.bottom <= platform.top + margin and dx < (3* TILE_SIZE) // 2):
              in_position = True

      elif self.direction == 'up':
          if (platform.bottom - margin <= self.top <= platform.bottom + margin and dx < (3* TILE_SIZE) // 2):
              in_position = True

      elif self.direction == 'right':
        # Robot left of platform, facing right
        if (hitbox.left - margin <= self.right <= hitbox.left + margin and
            self.bottom > hitbox.top and self.top < hitbox.bottom):
            in_position = True

      elif self.direction == 'left':
        # Robot right of platform, facing left
        if (hitbox.right - margin <= self.left <= hitbox.right + margin and
            self.bottom > hitbox.top and self.top < hitbox.bottom):
            in_position = True


      if in_position:
        self.loaded = False
        self.update_image()
        return True

      return False


  def handle_physics(self, obstracles):

    # move horizontally
    self.x += self.velocity_x
    if self.left < 0:
      self.left = 0
      self.velocity_x = 0
    elif self.right > GAME_WIDTH:
      self.right = GAME_WIDTH
      self.velocity_x = 0

    for obj in obstracles:
      hitbox = getattr(obj, 'hitbox', obj)
      if self.colliderect(hitbox) and self.bottom > hitbox.top and self.top < hitbox.bottom:
        if self.velocity_x > 0:
            self.right = hitbox.left
        elif self.velocity_x < 0:
            self.left = hitbox.right
        self.velocity_x = 0

    # move vertically
    self.y += self.velocity_y
    if self.top < 0:
      self.top = 0
      self.velocity_y = 0
    elif self.bottom > GAME_HEIGHT:
      self.bottom = GAME_HEIGHT
      self.velocity_y = 0

    for obj in obstracles:
      hitbox = getattr(obj, 'hitbox', obj)
      if self.colliderect(hitbox):
        if self.velocity_y > 0:
            self.bottom = hitbox.top
        elif self.velocity_y < 0:
            self.top = hitbox.bottom
        self.velocity_y = 0

    self.velocity_x *= FRICTION
    self.velocity_y *= FRICTION

    if abs(self.velocity_x) < 0.1: self.velocity_x = 0
    if abs(self.velocity_y) < 0.1: self.velocity_y = 0

# hehe
