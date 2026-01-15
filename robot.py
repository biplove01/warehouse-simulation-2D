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

    for shelf in shelves:
      if not shelf.has_box:
        continue

      # Use VISUAL dimensions for interaction
      shelf_visual_bottom = shelf.y + SHELF_IMAGE_HEIGHT
      shelf_visual_top = shelf.y
      shelf_right = shelf.x + TILE_SIZE
      shelf_left = shelf.x

      dx = abs(self.centerx - (shelf.x + TILE_SIZE // 2))
      dy = abs(self.centery - (shelf.y + SHELF_IMAGE_HEIGHT // 2))

      in_front = False
      margin = 15  # tolerance in pixels

      if self.direction == 'up':
          # Robot should be just below the visible shelf
          if (self.top >= shelf_visual_bottom - margin and
              self.top <= shelf_visual_bottom + margin and
              dx < TILE_SIZE // 2):
              in_front = True
      elif self.direction == 'down':
          if (self.bottom >= shelf_visual_top + margin and
              dx < TILE_SIZE // 2):
              in_front = True
      elif self.direction == 'left':
          if (self.right >= shelf_right - margin and
              self.right <= shelf_right + margin and
              dy < SHELF_IMAGE_HEIGHT // 2):
              in_front = True
      elif self.direction == 'right':
          if (self.left <= shelf_left + margin and
              self.left >= shelf_left - margin and
              dy < SHELF_IMAGE_HEIGHT // 2):
              in_front = True

      if in_front:
        print("PICKING UP BOX!")
        self.loaded = True
        shelf.has_box = False
        shelf.image = shelf.empty_image
        self.update_image()
        print(f"Robot now loaded: {self.loaded}")
        break



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
