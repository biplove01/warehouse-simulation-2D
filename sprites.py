import pygame
from constants import *


# class Box(pygame.Rect):
#   def __init__(self, x, y, image):
#     pygame.Rect.__init__(self, x, y, BOX_WIDTH, BOX_HEIGHT)
#     self.image = image

# SHELF
class Shelf(pygame.Rect):
  def __init__(self, x, y, empty_shelf_image, loaded_shelf_image):
    pygame.Rect.__init__(self, x, y, TILE_SIZE, TILE_SIZE)
    self.empty_image = empty_shelf_image
    self.loaded_image = loaded_shelf_image
    self.image = self.empty_image
    self.has_box = False
    self.hitbox = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)
    self.shadow_image = SHELF_SHADOW


# CHARGE STATION
class ChargeStation(pygame.Rect):
  def __init__(self, x, y, image):
    pygame.Rect.__init__(self, x, y, TILE_SIZE, CHARGE_STATION_HEIGHT)
    self.image = image



# DROPOFF PLATFORM (BOXES)
class DropoffPlatform(pygame.Rect):
  def __init__(self, x, y, image):
    pygame.Rect.__init__(self, x, y, TILE_SIZE, CHARGE_STATION_HEIGHT)
    self.image = image
    self.hitbox = pygame.Rect(x, y, 4*(TILE_SIZE + TILE_GAP), CHARGE_STATION_HEIGHT)
