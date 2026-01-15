import pygame
from constants import *


class Box(pygame.Rect):
  def __init__(self, x, y, image):
    pygame.Rect.__init__(self, x, y, BOX_WIDTH, BOX_HEIGHT)
    self.image = image

# SHELF
class Shelf(pygame.Rect):
  def __init__(self, x, y, empty_shelf_image, loaded_shelf_image):
    pygame.Rect.__init__(self, x, y, TILE_SIZE, SHELF_HEIGHT)
    self.empty_image = empty_shelf_image
    self.loaded_image = loaded_shelf_image
    self.image = self.empty_image
    self.has_box = False
    self.hitbox = pygame.Rect(x, y+ 30, TILE_SIZE, SHELF_HEIGHT-6)
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
