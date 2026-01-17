import pygame
from constants import *

# ROBOT (PLAYER)
class Robot():
  def __init__(self, start_x=0, start_y=0):
    self.grid_x = start_x
    self.grid_y = start_y
    self.direction = 'up'
    self.loaded = False
    self.update_image()


  def _grid_to_pixel_centered(self, gx, gy):
    base_x = PADDING_BORDER + gx * GRID_SPACING
    base_y = PADDING_BORDER + gy * GRID_SPACING
    offset_x = (TILE_SIZE - ROBOT_WIDTH) // 2
    offset_y = (TILE_SIZE - ROBOT_HEIGHT) // 2
    return base_x + offset_x, base_y + offset_y


  def update_image(self):
    if self.direction == 'right' or self.direction == 'left':
      self.image = ROBOT_IMAGE_SIDE_BOX if self.loaded else ROBOT_IMAGE_SIDE
    elif self.direction == 'up' or self.direction =='down':
      self.image = ROBOT_IMAGE_VERTICAL_BOX if self.loaded else ROBOT_IMAGE_VERTICAL

  def get_pixel_rect(self):
    x = PADDING_BORDER + self.grid_x * GRID_SPACING
    y = PADDING_BORDER + self.grid_y * GRID_SPACING
    return pygame.Rect(x, y, ROBOT_WIDTH, ROBOT_HEIGHT)


  def can_move_to(self, gx, gy, spaces):
    if gx < 0 or gx >= GRID_WIDTH or gy < 0 or gy >= GRID_HEIGHT:
      return False

    for space in spaces:
      space_grid_x = round((space.x - PADDING_BORDER) / GRID_SPACING)
      space_grid_y = round((space.y - PADDING_BORDER) / GRID_SPACING)
      if space_grid_x == gx and space_grid_y == gy:
        return False
    return True


  def handle_inputs_single(self, direction, obstracle):
    dx, dy = 0, 0
    if direction == 'up':
        dy = -1
    elif direction == 'down':
        dy = 1
    elif direction == 'left':
        dx = -1
    elif direction == 'right':
        dx = 1
    else:
        return

    self.direction = direction
    next_x = self.grid_x + dx
    next_y = self.grid_y + dy

    if self.can_move_to(next_x, next_y, obstracle):
      self.grid_x = next_x
      self.grid_y = next_y
    self.update_image()


  def pickup_box(self, shelves):

    if self.loaded:
      return

    robot_rect = self.get_pixel_rect()
    margin = 30

    for shelf in shelves:
      if not shelf.has_box:
        continue

      shelf_center_x = shelf.x + TILE_SIZE // 2
      shelf_center_y = shelf.y + TILE_SIZE // 2

      dx = abs(robot_rect.centerx - shelf_center_x)
      dy = abs(robot_rect.centery - shelf_center_y)

      in_front = False
      if self.direction == 'up':
        if (shelf.y + SHELF_IMAGE_HEIGHT - margin <= robot_rect.top <= shelf.y + SHELF_IMAGE_HEIGHT + margin and dx < TILE_SIZE // 2):
          in_front = True
      elif self.direction == 'down':
        if (shelf.y - margin <= robot_rect.bottom <= shelf.y + margin and dx < TILE_SIZE // 2):
          in_front = True
      elif self.direction == 'left':
        if (shelf.x + TILE_SIZE - margin <= robot_rect.left <= shelf.x + TILE_SIZE + margin and shelf.y < robot_rect.bottom and robot_rect.top < shelf.y + SHELF_IMAGE_HEIGHT):
          in_front = True
      elif self.direction == 'right':
        if (shelf.x - margin <= robot_rect.right <= shelf.x + margin and shelf.y < robot_rect.bottom and robot_rect.top < shelf.y + SHELF_IMAGE_HEIGHT):
          in_front = True

      if in_front:
        self.loaded = True
        shelf.has_box = False
        shelf.image = shelf.empty_image
        self.update_image()
        break


  def drop_box(self, dropoff_platforms):
    if not self.loaded:
        return False

    robot_rect = self.get_pixel_rect()
    margin = 15

    for platform in dropoff_platforms:
      # Platform bounds
      plat_left = platform.x
      plat_right = platform.x + platform.image.get_width()
      plat_top = platform.y
      plat_bottom = platform.y + platform.image.get_height()

      in_position = False

      if self.direction == 'down':
        if (plat_top - margin <= robot_rect.bottom <= plat_top + margin and plat_left <= robot_rect.centerx <= plat_right):
            in_position = True

      elif self.direction == 'up':
        if (plat_bottom - margin <= robot_rect.top <= plat_bottom + margin and plat_left <= robot_rect.centerx <= plat_right):
          in_position = True

      elif self.direction == 'right':
        if (plat_left - margin <= robot_rect.right <= plat_left + margin and plat_top <= robot_rect.centery <= plat_bottom):
          in_position = True

      elif self.direction == 'left':
        if (plat_right - margin <= robot_rect.left <= plat_right + margin and plat_top <= robot_rect.centery <= plat_bottom):
          in_position = True

      if in_position:
        self.loaded = False
        self.update_image()
        return True

    return False
