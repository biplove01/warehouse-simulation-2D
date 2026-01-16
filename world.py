from constants import *
from sprites import *
import random


def create_map():
  charge_stations = []
  shelves = []
  dropoff_platforms = []

  def grid_to_pixel(gx, gy):
      return PADDING_BORDER + gx * GRID_SPACING, PADDING_BORDER + gy * GRID_SPACING

  # Charge stations: row 0, cols 0–4
  for i in range(5):
      x, y = grid_to_pixel(i, 0)
      charge_stations.append(ChargeStation(x, y, ROBOT_CHARGE_STATION_IMAGE))

  # Horizontal shelves (vertical stacks)
  for col in range(5, 17):  # 12 columns starting at col 5
      for row_offset in [0, 3, 4, 7, 8, 11, 12]:  # approximate your spacing
          x, y = grid_to_pixel(col, row_offset)
          shelf = Shelf(x, y, SHELF_IMAGE_EMPTY, SHELF_IMAGE_FILLED)
          shelves.append(shelf)

  # Left vertical shelves
  for row in range(2, 11):
      for col in [1, 2]:
          x, y = grid_to_pixel(col, row)
          shelves.append(Shelf(x, y, SHELF_IMAGE_EMPTY, SHELF_IMAGE_FILLED))

  # Right vertical shelves
  for row in range(2, 12):
      for col in [19, 20]:
          x, y = grid_to_pixel(col, row)
          shelves.append(Shelf(x, y, SHELF_IMAGE_EMPTY, SHELF_IMAGE_FILLED))

  # Drop-off platform
  for i in range(4):
    x, y = grid_to_pixel(i, 14)
    dropoff_platform = DropoffPlatform(x, y, DROP_OFF_PLATFORM_IMAGE)
    dropoff_platforms.append(dropoff_platform)

  # Random box
  box_shelf = random.choice(shelves)
  box_shelf.has_box = True
  box_shelf.image = box_shelf.loaded_image

  return shelves, charge_stations, dropoff_platforms
