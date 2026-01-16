from constants import *
from sprites import *
import random


def create_map():
  charge_stations= []
  shelves = []

  dropoff_platforms = []

  # Charge stations
  for i in range(5):
    x = PADDING_BORDER + i * (TILE_SIZE + TILE_GAP)
    y = PADDING_BORDER
    charge_station = ChargeStation( x, y , ROBOT_CHARGE_STATION_IMAGE)
    charge_stations.append(charge_station)

  # shelves horizontal
  for i in range(12):
    x1 = PADDING_BORDER + (5 * (TILE_SIZE + TILE_GAP + TILE_GAP/2) ) + i * (TILE_SIZE + TILE_GAP)
    y1 = PADDING_BORDER
    shelf1 = Shelf( x1, y1, SHELF_IMAGE_EMPTY, SHELF_IMAGE_FILLED )

    y2 = y1 + SHELF_TOP_HEIGHT + 1.7 * (TILE_SIZE + TILE_GAP)
    y3 = y2 + SHELF_TOP_HEIGHT + TILE_GAP

    y4 = y3 + SHELF_TOP_HEIGHT + 1.7 * (TILE_SIZE + TILE_GAP)
    y5 = y4 + SHELF_TOP_HEIGHT + TILE_GAP

    y6 = y5 + SHELF_TOP_HEIGHT + 1.7 * (TILE_SIZE + TILE_GAP)
    y7 = y6 + SHELF_TOP_HEIGHT + TILE_GAP

    shelf2 = Shelf( x1, y2, SHELF_IMAGE_EMPTY, SHELF_IMAGE_FILLED )
    shelf3 = Shelf( x1, y3, SHELF_IMAGE_EMPTY, SHELF_IMAGE_FILLED )
    shelf4 = Shelf( x1, y4, SHELF_IMAGE_EMPTY, SHELF_IMAGE_FILLED )
    shelf5 = Shelf( x1, y5, SHELF_IMAGE_EMPTY, SHELF_IMAGE_FILLED )
    shelf6 = Shelf( x1, y6, SHELF_IMAGE_EMPTY, SHELF_IMAGE_FILLED )
    shelf7 = Shelf( x1, y7, SHELF_IMAGE_EMPTY, SHELF_IMAGE_FILLED )

    shelves.append(shelf1)
    shelves.append(shelf2)
    shelves.append(shelf3)
    shelves.append(shelf4)
    shelves.append(shelf5)
    shelves.append(shelf6)
    shelves.append(shelf7)

  # shelves vertical
  for i in range(9):
    x1 = PADDING_BORDER + (TILE_SIZE)
    y1 = PADDING_BORDER + (CHARGE_STATION_HEIGHT + 1.5* TILE_SIZE)  + i * (SHELF_TOP_HEIGHT + TILE_GAP)

    x2 = x1 + (TILE_SIZE + TILE_GAP)

    shelf1 = Shelf(x1, y1, SHELF_IMAGE_EMPTY, SHELF_IMAGE_FILLED)
    shelf2 = Shelf(x2, y1, SHELF_IMAGE_EMPTY, SHELF_IMAGE_FILLED)


    shelves.append(shelf1)
    shelves.append(shelf2)

  for i in range(10):
    y1 = PADDING_BORDER + (CHARGE_STATION_HEIGHT + 1.5* TILE_SIZE)  + i * (SHELF_TOP_HEIGHT + TILE_GAP)

    x3 = PADDING_BORDER + 19* (TILE_SIZE + TILE_GAP)
    x4 = x3 + TILE_SIZE + TILE_GAP
    shelf3 = Shelf(x3, y1, SHELF_IMAGE_EMPTY, SHELF_IMAGE_FILLED)
    shelf4 = Shelf(x4, y1, SHELF_IMAGE_EMPTY, SHELF_IMAGE_FILLED)
    shelves.append(shelf3)
    shelves.append(shelf4)

  # Drop off platform
  x_dropoff = PADDING_BORDER
  y_dropoff = y7 + 2*(TILE_SIZE+ TILE_GAP)
  dropoff_platform = DropoffPlatform(x_dropoff, y_dropoff, DROP_OFF_PLATFORM_IMAGE)
  dropoff_platforms.append(dropoff_platform)

   # randomly make shelf have box
  box_shelf = random.choice(shelves)
  box_shelf.has_box = True
  box_shelf.image = box_shelf.loaded_image

  return shelves, charge_stations, dropoff_platforms
