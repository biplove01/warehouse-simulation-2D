from constants import *

def create_map():

  # Charge stations
  for i in range(5):
    x = PADDING_BORDER + i * (TILE_SIZE + TILE_GAP)
    y = PADDING_BORDER
    charge_station = ChargeStation( x, y ,robot_charge_station)
    charge_stations.append(charge_station)

  # shelves horizontal
  for i in range(12):
    x1 = PADDING_BORDER + (5 * (TILE_SIZE + TILE_GAP + TILE_GAP/2) ) + i * (TILE_SIZE + TILE_GAP)
    y1 = PADDING_BORDER
    shelf1 = Shelf( x1, y1, shelf_image_empty )

    y2 = y1 + SHELF_TOP_HEIGHT + 1.5 * (TILE_SIZE + TILE_GAP)
    y3 = y2 + SHELF_TOP_HEIGHT + TILE_GAP

    y4 = y3 + SHELF_TOP_HEIGHT + 1.5 * (TILE_SIZE + TILE_GAP)
    y5 = y4 + SHELF_TOP_HEIGHT + TILE_GAP

    y6 = y5 + SHELF_TOP_HEIGHT + 1.5 * (TILE_SIZE + TILE_GAP)
    y7 = y6 + SHELF_TOP_HEIGHT + TILE_GAP

    shelf2 = Shelf( x1, y2, shelf_image_empty )
    shelf3 = Shelf( x1, y3, shelf_image_empty )
    shelf4 = Shelf( x1, y4, shelf_image_empty )
    shelf5 = Shelf( x1, y5, shelf_image_empty )
    shelf6 = Shelf( x1, y6, shelf_image_empty )
    shelf7 = Shelf( x1, y7, shelf_image_empty )

    shelves.append(shelf1)
    shelves.append(shelf2)
    shelves.append(shelf3)
    shelves.append(shelf4)
    shelves.append(shelf5)
    shelves.append(shelf6)
    shelves.append(shelf7)

  # shelves vertical
  for i in range(9):
    x1 = PADDING_BORDER
    y1 = PADDING_BORDER + (CHARGE_STATION_HEIGHT + TILE_SIZE + 3* TILE_GAP)  + i * (SHELF_TOP_HEIGHT + TILE_GAP)

    x2 = x1 + 2* (TILE_SIZE + TILE_GAP)
    x3 = x2 + 17* (TILE_SIZE + TILE_GAP)
    x4 = x3 + TILE_SIZE + TILE_GAP

    shelf1 = Shelf(x1, y1, shelf_image_empty)
    shelf2 = Shelf(x2, y1, shelf_image_empty)
    shelf3 = Shelf(x3, y1, shelf_image_empty)
    shelf4 = Shelf(x4, y1, shelf_image_empty)


    shelves.append(shelf1)
    shelves.append(shelf2)
    shelves.append(shelf3)
    shelves.append(shelf4)

  # Drop off platform
  x_dropoff = PADDING_BORDER
  y_dropoff = y7 + 2*(TILE_SIZE+ TILE_GAP)
  dropoff_platform = DropoffPlatform(x_dropoff, y_dropoff, drop_off_platform_image)
  dropoff_platforms.append(dropoff_platform)
