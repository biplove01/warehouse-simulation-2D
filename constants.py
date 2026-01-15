import pygame
import os

PADDING_BORDER = 10
PICKUP_DISTANCE_X = 70
PICKUP_DISTANCE_Y = 90

TILE_SIZE = 50
TILE_GAP = 3
ROAD_MARGIN = 6

GAME_WIDTH = 1170 + PADDING_BORDER     #22 cols
GAME_HEIGHT = 712 + PADDING_BORDER    #15 rows

CHARGE_STATION_HEIGHT = 58
SHELF_IMAGE_HEIGHT = 64
SHELF_HEIGHT = 40
SHELF_TOP_HEIGHT = 40

BOX_HEIGHT = 30
BOX_WIDTH = 30

ROBOT_WIDTH = 40
ROBOT_LENGTH = 40
ROBOT_DISTANCE = 5

# robot position and size
ROBOT_X = PADDING_BORDER + 5
ROBOT_Y = PADDING_BORDER + 5
ROBOT_IMAGE_WIDTH = 40
ROBOT_IMAGE_LENGTH = 48
ROBOT_VELOCITY_X = 3
ROBOT_VELOCITY_Y = 3

# environment variables
FRICTION = 0.2

NUMBER_OF_ROBOTS = 5

def load_img(image_name, scale=None):
  image = pygame.image.load(os.path.join("assets", image_name))
  if scale is not None:
    image = pygame.transform.scale(image, scale)
  return image


# Images
ROBOT_IMAGE_SIDE = load_img("robot-side.png", ( ROBOT_IMAGE_LENGTH, ROBOT_IMAGE_WIDTH))
ROBOT_IMAGE_SIDE_BOX = load_img("robot-side-box.png", ( ROBOT_IMAGE_LENGTH, ROBOT_IMAGE_WIDTH))
ROBOT_IMAGE_VERTICAL = load_img("robot-vertical.png", ( ROBOT_IMAGE_WIDTH, ROBOT_IMAGE_LENGTH))
ROBOT_IMAGE_VERTICAL_BOX = load_img("robot-vertical-box.png", (ROBOT_IMAGE_WIDTH, ROBOT_IMAGE_LENGTH))
SHELF_IMAGE_EMPTY = load_img("shelf-empty.png", (TILE_SIZE, SHELF_IMAGE_HEIGHT))
SHELF_IMAGE_FILLED = load_img("shelf-filled.png", (TILE_SIZE, SHELF_IMAGE_HEIGHT))
SHELF_SHADOW = load_img("shelf-shadow.png", (TILE_SIZE + 3, SHELF_IMAGE_HEIGHT))

BOX_IMAGE = load_img("box.png", (BOX_WIDTH, BOX_HEIGHT))

ROBOT_CHARGE_STATION_IMAGE = load_img("robot-charging.png", (TILE_SIZE, CHARGE_STATION_HEIGHT))
DROP_OFF_PLATFORM_IMAGE = load_img("drop-off.png", (4 * (TILE_SIZE + TILE_GAP), CHARGE_STATION_HEIGHT))
