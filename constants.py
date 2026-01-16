import pygame
import os

PADDING_BORDER = 10
TILE_SIZE = 50
TILE_GAP = 3
GRID_SPACING = TILE_SIZE + TILE_GAP

GRID_WIDTH = 22
GRID_HEIGHT = 15

# SHELF_IMAGE_HEIGHT = 64
SHELF_IMAGE_HEIGHT = 50
CHARGE_STATION_HEIGHT = 58

ROBOT_WIDTH = 40
ROBOT_HEIGHT = 48

NUMBER_OF_ROBOTS = 5

# load images
def load_img(image_name, scale=None):
  image = pygame.image.load(os.path.join("assets", image_name))
  if scale is not None:
    image = pygame.transform.scale(image, scale)
  return image


# Images
ROBOT_IMAGE_SIDE = load_img("robot-side.png", ( ROBOT_HEIGHT, ROBOT_WIDTH))
ROBOT_IMAGE_SIDE_BOX = load_img("robot-side-box.png", ( ROBOT_HEIGHT, ROBOT_WIDTH))
ROBOT_IMAGE_VERTICAL = load_img("robot-vertical.png", ( ROBOT_WIDTH, ROBOT_HEIGHT))
ROBOT_IMAGE_VERTICAL_BOX = load_img("robot-vertical-box.png", (ROBOT_WIDTH, ROBOT_HEIGHT))
SHELF_IMAGE_EMPTY = load_img("shelf-empty.png", (TILE_SIZE, SHELF_IMAGE_HEIGHT))
SHELF_IMAGE_FILLED = load_img("shelf-filled.png", (TILE_SIZE, SHELF_IMAGE_HEIGHT))
SHELF_SHADOW = load_img("shelf-shadow.png", (TILE_SIZE + 3, SHELF_IMAGE_HEIGHT))

ROBOT_CHARGE_STATION_IMAGE = load_img("robot-charging.png", (TILE_SIZE, CHARGE_STATION_HEIGHT))
DROP_OFF_PLATFORM_IMAGE = load_img("drop-off.png", (TILE_SIZE, CHARGE_STATION_HEIGHT))
