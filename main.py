import pygame
from sys import exit
import os

# from constants import *


PADDING_BORDER = 10

TILE_SIZE = 50
TILE_GAP = 3

CHARGE_STATION_HEIGHT = 58
SHELF_HEIGHT = 74
SHELF_TOP_HEIGHT = 36
GAME_WIDTH = 704 + PADDING_BORDER     #22 cols
GAME_HEIGHT = 512 + PADDING_BORDER    #15 rows
ROBOT_DISTANCE = 5

# warehouse layout
NUMBER_OF_ROBOTS = 5


# robot position and size
ROBOT_X = PADDING_BORDER + 5
ROBOT_Y = PADDING_BORDER + 5
ROBOT_WIDTH = 40
ROBOT_LENGTH = 48
ROBOT_VELOCITY_X = 3
ROBOT_VELOCITY_Y = 3

# environment variables
FRICTION = 0.4



def load_img(image_name, scale=None):
  image = pygame.image.load(os.path.join("assets", image_name))
  if scale is not None:
    image = pygame.transform.scale(image, scale)
  return image


# Images
robot_image_side = load_img("robot-side.png", ( ROBOT_LENGTH, ROBOT_WIDTH))
robot_image_side_box = load_img("robot-side-box.png", ( ROBOT_LENGTH, ROBOT_WIDTH))
robot_image_vertical = load_img("robot-vertical.png", ( ROBOT_WIDTH, ROBOT_LENGTH))
robot_image_vertical_box = load_img("robot-vertical-box.png", (ROBOT_WIDTH, ROBOT_LENGTH))
shelf_image_empty = load_img("shelf-empty.png", (TILE_SIZE, SHELF_HEIGHT))
shelf_image_filled = load_img("shelf-filled.png", (TILE_SIZE, SHELF_HEIGHT))

robot_charge_station = load_img("robot-charging.png", (TILE_SIZE, CHARGE_STATION_HEIGHT))
drop_off_platform_image = load_img("drop-off.png", (4 * (TILE_SIZE + TILE_GAP), CHARGE_STATION_HEIGHT))


pygame.init()
window = pygame.display.set_mode((GAME_WIDTH, GAME_HEIGHT))
pygame.display.set_caption("Warehouse Simulation")
pygame.display.set_icon(robot_image_side)
clock = pygame.time.Clock()


class Robot(pygame.Rect):
  def __init__(self):
    pygame.Rect.__init__(self, ROBOT_X, ROBOT_Y, ROBOT_WIDTH, ROBOT_LENGTH)
    self.image = robot_image_side
    self.velocity_x = 0
    self.velocity_y = 0
    self.direction = 'up'
    self.loaded = False

  def update_image(self):
    if self.direction == 'right' or self.direction == 'left':
      if self.loaded:
        self.image = robot_image_side_box
      self.image = robot_image_side
    elif self.direction == 'up' or self.direction =='down':
      if self.loaded:
        self.image = robot_image_vertical_box
      self.image = robot_image_vertical


class Shelf(pygame.Rect):
  def __init__(self, x, y, image):
    pygame.Rect.__init__(self, x, y, TILE_SIZE, SHELF_HEIGHT)
    self.image = image
    self.filled = False


class ChargeStation(pygame.Rect):
  def __init__(self, x, y, image):
    pygame.Rect.__init__(self, x, y, TILE_SIZE, CHARGE_STATION_HEIGHT)
    self.image = image


class DropoffPlatform(pygame.Rect):
  def __init__(self, x, y, image):
    pygame.Rect.__init__(self, x, y, TILE_SIZE, CHARGE_STATION_HEIGHT)
    self.image = image


def move():
  # horizontal friction
  if robot.velocity_x < 0:
    robot.velocity_x = min(0, robot.velocity_x + FRICTION)
  elif robot.velocity_x > 0:
    robot.velocity_x = max(0, robot.velocity_x - FRICTION)

  # vertical friction
  if robot.velocity_y < 0:
    robot.velocity_y = min(0, robot.velocity_y + FRICTION)
  elif robot.velocity_y > 0:
    robot.velocity_y = max(0, robot.velocity_y - FRICTION)

  robot.x += robot.velocity_x
  robot.y += robot.velocity_y


def handle_movements():
  keys = pygame.key.get_pressed()

  moving_up = keys[pygame.K_UP] or keys[pygame.K_w]
  moving_down = keys[pygame.K_DOWN] or keys[pygame.K_s]
  moving_left = keys[pygame.K_LEFT] or keys[pygame.K_a]
  moving_right = keys[pygame.K_RIGHT] or keys[pygame.K_d]

  if moving_up:
    robot.velocity_y = -ROBOT_VELOCITY_Y
    robot.direction = 'up'
  elif moving_down:
    robot.velocity_y = ROBOT_VELOCITY_Y
    robot.direction = 'down'
  elif moving_right:
    robot.velocity_x = ROBOT_VELOCITY_X
    robot.direction = 'right'
  elif moving_left:
    robot.velocity_x = -ROBOT_VELOCITY_X
    robot.direction = 'left'



def draw():
  window.fill("#5FCB9B")

  # draws charge stations
  for station in charge_stations:
    window.blit(station.image, station)

  # draws drop off platforms
  for platform in dropoff_platforms:
    window.blit(platform.image, platform)

  # updates the robot image file as per curr state
  robot.update_image()
  window.blit(robot.image, robot)

  # draws shelves
  for shelf in shelves:
    window.blit(shelf.image, shelf)


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



# GAME STARTS HERE
robot = Robot()
charge_stations = []
shelves = []
dropoff_platforms = []
create_map()

while True:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      pygame.quit()
      exit()


  handle_movements()
  move()
  draw()
  pygame.display.update()
  clock.tick(60)
