import pygame
from sys import exit
import random

from constants import *
from world import create_map
from robot import Robot

pygame.init()
window = pygame.display.set_mode((GRID_WIDTH * GRID_SPACING, GRID_HEIGHT * GRID_SPACING))
pygame.display.set_caption("Warehouse Simulation")
pygame.display.set_icon(ROBOT_IMAGE_SIDE)
clock = pygame.time.Clock()

score = 0

ast_move_time = 0
move_delay = 0.2
current_direction = None


robot = Robot(start_x=0, start_y=0)
shelves, charge_stations, dropoff_platforms = create_map()


def render():
  window.fill("#5FCB9B")

  for obj in  charge_stations + dropoff_platforms:
    window.blit(obj.image, obj)


  # shadows
  for obj in shelves:
    window.blit(obj.shadow_image, (obj.x -3, obj.y +12))

  robot_rect = robot.get_pixel_rect()
  window.blit(robot.image, (robot_rect.x, robot_rect.y))

  # actual shelves
  for obj in shelves:
    window.blit(obj.image, obj)

  pygame.display.update()


def respawn_box(passed_shelves):
    empty_shelves = [shelf for shelf in passed_shelves if not shelf.has_box]

    if empty_shelves:
        new_box_shelf = random.choice(empty_shelves)
        new_box_shelf.has_box = True
        new_box_shelf.image = new_box_shelf.loaded_image


# GAME STARTS HERE
last_keys = pygame.key.get_pressed()


while True:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      pygame.quit()
      exit()

  keys = pygame.key.get_pressed()

  if keys[pygame.K_e]:
    if robot.loaded:
      if robot.drop_box(dropoff_platforms):
        score += 1
        print(f"Total score: {score}")
        respawn_box(shelves)
    else:
      robot.pickup_box(shelves)

  if keys[pygame.K_SPACE]:
    respawn_box(shelves)

  direction_keys = {
    'up':    (pygame.K_UP, pygame.K_w),
    'down':  (pygame.K_DOWN, pygame.K_s),
    'left':  (pygame.K_LEFT, pygame.K_a),
    'right': (pygame.K_RIGHT, pygame.K_d)
  }

  moved = False
  for dir_name, (k1, k2) in direction_keys.items():
    if (keys[k1] or keys[k2]) and not (last_keys[k1] or last_keys[k2]):
      # Key was just pressed!
      robot.handle_inputs_single(dir_name, shelves)
      robot.handle_inputs_single(dir_name, dropoff_platforms)
      moved = True
      break

  last_keys = keys

  render()

  clock.tick(30)
