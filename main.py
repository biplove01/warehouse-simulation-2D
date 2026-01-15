import pygame
from sys import exit

from constants import *
from world import create_map
from robot import Robot

pygame.init()
window = pygame.display.set_mode((GAME_WIDTH, GAME_HEIGHT))
pygame.display.set_caption("Warehouse Simulation")
pygame.display.set_icon(ROBOT_IMAGE_SIDE)
clock = pygame.time.Clock()


robot = Robot()
shelves, charge_stations, dropoff_platforms = create_map()


def render():
  window.fill("#5FCB9B")

  for obj in  charge_stations + dropoff_platforms:
    window.blit(obj.image, obj)


  # shadows
  for obj in shelves:
    window.blit(obj.shadow_image, (obj.x -3, obj.y +12))

  window.blit(robot.image, robot)

  # actual shelves
  for obj in shelves:
    window.blit(obj.image, obj)

  pygame.display.update()



# GAME STARTS HERE

while True:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      pygame.quit()
      exit()

  keys = pygame.key.get_pressed()

  if keys[pygame.K_e]:
    robot.pickup_box(shelves)

  robot.handle_inputs()
  robot.handle_physics(shelves)
  render()

  clock.tick(60)
