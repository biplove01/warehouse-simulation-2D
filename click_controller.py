import sys
import pygame
from constants import PADDING_BORDER, GRID_SPACING, GRID_WIDTH, GRID_HEIGHT

class ClickController:
    """
    Listens for mouse clicks on the pygame window and converts them
    to shelf grid coordinates, then enqueues them in the environment.
    """

    def __init__(self, env):
        self.env = env

    def handle_pygame_events(self):
        """
        Call this every step. Processes all pending pygame events
        and enqueues any shelf clicks into the environment's target queue.
        """
        for pygame_event in pygame.event.get():
            if pygame_event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if pygame_event.type == pygame.MOUSEBUTTONDOWN and pygame_event.button == 1:
                mouse_pixel_x, mouse_pixel_y = pygame_event.pos
                self._handle_shelf_click(mouse_pixel_x, mouse_pixel_y)

    def _handle_shelf_click(self, mouse_pixel_x, mouse_pixel_y):
        """Convert pixel click position to grid coords and enqueue if it's a shelf."""
        clicked_grid_x = round((mouse_pixel_x - PADDING_BORDER) / GRID_SPACING)
        clicked_grid_y = round((mouse_pixel_y - PADDING_BORDER) / GRID_SPACING)

        is_in_bounds = 0 <= clicked_grid_x < GRID_WIDTH and 0 <= clicked_grid_y < GRID_HEIGHT
        if not is_in_bounds:
            print(f"  ✗ Click out of bounds, ignored.")
            return

        for shelf in self.env.shelves:
            shelf_grid_x, shelf_grid_y = self.env._to_grid_coords(shelf)
            if shelf_grid_x == clicked_grid_x and shelf_grid_y == clicked_grid_y:
                # Visually mark the shelf as having a box immediately
                shelf.has_box = True
                shelf.image = shelf.loaded_image
                self.env.enqueue_target(clicked_grid_x, clicked_grid_y)
                return

        print(f" No shelf at grid ({clicked_grid_x}, {clicked_grid_y}), click ignored.")