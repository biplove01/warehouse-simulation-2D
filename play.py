import pygame
import sys
import random
from collections import deque
from world import create_map
from constants import *
from robot import Robot


# ─── CONSTANTS ───────────────────────────────────────────────────────────────

BACKGROUND_COLOR = (30, 30, 30)
HUD_COLOR        = (20, 20, 20)
TEXT_COLOR       = (220, 220, 220)
ACCENT_COLOR     = (255, 215, 0)
SUCCESS_COLOR    = (80, 220, 120)
FAIL_COLOR       = (220, 80, 80)

HUD_HEIGHT       = 50
NOTIFICATION_DURATION = 90   # frames


# ─── HELPER: BFS DISTANCE MAP ────────────────────────────────────────────────

def bfs_distance_map(start_grid_x, start_grid_y, obstacle_positions):
    """Returns a dict mapping (grid_x, grid_y) → BFS distance from the start."""
    distance_map = {(start_grid_x, start_grid_y): 0}
    search_queue  = deque([(start_grid_x, start_grid_y, 0)])

    while search_queue:
        current_x, current_y, current_dist = search_queue.popleft()

        for delta_x, delta_y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor_x = current_x + delta_x
            neighbor_y = current_y + delta_y

            is_in_bounds  = 0 <= neighbor_x < GRID_WIDTH and 0 <= neighbor_y < GRID_HEIGHT
            is_free_cell  = (neighbor_x, neighbor_y) not in obstacle_positions
            is_unvisited  = (neighbor_x, neighbor_y) not in distance_map

            if is_in_bounds and is_free_cell and is_unvisited:
                distance_map[(neighbor_x, neighbor_y)] = current_dist + 1
                search_queue.append((neighbor_x, neighbor_y, current_dist + 1))

    return distance_map


def grid_coords_from_object(obj):
    """Convert a world object's pixel position to grid coordinates."""
    grid_x = round((obj.x - PADDING_BORDER) / GRID_SPACING)
    grid_y = round((obj.y - PADDING_BORDER) / GRID_SPACING)
    return grid_x, grid_y


# ─── NOTIFICATION SYSTEM ─────────────────────────────────────────────────────

class Notification:
    def __init__(self, message, color, duration=NOTIFICATION_DURATION):
        self.message          = message
        self.color            = color
        self.frames_remaining = duration

    @property
    def is_alive(self):
        return self.frames_remaining > 0

    def tick(self):
        self.frames_remaining -= 1

    @property
    def alpha(self):
        """Fade out in the last 30 frames."""
        return min(255, int(255 * self.frames_remaining / 30))


# ─── MAIN GAME ───────────────────────────────────────────────────────────────

class Game:

    def __init__(self):
        pygame.init()
        pygame.font.init()

        self.window_width  = GRID_WIDTH  * GRID_SPACING + 2 * PADDING_BORDER
        self.window_height = GRID_HEIGHT * GRID_SPACING + 2 * PADDING_BORDER + HUD_HEIGHT

        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Warehouse — Manual Play")
        self.clock  = pygame.time.Clock()

        self.font_large  = pygame.font.SysFont("monospace", 22, bold=True)
        self.font_medium = pygame.font.SysFont("monospace", 16)
        self.font_small  = pygame.font.SysFont("monospace", 13)

        # ── World ────────────────────────────────────────────────────────────
        self.shelves, self.charge_stations, self.dropoff_platforms = create_map()

        self.obstacle_positions = {
            grid_coords_from_object(obj)
            for obj in self.shelves + self.dropoff_platforms
        }

        # ── Dropoff (central platform) ───────────────────────────────────────
        central_platform = self.dropoff_platforms[len(self.dropoff_platforms) // 2]
        self.dropoff_grid_x, self.dropoff_grid_y = grid_coords_from_object(central_platform)

        # ── Robot ────────────────────────────────────────────────────────────
        self.robot = Robot(start_x=3, start_y=3)

        # ── Game state ───────────────────────────────────────────────────────
        self.score                = 0
        self.steps                = 0
        self.target_grid_x        = 0
        self.target_grid_y        = 0
        self.notifications: list[Notification] = []

        self._spawn_new_target()

    # ── Target management ────────────────────────────────────────────────────

    def _spawn_new_target(self):
        """Clear all shelves and place a box on a random one."""
        for shelf in self.shelves:
            shelf.has_box = False
            shelf.image   = shelf.empty_image

        new_target_shelf = random.choice(self.shelves)
        new_target_shelf.has_box = True
        new_target_shelf.image   = new_target_shelf.loaded_image

        self.target_grid_x, self.target_grid_y = grid_coords_from_object(new_target_shelf)

    # ── Input handling ───────────────────────────────────────────────────────

    def _handle_movement(self, direction):
        """Move the robot one cell in the given direction if the cell is free."""
        direction_to_delta = {
            "up":    (0, -1),
            "down":  (0,  1),
            "left":  (-1, 0),
            "right": (1,  0),
        }
        delta_x, delta_y = direction_to_delta[direction]
        next_x = self.robot.grid_x + delta_x
        next_y = self.robot.grid_y + delta_y

        is_in_bounds = 0 <= next_x < GRID_WIDTH and 0 <= next_y < GRID_HEIGHT
        is_passable  = (next_x, next_y) not in self.obstacle_positions

        if is_in_bounds and is_passable:
            self.robot.grid_x      = next_x
            self.robot.grid_y      = next_y
            self.robot.last_direction = direction
            self.steps += 1
        else:
            self.notifications.append(
                Notification("Blocked!", FAIL_COLOR, duration=40)
            )

        self.robot.update_image()

    def _handle_interact(self):
        """Pick up from the target shelf or drop off at the dropoff platform."""
        robot = self.robot

        dist_to_target_shelf = (
            abs(robot.grid_x - self.target_grid_x)
            + abs(robot.grid_y - self.target_grid_y)
        )
        dist_to_dropoff = (
            abs(robot.grid_x - self.dropoff_grid_x)
            + abs(robot.grid_y - self.dropoff_grid_y)
        )

        if not robot.loaded and dist_to_target_shelf == 1:
            robot.loaded = True
            robot.update_image()
            self.notifications.append(
                Notification("📦  Picked up!", SUCCESS_COLOR)
            )

        elif robot.loaded and dist_to_dropoff <= 2:
            robot.loaded = False
            robot.update_image()
            self.score += 1
            self.notifications.append(
                Notification(f"✔  Delivered!  Score: {self.score}", ACCENT_COLOR)
            )
            self._spawn_new_target()

        else:
            hint = "Get adjacent to the shelf first!" if not robot.loaded else "Get closer to the dropoff!"
            self.notifications.append(
                Notification(hint, FAIL_COLOR, duration=60)
            )

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    self._handle_movement("up")
                elif event.key == pygame.K_s:
                    self._handle_movement("down")
                elif event.key == pygame.K_a:
                    self._handle_movement("left")
                elif event.key == pygame.K_d:
                    self._handle_movement("right")
                elif event.key == pygame.K_e:
                    self._handle_interact()
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

    # ── Rendering ────────────────────────────────────────────────────────────

    def _draw_world(self):
        # Background
        self.screen.fill(BACKGROUND_COLOR)

        # Charge stations
        for charge_station in self.charge_stations:
            self.screen.blit(charge_station.image, (charge_station.x, charge_station.y))

        # Dropoff platforms
        for dropoff_platform in self.dropoff_platforms:
            self.screen.blit(dropoff_platform.image, (dropoff_platform.x, dropoff_platform.y))

        # Shelves (shadow → highlight → image)
        for shelf in self.shelves:
            shelf_grid_position = grid_coords_from_object(shelf)
            is_current_target   = (
                not self.robot.loaded
                and shelf_grid_position == (self.target_grid_x, self.target_grid_y)
            )

            # Shadow
            self.screen.blit(shelf.shadow_image, (shelf.x - 1, shelf.y + 4))

            # Yellow highlight ring around the target shelf
            if is_current_target:
                pygame.draw.rect(
                    self.screen,
                    ACCENT_COLOR,
                    (shelf.x - 3, shelf.y - 3, TILE_SIZE + 6, TILE_SIZE + 6),
                    2,
                )

            self.screen.blit(shelf.image, (shelf.x, shelf.y))

        # Dropoff highlight ring when robot is carrying a box
        if self.robot.loaded:
            dropoff_pixel_x = PADDING_BORDER + self.dropoff_grid_x * GRID_SPACING
            dropoff_pixel_y = PADDING_BORDER + self.dropoff_grid_y * GRID_SPACING
            pygame.draw.rect(
                self.screen,
                SUCCESS_COLOR,
                (dropoff_pixel_x - 3, dropoff_pixel_y - 3, TILE_SIZE + 6, TILE_SIZE + 6),
                2,
            )

    def _draw_robot(self):
        robot_pixel_x = PADDING_BORDER + self.robot.grid_x * GRID_SPACING
        robot_pixel_y = PADDING_BORDER + self.robot.grid_y * GRID_SPACING

        # Change last_direction to direction here
        if self.robot.direction in ("left", "right"):
            robot_image = ROBOT_IMAGE_SIDE_BOX if self.robot.loaded else ROBOT_IMAGE_SIDE
        else:
            robot_image = ROBOT_IMAGE_VERTICAL_BOX if self.robot.loaded else ROBOT_IMAGE_VERTICAL

        center_offset_x = (TILE_SIZE - ROBOT_WIDTH)  // 2
        center_offset_y = (TILE_SIZE - ROBOT_HEIGHT) // 2
        self.screen.blit(robot_image, (robot_pixel_x + center_offset_x, robot_pixel_y + center_offset_y))

    def _draw_hud(self):
        hud_y = GRID_HEIGHT * GRID_SPACING + 2 * PADDING_BORDER
        pygame.draw.rect(self.screen, HUD_COLOR, (0, hud_y, self.window_width, HUD_HEIGHT))
        pygame.draw.line(self.screen, (60, 60, 60), (0, hud_y), (self.window_width, hud_y), 1)

        score_surface = self.font_large.render(f"Score: {self.score}", True, ACCENT_COLOR)
        steps_surface = self.font_medium.render(f"Steps: {self.steps}", True, TEXT_COLOR)
        state_text    = "CARRYING BOX" if self.robot.loaded else "EMPTY"
        state_color   = SUCCESS_COLOR if self.robot.loaded else TEXT_COLOR
        state_surface = self.font_medium.render(state_text, True, state_color)
        keys_surface  = self.font_small.render("WASD: Move    E: Interact    ESC: Quit", True, (120, 120, 120))

        self.screen.blit(score_surface, (12, hud_y + 8))
        self.screen.blit(steps_surface, (160, hud_y + 10))
        self.screen.blit(state_surface, (280, hud_y + 10))
        self.screen.blit(keys_surface,  (self.window_width - 340, hud_y + 18))

    def _draw_notifications(self):
        """Draw stacked notification toasts near the top-center of the screen."""
        active_notifications = [n for n in self.notifications if n.is_alive]
        self.notifications   = active_notifications

        for index, notification in enumerate(reversed(active_notifications[-4:])):
            text_surface = self.font_medium.render(notification.message, True, notification.color)
            text_width   = text_surface.get_width()
            toast_x      = (self.window_width - text_width) // 2
            toast_y      = 12 + index * 26

            # Semi-transparent background pill
            background_rect = pygame.Rect(toast_x - 10, toast_y - 4, text_width + 20, 24)
            background_surf = pygame.Surface((background_rect.width, background_rect.height), pygame.SRCALPHA)
            background_surf.fill((0, 0, 0, 160))
            self.screen.blit(background_surf, background_rect.topleft)

            self.screen.blit(text_surface, (toast_x, toast_y))
            notification.tick()

    def render(self):
        self._draw_world()
        self._draw_robot()
        self._draw_hud()
        self._draw_notifications()
        pygame.display.flip()

    # ── Game loop ────────────────────────────────────────────────────────────

    def run(self):
        while True:
            self.handle_events()
            self.render()
            self.clock.tick(60)


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    game = Game()
    game.run()