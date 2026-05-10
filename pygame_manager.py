import pygame

_initialized = False


def ensure_init(width, height, caption="Warehouse"):
    global _initialized
    if not _initialized:
        pygame.init()
        _initialized = True

    if pygame.display.get_surface() is None:
        pygame.display.set_mode((width, height))
        pygame.display.set_caption(caption)

    return pygame.display.get_surface()