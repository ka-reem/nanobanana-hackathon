import os
import pygame
from PIL import Image

# Configuration
IMAGE_FILE = "images/elon-camera.png"  # Single image file
FRAME_DELAY = 500  # Delay between frames in milliseconds
GRID_SIZE = (4, 4)  # 4x4 grid


def crop_image(image_path, grid_size):
    """Crop a single image into a grid of smaller images."""
    image = Image.open(image_path)
    img_width, img_height = image.size
    cell_width = img_width // grid_size[1]
    cell_height = img_height // grid_size[0]

    cropped_images = []
    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            left = col * cell_width
            upper = row * cell_height
            right = left + cell_width
            lower = upper + cell_height
            cropped_image = image.crop((left, upper, right, lower))

            # Resize the cropped image to be 3x larger
            resized_image = cropped_image.resize((cell_width * 3, cell_height * 3))

            # Convert to a format pygame can use
            cropped_images.append(pygame.image.fromstring(
                resized_image.tobytes(), resized_image.size, resized_image.mode
            ))

    return cropped_images


def play_stop_motion(images, frame_delay):
    """Play the stop-motion animation."""
    pygame.init()

    # Calculate the display size based on the first image
    img_width, img_height = images[0].get_size()
    screen = pygame.display.set_mode((img_width, img_height))
    pygame.display.set_caption("Stop Motion Video")

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Play the animation frame by frame
        for image in images:
            screen.fill((0, 0, 0))  # Clear the screen
            screen.blit(image, (0, 0))  # Display the image full screen
            pygame.display.flip()
            pygame.time.delay(frame_delay)

        clock.tick(60)

    pygame.quit()


def main():
    # Crop the single image into a grid of smaller images
    images = crop_image(IMAGE_FILE, GRID_SIZE)
    if len(images) != GRID_SIZE[0] * GRID_SIZE[1]:
        print(f"Error: Expected {GRID_SIZE[0] * GRID_SIZE[1]} images, but found {len(images)}.")
        return

    # Play the stop-motion video
    play_stop_motion(images, FRAME_DELAY)


if __name__ == "__main__":
    main()