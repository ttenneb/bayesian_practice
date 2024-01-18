import pygame
import sys
import numpy as np
import math 
import torch

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
FPS = 120

# Set up the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("My Pygame 2D Game")

# Set up the clock for managing the frame rate
clock = pygame.time.Clock()

class GameObject:
    def __init__(self, x, y, width, height, color):
        self.x, self.y = x, y
        self.width, self.height = width, height
        self.color = color
        self.velocity = [0, 0]
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def update(self):
        self.x += self.velocity[0]
        self.y += self.velocity[1]
        self.rect.x, self.rect.y = self.x, self.y

    def apply_force(self, force):
        self.velocity[0] += force[0]
        self.velocity[1] += force[1]

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)


def calculate_orbit_quality(moving_particle_path, stationary_particle_pos):
    """
    Calculate the quality of an orbit based on the alignment of the normal vector
    and the vector pointing towards the stationary particle.

    Parameters:
    moving_particle_path (list of tuples): A list containing the positions (x, y) of the moving particle.
    stationary_particle_pos (tuple): The position (x, y) of the stationary particle.

    Returns:
    float: The quality of the orbit, measured as the absolute value of the dot product of the normalized vectors.
    """
    # Ensure there are at least two points to calculate velocity
    if len(moving_particle_path) < 2:
        raise ValueError("Need at least two points to estimate velocity.")

    # Calculate velocity using the last two points
    v_x = moving_particle_path[-1][0] - moving_particle_path[-2][0]
    v_y = moving_particle_path[-1][1] - moving_particle_path[-2][1]
    velocity = np.array([v_x, v_y])

    # Calculate the normal vector (perpendicular to velocity)
    normal = np.array([-v_y, v_x])

    # Calculate vector from moving particle to stationary particle
    moving_particle_pos = moving_particle_path[-1]
    stationary_vector = np.array([stationary_particle_pos[0] - moving_particle_pos[0],
                                  stationary_particle_pos[1] - moving_particle_pos[1]])

    # Normalize the vectors
    normal_normalized = normal / np.linalg.norm(normal)
    stationary_vector_normalized = stationary_vector / np.linalg.norm(stationary_vector)

    # Calculate the dot product (absolute value for direction invariance)
    quality = abs(np.dot(normal_normalized, stationary_vector_normalized))

    return quality

def model_entropy(variance_list):
    joint_entropy = 0
    for v in variance_list:
        joint_entropy = .5*math.log(math.pi*math.e*math.pow(v, 2))

    return joint_entropy

def ELBO(likelihood, entropy):
    return math.log(likelihood) - math.log(entropy)

# Game loop
running = True

fx_a, fx_v = 1, .5
fy_a, fy_v = .2, .5

force_x = np.random.normal(fx_a, fx_v, 1)[0]
force_y = np.random.normal(fy_a, fy_v, 1)[0]

px_a, px_v = 0, 200
py_a, py_v = 0, 150

pos_x = SCREEN_WIDTH/2 + np.random.normal(px_a, px_v, 1)[0]
pos_y = SCREEN_HEIGHT/2 + np.random.normal(py_a, py_v, 1)[0]

object1 = GameObject(pos_x, pos_y, 5, 5, (255, 0, 0))
center = GameObject(SCREEN_WIDTH/2, SCREEN_HEIGHT/2, 5, 5, (0, 0, 255))
object1.apply_force((force_x, force_y))
orbit_tick = 0
in_range_tick = 1
last_path_point = (object1.x, object1.y)
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update game state
    # (Update positions, check for collisions, etc.)
    object1.update()
    # Revolve around center
    distance = [center.x-object1.x, center.y - object1.y]
    norm = (distance[0]**2 + distance[1]**2)

    gravity = ((100000000 / norm**2)*(distance[0]/norm), (100000000 / norm ** 2)*(distance[1]/norm) )
    object1.apply_force(gravity)
   
    # data collection
    if math.sqrt(math.pow((object1.x-center.x), 2) + math.pow((object1.y-center.y), 2)) < 400:
        in_range_tick +=1
        orbit_tick += calculate_orbit_quality([last_path_point, (object1.x, object1.y)], (center.x, center.y))

    if object1.x < 0 or object1.x > SCREEN_WIDTH or object1.y < 0 or object1.y > SCREEN_HEIGHT:
        break

    last_path_point = (object1.x, object1.y)

    # Render (draw) the game
    screen.fill((0, 0, 0))  # Fill the screen with black (or any other color)
    
    # (Draw your game elements here)
    object1.draw(screen)
    center.draw(screen)

    pygame.display.flip()  # Update the full display Surface to the screen

    # Cap the frame rate
    clock.tick(FPS)

print(orbit_tick)

# Clean up
pygame.quit()
sys.exit()