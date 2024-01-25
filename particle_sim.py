import pygame
import sys
import numpy as np
import math 
import torch
import torch.distributions as dist
import scipy.stats as stats
import matplotlib.pyplot as plt

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
FPS = 60
lr = .1
steps = 1
entropy_norm = 2
batch_size = 100
# Set up the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Orbit")

# Set up the clock for managing the frame rate
clock = pygame.time.Clock()

class GameObject:
    def __init__(self, x, y, width, height, initial_velocity):

        self.position = torch.stack([x, y], dim=-1).reshape(-1).to(dtype=torch.float32)
        self.velocity = initial_velocity

        self.width, self.height = width, height

    def update(self):
        self.position = self.position + self.velocity

    def apply_force(self, force):
        self.velocity = self.velocity + force

def orbit_quality(object1, stationary_position):
    # Calculate velocity
    velocity = object1.velocity

    # Calculate the normal vector (perpendicular to velocity)
    normal = torch.tensor([-velocity[1], velocity[0]], dtype=torch.float32)

    # Calculate vector from moving particle (current position) to stationary particle
    stationary_vector = stationary_position - object1.position

    # Normalize the vectors
    normal_normalized = normal / torch.norm(normal)
    stationary_vector_normalized = stationary_vector / torch.norm(stationary_vector)

    # Calculate the dot product and square it
    quality = torch.dot(normal_normalized, stationary_vector_normalized)**2

    return quality

def plot_normal(mu_0, variance_0, mu_t, variance_t):
    fig, axs = plt.subplots(1, 2) 

    sigma = math.sqrt(variance_0)
    x = np.linspace(mu_0 - 3*sigma, mu_0 + 3*sigma, 100)
    axs[0].plot(x, stats.norm.pdf(x, mu_0, sigma))

    sigma = math.sqrt(variance_t)
    x = np.linspace(mu_t - 3*sigma, mu_t + 3*sigma, 100)
    axs[1].plot(x, stats.norm.pdf(x, mu_t, sigma))

    plt.show()

def model_entropy(variance_list):
    return 0.5 * torch.sum(torch.log(2 * torch.pi * torch.e * (variance_list**2)))

def ELBO(likelihood, entropy):
    return math.log(likelihood) - math.log(entropy)

center_x = torch.tensor([SCREEN_WIDTH/2], requires_grad=True)
center_y = torch.tensor([SCREEN_HEIGHT/2], requires_grad=True) 

elbo_list = []

trajectory_best = []
elbo_best = -10.0


fx_a, fx_v = torch.tensor([-0.10000000149011612], requires_grad=True), torch.tensor([-0.09999984502792358], requires_grad=True)
fy_a, fy_v = torch.tensor([0.30000007152557373], requires_grad=True), torch.tensor([0.09999991953372955], requires_grad=True)
px_a, px_v = torch.tensor([0.7600], requires_grad=True), torch.tensor([0.0000052], requires_grad=True)
py_a, py_v = torch.tensor([-1.4], requires_grad=True), torch.tensor([0.00000005], requires_grad=True)

# optimizer = torch.optim.Adam([fx_a, fx_v, fy_a, fy_v, px_a, px_v, py_a, py_v], lr=lr)
optimizer = torch.optim.Adam([fx_a, fx_v, fy_a, fy_v], lr=lr)

for _ in range(steps):
    trajectory = []
    orbit_tick = torch.tensor([0.0], requires_grad=True)
    in_range_tick = torch.tensor([1.0], requires_grad=True)
    velocity_x = dist.Normal(fx_a, fx_v**2)
    velocity_y = dist.Normal(fy_a, fy_v**2)
    
    for i in range(batch_size):
        

        pos_x = center_x + dist.Normal(px_a*100, (px_v**2)*1000).rsample()
        pos_y = center_y + dist.Normal(py_a*100, (py_v**2)*1000).rsample()
        initial_velocity =  torch.stack([velocity_x.rsample(), velocity_y.rsample()], dim=-1).reshape(-1).to(dtype=torch.float32)
        object1 = GameObject(pos_x, pos_y, 5, 5, initial_velocity)
        center = GameObject(center_x, center_y, 5, 5, torch.tensor([0.0, 0.0], dtype=torch.float32, requires_grad=True))

        
        last_path_point = object1.position

        if _ == 0:
            fx_a_start, fx_v_start = float(fx_a), float(fx_v)
            fy_a_start, fy_v_start = float(fy_a), float(fy_v)
            px_a_start, px_v_start = float(px_a), float(px_v)
            py_a_start, py_v_start = float(py_a), float(py_v)

        while True:
        # Update game state
            # (Update positions, check for collisions, etc.)
            object1.update()
            # Revolve around center
            distance = center.position - object1.position
            norm = torch.norm(distance, p=2)**2
            gravity_constant = 10000000  # You may adjust this constant as needed
            gravity = gravity_constant / norm**2 * distance / norm
            object1.apply_force(gravity)
        
            in_range_tick = in_range_tick + 1
            orbit_tick = orbit_tick + orbit_quality(object1, center.position)

            if object1.position[0] < 0 or object1.position[0] > SCREEN_WIDTH or object1.position[1] < 0 or object1.position[1] > SCREEN_HEIGHT:
                break
            
            last_path_point = object1.position
            trajectory.append(last_path_point)

    # variance_tensor = torch.stack([fx_v, fy_v, px_v, py_v], dim=-1).reshape(-1).to(dtype=torch.float32)
    variance_tensor = torch.stack([fx_v, fy_v], dim=-1).reshape(-1).to(dtype=torch.float32)

    likelihood = orbit_tick/in_range_tick
    entropy = model_entropy(variance_tensor)/entropy_norm
    
    elbo = torch.log(likelihood) - torch.log(entropy)
    print("Objectives " + str(float(torch.log(likelihood))), float(torch.log(entropy)), float(elbo))

    if elbo > elbo_best:
        elbo_best = elbo.clone()
        trajectory_best = trajectory.copy()

    elbo_list.append(float(elbo))

    elbo.backward()

    print("Gradients:")
    print("fx_a:", float(fx_a))
    print("fx_v:", float(fx_v))
    print("fy_a:", float(fy_a))
    print("fy_v:", float(fy_v))
    # print("px_a:", float(px_a))
    # print("px_v:", float(px_v))
    # print("py_a:", float(py_a))
    # print("py_v:", float(py_v))

    # optimizer = torch.optim.Adam([fx_a, fx_v, fy_a, fy_v, px_a, px_v, py_a, py_v], lr=lr)
    optimizer = torch.optim.Adam([fx_a, fx_v, fy_a, fy_v], lr=lr)

    # Inside your game loop or after certain iterations
    optimizer.step()  # Adjust parameters
    optimizer.zero_grad()  # Clear gradients for the next iteration

print(elbo_best)
for p in trajectory_best:
    # Handle  events
    find = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            find=True
    if find:
        break
    # Render (draw) the game
    screen.fill((0, 0, 0))  # Fill the screen with black (or any other color)
    x, y = p.tolist()
    # (Draw your game elements here)
    pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(x, y, 5, 5))
    
    pygame.draw.rect(screen, (0, 0, 255), pygame.Rect(SCREEN_WIDTH/2, SCREEN_HEIGHT/2, 5, 5))

    pygame.display.flip()  # Update the full display Surface to the screen
        
    # Cap the frame rate
    clock.tick(FPS)

time_points = list(range(len(elbo_list)))

# Create the plot

print("FX")
print(fx_a_start, fx_v_start, fx_a, fx_v)
print("FY")
print(fy_a_start, fy_v_start, fy_a, fy_v)
# print("PX")
# print(px_a_start, px_v_start, px_a, px_v)
# print("PY")
# print(py_a_start, py_v_start, py_a, py_v)

# Show the plot

plot_normal(fx_a_start, fx_v_start**2, float(fx_a), float(fx_v**2))

plot_normal(fy_a_start, fy_v_start**2, float(fy_a), float(fy_v**2))

# plot_normal(px_a_start, (px_v_start**2), float(px_a), float((px_v**2)))

# plot_normal(py_a_start, (py_v_start**2), float(py_a), float((py_v**2)))


print(elbo)
for p in trajectory:
    # Handle events
    find = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            find=True
    if find:
        break

    # Render (draw) the game
    screen.fill((0, 0, 0))  # Fill the screen with black (or any other color)
    x, y = p.tolist()
    # (Draw your game elements here)
    pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(x, y, 5, 5))
    
    pygame.draw.rect(screen, (0, 0, 255), pygame.Rect(SCREEN_WIDTH/2, SCREEN_HEIGHT/2, 5, 5))

    pygame.display.flip()  # Update the full display Surface to the screen
        
    # Cap the frame rate
    clock.tick(FPS)


print("TRAJECTORY")
trajectory = [tuple(tensor.tolist()) for tensor in trajectory]
print(trajectory)
print("TRAJECTORY BEST")
trajectory_best = [tuple(tensor.tolist()) for tensor in trajectory_best]
print(trajectory_best)
plt.plot(time_points, elbo_list)
# Add labels and title
plt.xlabel('Time (Iterations)')
plt.ylabel('ELBO')
plt.title('ELBO Over Time')
plt.show()
# Clean up
pygame.quit()
sys.exit()