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
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 800
RUN_LIMIT = 5000
FPS = 320
lr = .12
steps = 200
entropy_norm = 10
batch_size = 50
gravity_constant = 50
# Set up the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Orbit")

# Set up the clock for managing the frame rate
clock = pygame.time.Clock()

class GameObject:
    def __init__(self, x, y, width, height, initial_velocity):

        self.position = torch.squeeze(torch.stack([x, y], dim=-1).to(dtype=torch.float32))
        self.velocity = torch.squeeze(initial_velocity)
        self.width, self.height = width, height

    def update(self):
        self.position = self.position + self.velocity
        self.position.clamp(min=-5000, max=5000)
    def apply_force(self, force):
        self.velocity = self.velocity + force

def orbit_quality(object1, stationary_positions):
# with torch.autograd.detect_anomaly():
    # Extract velocities from objects
    velocities = object1.velocity

    # Calculate the normal vectors (perpendicular to velocities)
    # We use a negative sign for the second component to get the perpendicular vector
    normals = torch.stack([-velocities[:, 1], velocities[:, 0]], dim=1)

    # Calculate vectors from moving particles (current positions) to stationary particles
    epsilon = 1e-8
    
    stationary_vectors = (stationary_positions.detach() - object1.position)

    # Normalize the vectors
    # We add a small epsilon to avoid division by zero
    

    normals_normalized = normals / (torch.norm(normals, dim=1, keepdim=True) + epsilon)
    
    stationary_vectors_normalized = stationary_vectors / (torch.norm(stationary_vectors, dim=1, keepdim=True) + epsilon)
    # Calculate the dot product and square it

    result = torch.sum((normals_normalized * stationary_vectors_normalized)**2, dim=1, keepdim=True)
# Take the average of the results
    average = torch.mean(result)
    # print(average)
    return average

def plot_normal(mu_0, variance_0, mu_t, variance_t):
    fig, axs = plt.subplots(1, 2) 

    sigma = math.sqrt(variance_0)
    x = np.linspace(mu_0 - 3*sigma, mu_0 + 3*sigma, 100)
    axs[0].plot(x, stats.norm.pdf(x, mu_0, sigma))

    sigma = math.sqrt(variance_t)
    x = np.linspace(mu_t - 3*sigma, mu_t + 3*sigma, 100)
    axs[1].plot(x, stats.norm.pdf(x, mu_t, sigma))

    plt.show()

center_x = torch.full((batch_size,), SCREEN_WIDTH / 2, requires_grad=True)
center_y = torch.full((batch_size,), SCREEN_HEIGHT / 2, requires_grad=True)

elbo_list = []

trajectory_best = []
elbo_best = 10.0
epsilon = 1e-6

fx_a, fx_v = torch.tensor([epsilon], requires_grad=True), torch.tensor([0.5], requires_grad=True)
fy_a, fy_v = torch.tensor([epsilon], requires_grad=True), torch.tensor([0.5], requires_grad=True)
px_a, px_v = torch.tensor([epsilon], requires_grad=True), torch.tensor([0.66], requires_grad=True)
py_a, py_v = torch.tensor([epsilon], requires_grad=True), torch.tensor([0.66], requires_grad=True)


optimizer = torch.optim.Adamax([fx_a, fx_v, fy_a, fy_v, px_a, px_v, py_a, py_v], lr=lr)
# optimizer = torch.optim.Adam([fx_a, fx_v, fy_a, fy_v], lr=lr)
distances = []
torch.autograd.set_detect_anomaly(True)
try:
    for _ in range(steps):
        trajectory = []
        orbit_tick = torch.tensor([0.0], requires_grad=True)
        in_range_tick = torch.tensor([0.0], requires_grad=True)
        velocity_x = dist.Normal(fx_a, (fx_v+epsilon)**2)
        velocity_y = dist.Normal(fy_a, (fy_v+epsilon)**2)


        pos_x =  dist.Normal((px_a)*SCREEN_WIDTH/2, (px_v ** 2)* 200)
        pos_y =  dist.Normal((py_a)*SCREEN_HEIGHT/2, (py_v ** 2)* 200)
        initial_velocity =  torch.stack([velocity_x.rsample((batch_size,)), velocity_y.rsample((batch_size,))], dim=-1).to(dtype=torch.float32)
        # initial_velocity =  torch.stack([torch.zeros((batch_size, 1)), torch.zeros((batch_size, 1))], dim=-1).to(dtype=torch.float32)
        start_x = torch.reshape(center_x, (batch_size, 1)) + torch.reshape(pos_x.rsample((batch_size,1)), (batch_size, 1))
        start_y = torch.reshape(center_y, (batch_size, 1)) + torch.reshape(pos_y.rsample((batch_size,1)), (batch_size, 1))
        objects = GameObject(start_x, start_y, 5, 5, initial_velocity)
        center = GameObject(center_x, center_y, 5, 5, torch.tensor([0.0, 0.0], dtype=torch.float32, requires_grad=False))
        last_path_point = objects.position
    
        if _ == 0:
            fx_a_start, fx_v_start = float(fx_a), float(fx_v)
            fy_a_start, fy_v_start = float(fy_a), float(fy_v)
            px_a_start, px_v_start = float(px_a), float(px_v)
            py_a_start, py_v_start = float(py_a), float(py_v)
    # with torch.autograd.detect_anomaly():
        while True:
            # Update game state
            objects.update()
            epsilon = 1e-6
            # Revolve around center
            distance = center.position.detach() - objects.position 
            if(torch.any(distance) == 0):
                print("ZEROSSS")
            if(torch.all(distance) != 0):
                norms = torch.norm(distance, p=2, dim=1, keepdim=True)
                # You may adjust this constant as needed
                
                gravity = (gravity_constant / (norms**2 + epsilon)) * (distance / (norms + epsilon)) + epsilon

                objects.apply_force(gravity)

            

            in_range_tick = in_range_tick + 1

            last_path_point = objects.position
            trajectory.append(last_path_point)


            orbit_count = 0
            out_orbit_count = 0
            for i in range(batch_size):
                if objects.position[i, 0] > 0 and objects.position[i, 0] < SCREEN_WIDTH and objects.position[i, 1] > 0 and objects.position[i, 1] < SCREEN_HEIGHT:
                    orbit_count += 1
                else:
                    out_orbit_count += 1
            
            orbit_tick = orbit_tick + (((orbit_count)**2)*(orbit_quality(objects, center.position)))*(in_range_tick/RUN_LIMIT)
            
            done = False
            if in_range_tick > RUN_LIMIT:
                done = True
            if done:
                print("simulation done")

                # RENDER EVERY STEP
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
                    points = p.tolist()
                    # (Draw your game elements here)
                    for i in range(batch_size):
                        pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(points[i][0], points[i][1], 5, 5))
                    
                    pygame.draw.rect(screen, (0, 0, 255), pygame.Rect(SCREEN_WIDTH/2, SCREEN_HEIGHT/2, 5, 5))

                    pygame.display.flip()  # Update the full display Surface to the screen
                        
                    # Cap the frame rate
                    clock.tick(FPS)
                break
        print("ORBIT TICK: ", orbit_tick, "RUN LIMIT: ", RUN_LIMIT, "BATCH SIZE: ", batch_size)
        likelihood = torch.log((orbit_tick/RUN_LIMIT)/batch_size)
        entropy = (pos_x.entropy() + pos_y.entropy() + velocity_x.entropy() + velocity_y.entropy())/entropy_norm

        elbo = -1*likelihood - entropy 
        print("Params:")
        print("fx_a:", float(fx_a))
        print("fx_v:", float(fx_v))
        print("fy_a:", float(fy_a))
        print("fy_v:", float(fy_v))
        print("px_a:", float(px_a))
        print("px_v:", float(px_v))
        print("py_a:", float(py_a))
        print("py_v:", float(py_v))

        print("Objectives:  log Likelihood: " + str(float(-1*likelihood)), "    Entropy: " + str(float(entropy)), "   Negative ELBO: " + str(float(elbo)))
        orbit_tick.retain_grad()
        in_range_tick.retain_grad()
        elbo.retain_grad()
        elbo.backward()
        
        # Gradient clipping
        # torch.nn.utils.clip_grad_norm_(fx_a, 1)
        # torch.nn.utils.clip_grad_norm_(fy_a, 1)
        # torch.nn.utils.clip_grad_norm_(fx_v, 1)
        # torch.nn.utils.clip_grad_norm_(fy_v, 1)
        # torch.nn.utils.clip_grad_norm_(px_a, 1)
        # torch.nn.utils.clip_grad_norm_(py_a, 1)
        # torch.nn.utils.clip_grad_norm_(px_v, 1)
        # torch.nn.utils.clip_grad_norm_(py_v, 1)
        print(fx_a.grad, fy_a.grad, fx_v.grad, fy_v.grad, px_a.grad, py_a.grad, px_v.grad, py_v.grad)
        max_grad = max([fx_a.grad, fx_v.grad, fy_a.grad, fy_v.grad, px_a.grad, px_v.grad,py_a.grad, py_v.grad])
        min_grad = min([fx_a.grad, fx_v.grad, fy_a.grad, fy_v.grad, px_a.grad, px_v.grad,py_a.grad, py_v.grad])

        fx_v.grad = (fx_v.grad/max_grad)
        fx_a.grad = (fx_a.grad/max_grad)
        fy_a.grad = (fy_a.grad/max_grad)
        fy_v.grad = (fy_v.grad/max_grad)
        px_a.grad = (px_a.grad/max_grad)
        px_v.grad = (px_v.grad/max_grad)
        py_a.grad = (py_a.grad/max_grad)
        py_v.grad = (py_v.grad/max_grad)
        # norm = max([torch.abs(fx_a.grad), torch.abs(fx_v.grad), torch.abs(fy_a.grad), torch.abs(fy_v.grad)])
        # fx_a.grad = fx_a.grad / norm
        # fx_v.grad = fx_v.grad / norm
        # fy_a.grad = fx_a.grad / norm
        # fy_v.grad = fx_v.grad / norm

        print(fx_a.grad, fy_a.grad, fx_v.grad, fy_v.grad, px_a.grad, py_a.grad, px_v.grad, py_v.grad)
        print(orbit_tick.grad, in_range_tick.grad)
        print(elbo.grad)
        # Inside your game loop or after certain iterations
        optimizer.step() 
        if elbo.detach() < elbo_best:
            elbo_best = elbo.detach().clone()
            trajectory_best = trajectory.copy()

        elbo_list.append(float(elbo.detach()))
        optimizer.zero_grad()  # Clear gradients for the next iteration
        print("batch done", _ / steps)
except Exception as e:
    print("Training failed")
    # raise e

time_points = list(range(len(elbo_list)))

# Create the plot
FPS = 260

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
    points = p.tolist()
    # (Draw your game elements here)
    for i in range(batch_size):
        pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(points[i][0], points[i][1], 5, 5))
    
    pygame.draw.rect(screen, (0, 0, 255), pygame.Rect(SCREEN_WIDTH/2, SCREEN_HEIGHT/2, 5, 5))

    pygame.display.flip()  # Update the full display Surface to the screen
        
    # Cap the frame rate
    clock.tick(FPS)

plt.plot(time_points, elbo_list)
# Add labels and title
plt.xlabel('Time (Iterations)')
plt.ylabel('Negative ELBO')
plt.title('Negative ELBO Over Time')
plt.show()

while True:
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
        
        points = p.tolist()
        # (Draw your game elements here)
        for i in range(batch_size):
            pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(points[i][0], points[i][1], 5, 5))
        
        pygame.draw.rect(screen, (0, 0, 255), pygame.Rect(SCREEN_WIDTH/2, SCREEN_HEIGHT/2, 5, 5))

        pygame.display.flip()  # Update the full display Surface to the screen
            
        # Cap the frame rate
        clock.tick(FPS)
# Clean up
pygame.quit()
sys.exit()