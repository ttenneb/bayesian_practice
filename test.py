import torch

torch.manual_seed(0)
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
n = 10  # Number of samples

center_x = SCREEN_WIDTH / 2
center_y = SCREEN_HEIGHT / 2

px_a, px_v = 0.7600, 0.052
py_a, py_v = -1.4, 0.0005

# Sample directly using torch.normal()
pos_x = torch.normal(mean=center_x + (px_a * 100), std=(px_v ** 2) * 1000, size=(n,))
pos_y = torch.normal(mean=center_y + (py_a * 100), std=(py_v ** 2) * 1000, size=(n,))

print(pos_x, pos_y)
