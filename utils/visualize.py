import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_psf_grid(physical_layer, H, W, device, filename='psf_grid.png'):
    H_pad = H
    W_pad = W
    (rows, cols) = (5, 5)
    y = torch.linspace(-0.9, 0.9, rows, device=device)
    x = torch.linspace(-0.9, 0.9, cols, device=device)
    (yy, xx) = torch.meshgrid(y, x, indexing='ij')
    coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
    with torch.no_grad():
        coeffs = physical_layer.aberration_net(coords)
        kernels = physical_layer.zernike_generator(coeffs)
    kernels = kernels.cpu().squeeze(1).numpy()
    (fig, axes) = plt.subplots(rows, cols, figsize=(10, 10))
    for i in range(rows * cols):
        (r, c) = (i // cols, i % cols)
        ax = axes[r, c]
        k = kernels[i]
        im = ax.imshow(np.log1p(k * 100), cmap='inferno')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_coefficient_maps(physical_layer, H, W, device, filename='coeff_maps.png'):
    grid_size = 128
    y = torch.linspace(-1, 1, grid_size, device=device)
    x = torch.linspace(-1, 1, grid_size, device=device)
    (yy, xx) = torch.meshgrid(y, x, indexing='ij')
    coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
    with torch.no_grad():
        coeffs = physical_layer.aberration_net(coords)
    coeffs = coeffs.reshape(grid_size, grid_size, -1).cpu().numpy()
    (fig, axes) = plt.subplots(2, 2, figsize=(10, 8))
    labels = ['Defocus (Noll 4)', 'Astig 1 (Noll 5)', 'Astig 2 (Noll 6)', 'Coma (Noll 7)']
    indices = [3, 4, 5, 6]
    for (i, (idx, label)) in enumerate(zip(indices, labels)):
        if idx < coeffs.shape[-1]:
            ax = axes.flatten()[i]
            im = ax.imshow(coeffs[..., idx], cmap='viridis')
            ax.set_title(label)
            plt.colorbar(im, ax=ax)
            ax.axis('off')
    plt.savefig(filename)
    plt.close()