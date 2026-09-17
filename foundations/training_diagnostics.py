import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        activation_stats = []
        with torch.no_grad():
            for module in model.children():
                x = module(x)
                if isinstance(module, nn.Linear):
                    mean = round(float(x.mean()), 4)
                    std = round(float(x.std()), 4)
                    dead_fraction = round(((x <= 0).all(dim=0)).float().mean().item(), 4)
                    
                    activation_stats.append({'mean': mean, 'std': std, 'dead_fraction': dead_fraction})

        return activation_stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        y_hat = model(x)
        loss = nn.MSELoss()(y_hat, y)
        loss.backward()
        gradient_stats = []
        
        for module in model.children():
            if isinstance(module, nn.Linear):
                grad = module.weight.grad
                mean = round(grad.mean().item(), 4)
                std= round(grad.std().item(), 4)
                norm = round(torch.norm(grad).item(), 4)
                gradient_stats.append({'mean': mean, 'std': std, 'norm': norm})

        return gradient_stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)

        for layer in activation_stats:
            if layer['dead_fraction'] > 0.5:
                return 'dead_neurons'
        for layer in gradient_stats:
            if layer['norm'] > 1000:
                return 'exploding_gradients'

        if gradient_stats and gradient_stats[-1]['norm'] < 1e-5:
            return 'vanishing_gradients'

        for layer in activation_stats:
            if layer['std'] < 0.1:
                return 'vanishing_gradients'
            if layer['std'] > 10.0:
                return 'exploding_gradients'

        return 'healthy'
