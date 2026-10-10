import torch


def _sample_sigma(
    batch_size: int,
    timestep_range: tuple[float, float],
    distribution: str,
    device: torch.device,
) -> torch.Tensor:
    """Sample sigma values from a specified distribution within [low, high].

    Args:
        batch_size: number of samples
        timestep_range: (low, high) bounds
        distribution: one of
            "uniform"     — U[low, high]
            "center"      — Beta(2,2) scaled to [low, high], peaks at midpoint
                            (release default — endpoints carry little learning signal)

    Returns:
        sigma: (B,) tensor in [low, high]
    """
    low, high = timestep_range
    if distribution == "uniform":
        u = torch.rand(batch_size, device=device, dtype=torch.float32)
    elif distribution == "center":
        # Beta(2,2): symmetric, peaks at 0.5
        dist = torch.distributions.Beta(2.0, 2.0)
        u = dist.sample((batch_size,)).to(device=device, dtype=torch.float32)
    else:
        raise ValueError(
            f"Unknown timestep_distribution: '{distribution}'. "
            f"Choose from: uniform, center"
        )
    sigma = u * (high - low) + low
    return sigma.clamp(0.0, 1.0)


@torch.no_grad()
def make_noisy_latent(
    latent: torch.Tensor,
    timestep_range: tuple[float, float],
    distribution: str = "center",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add flow-matching noise to a normalized latent, on the latent's device.

    Noise schedule:  z_sigma = (1 - sigma) * latent + sigma * eps
    where sigma is sampled from the specified distribution within timestep_range.

    Args:
        latent: (B, C, H, W) normalized VAE latent
        timestep_range: (low, high) bounds for sigma sampling; pass (s, s)
            for an exact deterministic noise level
        distribution: "uniform" or "center" (release default)

    Returns:
        noisy_latent: same shape as latent, float32
        sigma: (B,) sampled noise levels
    """
    device = latent.device

    eps = torch.randn_like(latent, device=device, dtype=torch.float32)
    sigma = _sample_sigma(latent.shape[0], timestep_range, distribution, device)

    sigma_expanded = sigma.view(-1, *([1] * (latent.ndim - 1)))
    noisy_latent = (1 - sigma_expanded) * latent.float() + sigma_expanded * eps

    return noisy_latent, sigma
