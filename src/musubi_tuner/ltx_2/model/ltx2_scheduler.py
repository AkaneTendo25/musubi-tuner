# LTX-2 specific scheduler and diffusion step implementations
# Ported from ltx_core.components to work without external dependencies

import torch


class LTX2Scheduler:
    """
    LTX-2 sigma schedule generator.
    
    Generates a sequence of sigma values for the diffusion sampling process.
    Based on ltx_core.components.schedulers.LTX2Scheduler
    """
    
    def __init__(self, shift: float = 1.0):
        """
        Initialize the scheduler.
        
        Args:
            shift: Shift parameter for sigma schedule (1.0 = no shift)
        """
        self.shift = shift
    
    def execute(self, steps: int) -> torch.Tensor:
        """
        Generate sigma schedule for the given number of steps.
        
        Args:
            steps: Number of inference steps
            
        Returns:
            Tensor of shape (steps + 1,) with sigma values from 1.0 to 0.0
        """
        # Linear schedule from 1 to 0
        sigmas = torch.linspace(1.0, 0.0, steps + 1)
        
        # Apply shift if not 1.0 (SD3-style time shift)
        if self.shift != 1.0:
            sigmas = self._apply_shift(sigmas)
        
        return sigmas
    
    def _apply_shift(self, sigmas: torch.Tensor) -> torch.Tensor:
        """Apply shift transformation to sigmas."""
        # SD3-style shift: sigma_shifted = (shift * sigma) / (1 + (shift - 1) * sigma)
        return (self.shift * sigmas) / (1 + (self.shift - 1) * sigmas)


class EulerDiffusionStep:
    """
    Euler diffusion step implementation.
    
    Implements the Euler method for stepping through the diffusion process.
    Based on ltx_core.components.diffusion_steps.EulerDiffusionStep
    """
    
    def step(
        self,
        sample: torch.Tensor,
        denoised_sample: torch.Tensor,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        """
        Perform one Euler step in the diffusion process.
        
        Args:
            sample: Current noisy sample (x_t)
            denoised_sample: Predicted denoised sample (x_0)
            sigmas: Full sigma schedule
            step_index: Current step index
            
        Returns:
            Updated sample for the next step (x_{t-1})
        """
        sigma_current = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]
        
        # Compute dt (negative because we go from high sigma to low)
        dt = sigma_next - sigma_current
        
        # Euler step: x_{t-1} = x_t + (x_0 - x_t) * (sigma_next - sigma_current) / sigma_current
        # Simplified: x_{t-1} = x_t + d_x * dt where d_x = (x_0 - x_t) / sigma_current
        
        # The direction from x to x_0
        if sigma_current > 0:
            d = (sample - denoised_sample) / sigma_current
        else:
            d = torch.zeros_like(sample)
        
        # Euler step
        prev_sample = sample + d * dt
        
        return prev_sample


class X0PredictionWrapper:
    """
    Wrapper to convert velocity predictions to x0 (denoised) predictions.
    
    LTX-2 model outputs velocity = noise - latents.
    This wrapper converts velocity to x0 prediction.
    """
    
    @staticmethod
    def velocity_to_x0(
        noisy_sample: torch.Tensor,
        velocity: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        """
        Convert velocity prediction to x0 prediction.
        
        The flow matching formulation:
        - noisy = (1 - sigma) * x0 + sigma * noise
        - velocity = noise - x0
        
        Therefore:
        - x0 = noisy - sigma * velocity
        
        Args:
            noisy_sample: The current noisy sample
            velocity: The model's velocity prediction
            sigma: Current sigma value (0 to 1)
            
        Returns:
            Predicted x0 (denoised sample)
        """
        return noisy_sample - sigma * velocity
