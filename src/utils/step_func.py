import torch
import numpy as np

def get_alpha(alphas_cumprod, timestep):
    timestep_lt_zero_mask = torch.lt(timestep, 0).to(alphas_cumprod.dtype)
    normal_alpha = alphas_cumprod[torch.clip(timestep, 0)]
    one_alpha = torch.ones_like(normal_alpha).to(normal_alpha.dtype).to(normal_alpha.dtype) 
    return normal_alpha * (1 - timestep_lt_zero_mask) + one_alpha * timestep_lt_zero_mask

def psuedo_velocity_wrt_noisy_and_timestep(noisy_images, noisy_images_pre, alphas_cumprod, timestep, timestep_prev):
    alpha_prod_t = get_alpha(alphas_cumprod, timestep).view(-1, 1, 1, 1, 1).detach()
    beta_prod_t = 1 - alpha_prod_t
    alpha_prod_t_prev = get_alpha(alphas_cumprod, timestep_prev).view(-1, 1, 1, 1, 1).detach()
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    a_s = (alpha_prod_t_prev ** (0.5)).to(noisy_images.dtype)
    a_t = (alpha_prod_t ** (0.5)).to(noisy_images.dtype)
    b_s = (beta_prod_t_prev ** (0.5)).to(noisy_images.dtype)
    b_t = (beta_prod_t ** (0.5)).to(noisy_images.dtype)

    psuedo_velocity = (noisy_images_pre - (
        a_s * a_t + b_s * b_t
    ) * noisy_images) / (
        b_s * a_t -  a_s * b_t
    )

    return psuedo_velocity

def origin_by_velocity_and_sample(velocity, noisy_images, alphas_cumprod, timestep):
    alpha_prod_t = get_alpha(alphas_cumprod, timestep).view(-1, 1, 1, 1, 1).detach()
    beta_prod_t = 1 - alpha_prod_t
    a_t = (alpha_prod_t ** (0.5)).to(noisy_images.dtype)
    b_t = (beta_prod_t ** (0.5)).to(noisy_images.dtype)

    pred_original_sample = a_t * noisy_images - b_t * velocity
    return pred_original_sample
    