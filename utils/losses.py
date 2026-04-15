import torch
import torch.nn.functional as F

# basic masked helpers functions
# compute masked mean of a tensor, returns mean of 'value' over the masked region
def safe_mean(value, mask, eps=1e-8):
    return (value * mask).sum() / (mask.sum() + eps)

# split binary mask into gap mask and contex mask
def split_gap_context(mask):
    """
    Assumes: mask == 1 in missing and mask == 0 in known
    """
    gap_mask = mask
    context_mask = 1.0 - mask
    return gap_mask, context_mask

##############################################################################################
# Losses for CNN and U-Net
##############################################################################################

# basic reconstruction loss

# defaul reconstruction loss
def masked_l1_loss(pred, target, mask, context_weight=0.1, eps=1e-8):
    # split mask into missing region and known region
    gap_mask, context_mask = split_gap_context(mask)
    abs_error = (pred - target).abs() # per pixel abs error

    # avg error in missing region
    gap_loss = safe_mean(abs_error, gap_mask, eps)

    # avg error in visible region
    context_loss = safe_mean(abs_error, context_mask, eps)

    total_loss = gap_loss + context_weight * context_loss
    return total_loss, gap_loss, context_loss

# balance between L1 for larger errors and L2 for small errors
def masked_huber_loss(pred, target, mask, context_weight=0.1, delta=1.0, eps=1e-8):
    # split mask
    gap_mask, context_mask = split_gap_context(mask)

    # raw pred error
    error = pred - target
    abs_error = error.abs()

    # element wise huber loss
    huber = torch.where(
        abs_error < delta,
        0.5 * error ** 2,
        delta * (abs_error - 0.5 * delta)
    )
    # avg error in missing region
    gap_loss = safe_mean(huber, gap_mask, eps)

    # avg error in visible region
    context_loss = safe_mean(huber, context_mask, eps)

    total_loss = gap_loss + context_weight * context_loss
    return total_loss, gap_loss, context_loss

# gradient-aware loss, preserves harmonics, onset boundaries and reduc overly smooth reconstructions
def gradient_consistency_loss(pred, target, reduction="mean"):
    """
    Method:
    - compute horizontal differences (time-direction changes)
    - compute vertical differences (frequency-direction changes)
    - match those gradients using L1 loss
    """

    # horizontal gradients; difference across time axis
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]

    # vertical gradientsl difference across freq axis
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

    # match gradients in both directions
    loss_x = F.l1_loss(pred_dx, target_dx, reduction=reduction)
    loss_y = F.l1_loss(pred_dy, target_dy, reduction=reduction)

    return loss_x + loss_y

# masked L1 + gradient consistency loss
def masked_l1_grad_loss(pred, target, mask, context_weight=0.1, grad_weight=0.1, eps=1e-8):

    # base reconstruction term
    recon_loss, gap_loss, context_loss = masked_l1_loss(
        pred, target, mask, context_weight=context_weight, eps=eps
    )

    # structure-preserving term
    grad_loss = gradient_consistency_loss(pred, target)

    total_loss = recon_loss + grad_weight * grad_loss

    return total_loss, gap_loss, context_loss, grad_loss

# masked huber + gradient consistency loss
def masked_huber_grad_loss(pred, target, mask, context_weight=0.1, grad_weight=0.1, delta=1.0, eps=1e-8):
    # base reconstruction term
    recon_loss, gap_loss, context_loss = masked_huber_loss(
        pred, target, mask, context_weight=context_weight, delta=delta, eps=eps
    )
    # add structure consistency
    grad_loss = gradient_consistency_loss(pred, target)
    total_loss = recon_loss + grad_weight * grad_loss

    return total_loss, gap_loss, context_loss, grad_loss

##############################################################################################
# Losses for larger gaps 3.0 and 5.0 still for U-NEt and CNN
##############################################################################################

# multi-resoltuion masked l1 loss for larger gaps 3.0 and 5.0
def masked_multires_l1_loss(pred, target, mask, context_weight=0.1, scales=(1, 2, 4), scale_weights=None, eps=1e-8):
    """
    Compare prediction and target at multiple spatial resolutions.
    - helps for larger gaps such as 3.0s and 5.0s
    - encourages both local detail and broader structure matching
    - useful when long gaps require more global consistency

    scales:
    - 1 = original resolution
    - 2 = pooled by factor 2
    - 4 = pooled by factor 4

    """
    # if no explicit weights are provided, weight all scales equally
    if scale_weights is None:
        scale_weights = [1.0 / len(scales)] * len(scales)

    # accumulated weighted losses across resolutions
    total = 0.0
    gap_total = 0.0
    context_total = 0.0

    for s, w in zip(scales, scale_weights):
        if s == 1:
            # original resolution
            pred_s = pred
            target_s = target
            mask_s = mask
        else:
            # downsampling predictions, target, and mask
            pred_s = F.avg_pool2d(pred, kernel_size=s, stride=s)
            target_s = F.avg_pool2d(target, kernel_size=s, stride=s)
            mask_s = F.avg_pool2d(mask, kernel_size=s, stride=s)

            # convert pooled mask back to binary
            mask_s = (mask_s > 0).float()

        # compute masked l1
        loss_s, gap_s, context_s = masked_l1_loss(
            pred_s, target_s, mask_s, context_weight=context_weight, eps=eps
        )

        # weighted accumulation
        total = total + w * loss_s
        gap_total = gap_total + w * gap_s
        context_total = context_total + w * context_s

    return total, gap_total, context_total

# multi-resoltuion masked l1 + gradient consistency loss for larger gaps 3.0 and 5.0
def masked_multires_l1_grad_loss(pred, target, mask, context_weight=0.1, grad_weight=0.1, scales=(1, 2, 4),
                                  scale_weights=None, eps=1e-8):
    """
    - stronger than plain masked_multires_l1_loss
    - captures both large-scale structure and local sharpness
    - useful for stronger CNN baselines or U-Nets on larger gaps
    """

    # multiscale reconstruction term
    recon_loss, gap_loss, context_loss = masked_multires_l1_loss(
        pred,
        target,
        mask,
        context_weight=context_weight,
        scales=scales,
        scale_weights=scale_weights,
        eps=eps,
    )
    # add structure preseving gradient term
    grad_loss = gradient_consistency_loss(pred, target)
    total_loss = recon_loss + grad_weight * grad_loss

    return total_loss, gap_loss, context_loss, grad_loss


##############################################################################################
# Losses for Diffusion Model
##############################################################################################

# DDPm diffusion training loss
def diffusion_noise_mse_loss(pred_noise, true_noise):
    loss = F.mse_loss(pred_noise, true_noise)
    return loss

# diffusion noise prediction loss with l1
def diffusion_noise_l1_loss(pred_noise, true_noise):
    loss = F.l1_loss(pred_noise, true_noise)
    return loss

# diffusion noise prediciton using Huber for whenn target produces larger errors
def diffusion_noise_huber_loss(pred_noise, true_noise, delta=1.0):
    error = pred_noise - true_noise
    abs_error = error.abs()

    # element wise huber penalty
    huber = torch.where(
        abs_error < delta,
        0.5 * error ** 2,
        delta * (abs_error - 0.5 * delta)
    )
    return huber.mean()

# latent reconstruction loss for diffusion

# latent space l1 loss
def latent_l1_loss(pred_latent, target_latent):
    return F.l1_loss(pred_latent, target_latent)

# latent space l2 loss
def latent_l2_loss(pred_latent, target_latent):
    return F.mse_loss(pred_latent, target_latent)

##############################################################################################
# Masked losses for Diffusion Model
##############################################################################################

# masked weighted MSE noise loss
def masked_diffusion_noise_mse_loss(pred_noise, true_noise, mask_latent, masked_weight=3.0, eps=1e-8):
    """
    diffusion noise mse treated every latent position equally, but for inpainting the masked region 
    is harder portion of learning, so we upweight the masked region

    mask_latent: 1 in missing region, 0 in known region

    masked_weight: weight to care more about missing region than known region
    """
    se = (pred_noise - true_noise) ** 2

    # known region weight = 1
    # missing region weight = masked_weight
    weight = 1.0 + (masked_weight - 1.0) * mask_latent

    # multiply squared error by weights, then average
    loss = (se * weight).sum() / (weight.sum() * pred_noise.shape[1] + eps)
    return loss

# return on mse value per batch item
def per_sample_mse_loss(pred, target):
    return ((pred - target) ** 2).mean(dim=(1, 2, 3))

# return one mask-weight MSE value per batc item
def per_sample_masked_mse_loss(pred, target, mask_latent, masked_weight=3.0, eps=1e-8):
    se = (pred - target) ** 2
    weight = 1.0 + (masked_weight - 1.0) * mask_latent

    numer = (se * weight).sum(dim=(1, 2, 3))
    denom = (weight.sum(dim=(1, 2, 3)) * pred.shape[1]) + eps
    return numer / denom

# compute signal-to-noise ratio for each timestep
def compute_snr(schedule, t, eps=1e-8):
  alpha_bar_t = schedule.alpha_bars.gather(0, t)
  snr = alpha_bar_t / (1.0 - alpha_bar_t + eps)
  return snr

# min-snr weighting
def min_snr(schedule, t, gamma=5.0, eps=1e-8):
  """
  high snr timesteps can dominate training, hence we clamping them using gamma 
  weight = min(snr, gamma)/snr
  """
  snr = compute_snr(schedule, t, eps=eps)
  clipped = torch.clamp(snr, max=gamma)
  weight = clipped / (snr + eps)
  return weight  # [B]

##############################################################################################
# Metric helpers for evaluation
##############################################################################################

# mean absolute error inside missing gap
def masked_mae(pred, target, mask, eps=1e-8):
    gap_mask, _ = split_gap_context(mask)
    abs_error = (pred - target).abs()
    return safe_mean(abs_error, gap_mask, eps)

# rmse inside missing gap
def masked_rmse(pred, target, mask, eps=1e-8):
    gap_mask, _ = split_gap_context(mask)
    sq_error = (pred - target) ** 2
    mse = safe_mean(sq_error, gap_mask, eps)
    return torch.sqrt(mse + eps)

# mae over full spectrogram
def full_mae(pred, target):
    return F.l1_loss(pred, target)

# rmse over full spectrogram
def full_rmse(pred, target, eps=1e-8):
    mse = F.mse_loss(pred, target)
    return torch.sqrt(mse + eps)

# peak signal to noise ratio
def psnr(pred, target, data_range=1.0, eps=1e-8):
    """
    Higher is better
    """
    mse = F.mse_loss(pred, target)
    return 20.0 * torch.log10(torch.tensor(data_range, device=pred.device)) - 10.0 * torch.log10(mse + eps)