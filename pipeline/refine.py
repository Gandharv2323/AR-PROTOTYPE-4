# Module: refine
# License: MIT (ARVTON project)
# Description: Phase 3 — GAN-based refinement with perceptual loss and discriminator.
# Platform: Colab T4/A100 + AMD ROCm
# Dependencies: torch, torchvision, Pillow, numpy, opencv-python-headless

"""
===================================
PHASE 3 — GAN REFINEMENT
File: refine.py
===================================

Refinement Module
==================
Applies a lightweight U-Net GAN with multi-scale discriminator and
VGG perceptual loss to sharpen try-on outputs from Phase 2.

Architecture:
    - Generator: U-Net with skip connections (encoder-decoder)
    - Discriminator: Multi-scale PatchGAN (70×70 receptive field)
    - Loss: L1 + VGG-perceptual + adversarial

Training and inference functions included.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger("arvton.refine")


# ═══════════════════════════════════════════════════════════════════════
# GENERATOR — U-Net with skip connections
# ═══════════════════════════════════════════════════════════════════════


class UNetBlock(nn.Module):
    """U-Net encoder/decoder block with optional normalization and dropout."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        down: bool = True,
        use_bn: bool = True,
        use_dropout: bool = False,
    ):
        super().__init__()
        if down:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            layers = [
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
            ]
            if use_dropout:
                layers.append(nn.Dropout(0.5))
            layers.append(nn.ReLU(inplace=True))
            self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class RefineGenerator(nn.Module):
    """
    U-Net generator for try-on output refinement.

    Input: 6-channel tensor (3ch try-on + 3ch person original)
    Output: 3-channel refined image

    Architecture follows pix2pix-style U-Net:
        Encoder: 64 → 128 → 256 → 512 → 512 → 512 → 512 → 512
        Decoder: mirrors encoder with skip connections
    """

    def __init__(self, in_channels: int = 6, out_channels: int = 3):
        super().__init__()

        # Encoder layers
        self.enc1 = UNetBlock(in_channels, 64, down=True, use_bn=False)
        self.enc2 = UNetBlock(64, 128, down=True)
        self.enc3 = UNetBlock(128, 256, down=True)
        self.enc4 = UNetBlock(256, 512, down=True)
        self.enc5 = UNetBlock(512, 512, down=True)
        self.enc6 = UNetBlock(512, 512, down=True)
        self.enc7 = UNetBlock(512, 512, down=True)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder layers (with skip connections, so in_ch = out_ch + skip_ch)
        self.dec1 = UNetBlock(512, 512, down=False, use_dropout=True)
        self.dec2 = UNetBlock(1024, 512, down=False, use_dropout=True)
        self.dec3 = UNetBlock(1024, 512, down=False, use_dropout=True)
        self.dec4 = UNetBlock(1024, 512, down=False)
        self.dec5 = UNetBlock(1024, 256, down=False)
        self.dec6 = UNetBlock(512, 128, down=False)
        self.dec7 = UNetBlock(256, 64, down=False)

        # Final output layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)

        # Bottleneck
        b = self.bottleneck(e7)

        # Decoder with skip connections
        d1 = self.dec1(b)
        d2 = self.dec2(torch.cat([d1, e7], dim=1))
        d3 = self.dec3(torch.cat([d2, e6], dim=1))
        d4 = self.dec4(torch.cat([d3, e5], dim=1))
        d5 = self.dec5(torch.cat([d4, e4], dim=1))
        d6 = self.dec6(torch.cat([d5, e3], dim=1))
        d7 = self.dec7(torch.cat([d6, e2], dim=1))

        return self.final(torch.cat([d7, e1], dim=1))


# ═══════════════════════════════════════════════════════════════════════
# DISCRIMINATOR — Multi-scale PatchGAN
# ═══════════════════════════════════════════════════════════════════════


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator (70×70 receptive field).
    Input: 6-channel (3ch image + 3ch condition)
    Output: Real/fake patch map.
    """

    def __init__(self, in_channels: int = 6):
        super().__init__()

        self.model = nn.Sequential(
            # Layer 1: no norm
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 3
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 4
            nn.Conv2d(256, 512, 4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # Output
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator: applies PatchGAN at 2 scales
    for better texture fidelity at different resolutions.
    """

    def __init__(self, in_channels: int = 6, num_scales: int = 2):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PatchDiscriminator(in_channels) for _ in range(num_scales)
        ])
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        current = x
        for disc in self.discriminators:
            outputs.append(disc(current))
            current = self.downsample(current)
        return outputs


# ═══════════════════════════════════════════════════════════════════════
# PERCEPTUAL LOSS — VGG-based
# ═══════════════════════════════════════════════════════════════════════


class VGGPerceptualLoss(nn.Module):
    """
    VGG-19 perceptual loss.
    Extracts features at layers: relu1_2, relu2_2, relu3_4, relu4_4.

    Uses torchvision's VGG19 weights (weight file under BSD-like license,
    model code under BSD-3-Clause).
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()
        import torchvision.models as models

        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slices = nn.ModuleList([
            nn.Sequential(*list(vgg.children())[:4]),    # relu1_2
            nn.Sequential(*list(vgg.children())[4:9]),   # relu2_2
            nn.Sequential(*list(vgg.children())[9:18]),  # relu3_4
            nn.Sequential(*list(vgg.children())[18:27]), # relu4_4
        ])

        # Freeze VGG weights
        for param in self.parameters():
            param.requires_grad = False

        self.to(device)
        self.eval()

        # ImageNet normalization
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize from [-1, 1] to ImageNet range."""
        x = (x + 1) / 2  # [-1,1] → [0,1]
        return (x - self.mean) / self.std

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pred_norm = self.normalize(pred)
        target_norm = self.normalize(target)

        loss = torch.tensor(0.0, device=pred.device)
        x_pred = pred_norm
        x_target = target_norm

        for slice_module in self.slices:
            x_pred = slice_module(x_pred)
            x_target = slice_module(x_target)
            loss = loss + F.l1_loss(x_pred, x_target)

        return loss


# ═══════════════════════════════════════════════════════════════════════
# COMBINED LOSS
# ═══════════════════════════════════════════════════════════════════════


class TryOnCombinedLoss(nn.Module):
    """
    Combined loss for try-on refinement:
        L = λ_l1 * L1 + λ_perc * L_perceptual + λ_adv * L_adversarial

    Default weights: L1=10.0, Perceptual=10.0, Adversarial=1.0
    """

    def __init__(
        self,
        device: str = "cuda",
        lambda_l1: float = 10.0,
        lambda_perceptual: float = 10.0,
        lambda_adversarial: float = 1.0,
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_adversarial = lambda_adversarial

        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss(device)
        self.adversarial_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        disc_pred: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        losses = {}

        # L1 loss
        losses["l1"] = self.l1_loss(pred, target) * self.lambda_l1

        # Perceptual loss
        losses["perceptual"] = self.perceptual_loss(pred, target) * self.lambda_perceptual

        # Adversarial loss (generator wants discriminator to output 1)
        if disc_pred is not None:
            adv_loss = torch.tensor(0.0, device=pred.device)
            for dp in disc_pred:
                target_real = torch.ones_like(dp)
                adv_loss = adv_loss + self.adversarial_loss(dp, target_real)
            losses["adversarial"] = adv_loss * self.lambda_adversarial
        else:
            losses["adversarial"] = torch.tensor(0.0, device=pred.device)

        losses["total"] = sum(losses.values())
        return losses


# ═══════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════

_refine_model = None


def load_refinement_model(
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
) -> RefineGenerator:
    """
    Load the refinement generator (trained or untrained).

    Args:
        checkpoint_path: Path to trained checkpoint. None = fresh model.
        device: Torch device.

    Returns:
        RefineGenerator model.
    """
    global _refine_model
    if _refine_model is not None:
        return _refine_model

    model = RefineGenerator(in_channels=6, out_channels=3)

    if checkpoint_path and Path(checkpoint_path).exists():
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        logger.info("Loaded refinement checkpoint from %s", checkpoint_path)
    else:
        logger.warning("No refinement checkpoint found. Using untrained model.")

    model = model.to(device)
    model.eval()
    _refine_model = model
    return model


def refine(
    tryon_image: Image.Image,
    person_image: Image.Image,
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
) -> Image.Image:
    """
    Refine a try-on result using the GAN refinement model.

    Args:
        tryon_image: Try-on result from Phase 2.
        person_image: Original person image.
        checkpoint_path: Path to trained generator checkpoint.
        device: Torch device.

    Returns:
        Refined try-on image (768×1024).
    """
    model = load_refinement_model(checkpoint_path, device)

    # Preprocess: resize to 768×1024, normalize to [-1, 1]
    target_size = (768, 1024)
    tryon_resized = tryon_image.resize(target_size, Image.LANCZOS)
    person_resized = person_image.resize(target_size, Image.LANCZOS)

    tryon_tensor = torch.from_numpy(
        np.array(tryon_resized).transpose(2, 0, 1).astype(np.float32) / 127.5 - 1.0
    ).unsqueeze(0).to(device)

    person_tensor = torch.from_numpy(
        np.array(person_resized).transpose(2, 0, 1).astype(np.float32) / 127.5 - 1.0
    ).unsqueeze(0).to(device)

    # Concatenate: 6-channel input
    input_tensor = torch.cat([tryon_tensor, person_tensor], dim=1)

    with torch.no_grad():
        output = model(input_tensor)

    # Postprocess: [-1, 1] → [0, 255]
    output_np = ((output[0].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).astype(np.uint8)
    result = Image.fromarray(output_np)

    logger.info("Refinement complete.")
    return result


def clear_cache():
    """Clear cached refinement model."""
    global _refine_model
    _refine_model = None
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    print("Phase 3 — GAN Refinement Module")
    print("Architecture:")
    g = RefineGenerator()
    d = MultiScaleDiscriminator()
    g_params = sum(p.numel() for p in g.parameters())
    d_params = sum(p.numel() for p in d.parameters())
    print(f"  Generator params: {g_params:,}")
    print(f"  Discriminator params: {d_params:,}")
    print(f"\nPhase 3 complete. Files written: [refine.py]")
    print("Reply 'continue' to proceed to Phase 4.")
