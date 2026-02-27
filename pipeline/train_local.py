# Module: train_local
# License: MIT (ARVTON project)
# Description: Phase 8 — Local/Colab training orchestrator for the GAN refinement model.
# Platform: Colab T4/A100 + AMD ROCm + Local
# Dependencies: torch, tqdm

"""
===================================
PHASE 8 — TRAINING ORCHESTRATOR
File: train_local.py
===================================

Training Orchestrator
======================
Trains the RefineGenerator (from Phase 3) end-to-end using:
    - TryOnDataset from manifests
    - U-Net Generator + MultiScale Discriminator
    - Combined loss: L1 + VGG Perceptual + Adversarial
    - LoRA (optional) + Mixed precision + Gradient checkpointing

Usage:
    python -m pipeline.train_local --epochs 50 --batch-size 4

For Colab with T4:
    python -m pipeline.train_local --epochs 30 --batch-size 2 --amp
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from pipeline.fine_tune import TryOnDataset, apply_lora
from pipeline.platform_utils import (
    detect_gpu_info,
    detect_platform,
    get_paths,
    get_precision_dtype,
    load_config,
    mount_google_drive,
    print_system_info,
    setup_logging,
)
from pipeline.refine import (
    MultiScaleDiscriminator,
    RefineGenerator,
    TryOnCombinedLoss,
)

logger = logging.getLogger("arvton.train_local")


def train_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 50,
    lr_g: float = 2e-4,
    lr_d: float = 2e-4,
    device: str = "cuda",
    checkpoint_dir: str = "checkpoints",
    save_every: int = 10,
    use_amp: bool = True,
) -> Dict[str, List[float]]:
    """
    Full GAN training loop for the refinement model.

    Args:
        generator: RefineGenerator model.
        discriminator: MultiScaleDiscriminator model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader (optional).
        epochs: Training epochs.
        lr_g: Generator learning rate.
        lr_d: Discriminator learning rate.
        device: Torch device.
        checkpoint_dir: Checkpoint save directory.
        save_every: Save frequency (epochs).
        use_amp: Use automatic mixed precision.

    Returns:
        Training history dict.
    """
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # Optimizers
    opt_g = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, generator.parameters()),
        lr=lr_g, betas=(0.5, 0.999), weight_decay=0.01,
    )
    opt_d = torch.optim.AdamW(
        discriminator.parameters(),
        lr=lr_d, betas=(0.5, 0.999), weight_decay=0.01,
    )

    # Schedulers
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=epochs)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=epochs)

    # Losses
    criterion = TryOnCombinedLoss(device=device)
    adv_criterion = nn.BCEWithLogitsLoss()

    dtype = get_precision_dtype()
    scaler_g = torch.amp.GradScaler("cuda") if use_amp else None
    scaler_d = torch.amp.GradScaler("cuda") if use_amp else None

    history = {
        "g_loss": [], "d_loss": [],
        "l1_loss": [], "perceptual_loss": [],
        "val_loss": [],
    }

    logger.info("Starting GAN training: %d epochs, batch=%d, AMP=%s",
                epochs, train_loader.batch_size, use_amp)

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            person = batch["person"].to(device)
            garment = batch["garment"].to(device)
            gt = batch["gt"].to(device)

            # ── Discriminator step ──
            opt_d.zero_grad()
            gen_input = torch.cat([person, garment], dim=1)

            if use_amp:
                with torch.amp.autocast("cuda", dtype=dtype):
                    fake = generator(gen_input).detach()
                    # Real
                    d_real = discriminator(torch.cat([gt, person], dim=1))
                    # Fake
                    d_fake = discriminator(torch.cat([fake, person], dim=1))
                    d_loss = 0.0
                    for dr, df in zip(d_real, d_fake):
                        d_loss += (
                            adv_criterion(dr, torch.ones_like(dr)) +
                            adv_criterion(df, torch.zeros_like(df))
                        ) * 0.5
                scaler_d.scale(d_loss).backward()
                scaler_d.step(opt_d)
                scaler_d.update()
            else:
                fake = generator(gen_input).detach()
                d_real = discriminator(torch.cat([gt, person], dim=1))
                d_fake = discriminator(torch.cat([fake, person], dim=1))
                d_loss = sum(
                    (adv_criterion(dr, torch.ones_like(dr)) +
                     adv_criterion(df, torch.zeros_like(df))) * 0.5
                    for dr, df in zip(d_real, d_fake)
                )
                d_loss.backward()
                opt_d.step()

            # ── Generator step ──
            opt_g.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda", dtype=dtype):
                    fake = generator(gen_input)
                    d_fake = discriminator(torch.cat([fake, person], dim=1))
                    losses = criterion(fake, gt, d_fake)
                scaler_g.scale(losses["total"]).backward()
                scaler_g.step(opt_g)
                scaler_g.update()
            else:
                fake = generator(gen_input)
                d_fake = discriminator(torch.cat([fake, person], dim=1))
                losses = criterion(fake, gt, d_fake)
                losses["total"].backward()
                opt_g.step()

            epoch_g_loss += losses["total"].item()
            epoch_d_loss += d_loss.item() if isinstance(d_loss, torch.Tensor) else d_loss
            num_batches += 1

            pbar.set_postfix({
                "G": f"{losses['total'].item():.4f}",
                "D": f"{d_loss.item() if isinstance(d_loss, torch.Tensor) else d_loss:.4f}",
            })

        sched_g.step()
        sched_d.step()

        avg_g = epoch_g_loss / max(num_batches, 1)
        avg_d = epoch_d_loss / max(num_batches, 1)
        history["g_loss"].append(avg_g)
        history["d_loss"].append(avg_d)

        # Validation
        if val_loader is not None:
            generator.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    person = batch["person"].to(device)
                    garment = batch["garment"].to(device)
                    gt = batch["gt"].to(device)
                    gen_input = torch.cat([person, garment], dim=1)

                    if use_amp:
                        with torch.amp.autocast("cuda", dtype=dtype):
                            fake = generator(gen_input)
                            losses = criterion(fake, gt)
                    else:
                        fake = generator(gen_input)
                        losses = criterion(fake, gt)

                    val_loss += losses["total"].item()
                    val_batches += 1

            avg_val = val_loss / max(val_batches, 1)
            history["val_loss"].append(avg_val)
            logger.info("Epoch %d — G: %.4f, D: %.4f, Val: %.4f", epoch, avg_g, avg_d, avg_val)
        else:
            logger.info("Epoch %d — G: %.4f, D: %.4f", epoch, avg_g, avg_d)

        # Save checkpoint
        if epoch % save_every == 0 or epoch == epochs:
            torch.save({
                "epoch": epoch,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "history": history,
            }, str(ckpt_path / f"gan_epoch_{epoch:03d}.pt"))
            logger.info("Checkpoint saved: epoch %d", epoch)

        # VRAM monitoring
        if torch.cuda.is_available() and epoch % 5 == 0:
            vram = torch.cuda.memory_allocated() / (1024 ** 3)
            vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            logger.info("VRAM: %.2f GB (peak: %.2f GB)", vram, vram_peak)

    logger.info("GAN training complete: %d epochs", epochs)
    return history


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="ARVTON — Training Orchestrator")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--platform", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision")
    parser.add_argument("--lora", action="store_true", help="Use LoRA adapters")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # Setup
    setup_logging("INFO")
    logger.info("=" * 60)
    logger.info("ARVTON — GAN Training Orchestrator")
    logger.info("=" * 60)

    platform = args.platform or detect_platform()
    paths = get_paths(platform)

    if platform == "colab":
        mount_google_drive()

    print_system_info()

    gpu_info = detect_gpu_info()
    device = args.device
    if not gpu_info["available"] and device == "cuda":
        device = "cpu"
        logger.warning("No GPU available. Training on CPU (very slow).")

    # Dataset
    train_manifest = str(paths["train_manifest"])
    val_manifest = str(paths["val_manifest"])

    if not Path(train_manifest).exists():
        logger.error("Train manifest not found: %s", train_manifest)
        logger.error("Run Phase 0 first to generate the dataset.")
        return

    train_ds = TryOnDataset(train_manifest, augment=True)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=True,
    )

    val_loader = None
    if Path(val_manifest).exists():
        val_ds = TryOnDataset(val_manifest, augment=False)
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size,
            shuffle=False, num_workers=2, pin_memory=True,
        )

    # Models
    generator = RefineGenerator(in_channels=6, out_channels=3)
    discriminator = MultiScaleDiscriminator(in_channels=6)

    if args.lora:
        generator = apply_lora(generator, rank=args.lora_rank)

    # Resume from checkpoint
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt["generator"])
        discriminator.load_state_dict(ckpt["discriminator"])
        logger.info("Resumed from checkpoint: %s (epoch %d)", args.resume, ckpt["epoch"])

    g_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    d_params = sum(p.numel() for p in discriminator.parameters())
    logger.info("Generator params: %s, Discriminator params: %s",
                f"{g_params:,}", f"{d_params:,}")

    # Train
    history = train_gan(
        generator, discriminator,
        train_loader, val_loader,
        epochs=args.epochs,
        lr_g=args.lr, lr_d=args.lr,
        device=device,
        checkpoint_dir=str(paths["checkpoints"] / "gan"),
        use_amp=args.amp,
    )

    # Summary
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"  Epochs: {args.epochs}")
    print(f"  Final G loss: {history['g_loss'][-1]:.4f}")
    print(f"  Final D loss: {history['d_loss'][-1]:.4f}")
    if history["val_loss"]:
        print(f"  Final Val loss: {history['val_loss'][-1]:.4f}")
    print(f"  Checkpoints: {paths['checkpoints'] / 'gan'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
