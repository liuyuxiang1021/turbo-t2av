"""
Evaluate checkpoint diversity: 8 prompts × 5 seeds, 4-step inference.
Lightweight — no Trainer, no real_score, no fake_score, no optimizer.

Usage:
  torchrun --nnodes=1 --nproc_per_node=8 --master_addr=localhost --master_port=PORT \
    -m ltx_distillation.eval_diversity \
    --config_path CONFIG_YAML \
    --checkpoint_path CKPT_MODEL \
    --output_dir OUT_DIR
"""

import os
import sys
import math
import argparse
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from ltx_distillation.util import launch_distributed_job, fsdp_wrap, set_seed
from ltx_distillation.models.ltx_trig_wrapper import create_ltx2_trig_wrapper
from ltx_distillation.models.text_encoder_wrapper import create_text_encoder_wrapper
from ltx_distillation.models.vae_wrapper import create_vae_wrappers
from ltx_distillation.inference.bidirectional_pipeline import BidirectionalAVInferencePipeline
from ltx_distillation.train_distillation import compute_latent_shapes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--num_prompts", type=int, default=8)
    args = parser.parse_args()

    # ── Distributed init ──────────────────────────────────────────────
    rank, world_size, local_rank = launch_distributed_job()
    device = torch.cuda.current_device()
    is_main = rank == 0

    config = OmegaConf.load(args.config_path)
    if args.checkpoint_path:
        config.resume_checkpoint = args.checkpoint_path

    dtype = torch.bfloat16 if config.mixed_precision else torch.float32

    # ── Build generator only (no real_score / fake_score) ────────────
    if is_main:
        print("[Eval] Building generator wrapper...")
    generator = create_ltx2_trig_wrapper(
        checkpoint_path=config.checkpoint_path,
        gemma_path=config.gemma_path,
        device=torch.device("cpu"),
        dtype=dtype,
        video_height=config.video_height,
        video_width=config.video_width,
    )
    # Move model to GPU
    generator.model = generator.model.to(device=device, dtype=dtype)

    # ── Build text encoder ───────────────────────────────────────────
    if is_main:
        print("[Eval] Building text encoder...")
    text_encoder = create_text_encoder_wrapper(
        checkpoint_path=config.checkpoint_path,
        gemma_path=config.gemma_path,
        device=device,
        dtype=dtype,
    )

    # ── Build VAEs ───────────────────────────────────────────────────
    if is_main:
        print("[Eval] Building VAEs...")
    video_vae, audio_vae = create_vae_wrappers(
        checkpoint_path=config.checkpoint_path,
        device=device,
        dtype=dtype,
    )

    # ── FSDP wrap generator + text encoder ───────────────────────────
    try:
        from ltx_core.model.transformer.transformer import BasicAVTransformerBlock
        transformer_module = (BasicAVTransformerBlock,)
    except Exception:
        transformer_module = None

    generator = fsdp_wrap(
        generator,
        sharding_strategy=config.sharding_strategy,
        mixed_precision=config.mixed_precision,
        wrap_strategy=config.generator_fsdp_wrap_strategy,
        transformer_module=transformer_module,
        cpu_offload=False,
        use_orig_params=True,
    )
    text_encoder = fsdp_wrap(
        text_encoder,
        sharding_strategy=config.sharding_strategy,
        mixed_precision=config.mixed_precision,
        wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
        cpu_offload=False,
        use_orig_params=True,
    )

    # ── Load checkpoint generator weights (after FSDP wrap) ─────────
    ckpt_path = config.resume_checkpoint
    if is_main:
        print(f"[Eval] Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    gen_sd = ckpt["generator"]
    generator.load_state_dict(gen_sd, strict=False)
    if is_main:
        print(f"[Eval] Loaded generator from step {ckpt.get('completed_step', '?')}")

    # ── Denoising schedule (rcm_trig: [π/2, backward_timesteps..., 0]) ──
    backward_trig = [float(t) for t in getattr(config, "backward_trig_timesteps", [1.5, 1.4, 1.0])]
    denoising_sigmas = torch.tensor(
        [math.pi / 2, *backward_trig, 0.0], device=device, dtype=torch.float32
    )

    # ── Build inference pipeline ─────────────────────────────────────
    pipeline = BidirectionalAVInferencePipeline(
        generator=generator,
        add_noise_fn=lambda orig, noise, sigma: (1 - sigma) * orig + sigma * noise,
        denoising_sigmas=denoising_sigmas,
        use_trigflow=True,
    )

    video_shape, audio_shape = compute_latent_shapes(
        num_frames=config.num_frames,
        video_height=config.video_height,
        video_width=config.video_width,
        batch_size=1,
    )

    # ── Load prompts ─────────────────────────────────────────────────
    prompt_path = config.data_path
    prompts = []
    with open(prompt_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    prompts = prompts[:args.num_prompts]

    generator.eval()
    video_vae.to(device=device)
    audio_vae.to(device=device)

    seed_base = config.seed
    total_tasks = len(prompts) * args.num_seeds
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Inference loop ───────────────────────────────────────────────
    for prompt_idx, prompt_text in enumerate(prompts):
        for seed_idx in range(args.num_seeds):
            task_id = prompt_idx * args.num_seeds + seed_idx
            if task_id % world_size != rank:
                continue

            seed = seed_base + prompt_idx * 100 + seed_idx

            if is_main:
                print(f"[Eval {task_id+1}/{total_tasks}] p{prompt_idx} s{seed_idx}: {prompt_text[:60]}...")

            with torch.no_grad():
                cond = text_encoder(text_prompts=[prompt_text])

            with torch.random.fork_rng(devices=[device]):
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                video, audio = pipeline.generate(
                    video_shape=video_shape,
                    audio_shape=audio_shape,
                    conditional_dict=cond,
                )

            # Decode video
            video_pixel = video_vae.decode_to_pixel(video)
            vid = video_pixel[0]
            if vid.shape[0] == 3:
                vid = vid.permute(1, 0, 2, 3)
            vid = vid.permute(0, 2, 3, 1)
            vid_uint8 = (vid.clamp(0, 1) * 255).cpu().to(torch.uint8)

            # Decode audio
            audio_waveform = None
            try:
                audio_waveform = audio_vae.decode_to_waveform(audio)
            except Exception as e:
                if is_main:
                    print(f"  [warn] audio decode: {e}")

            from torchvision.io import write_video
            out_path = os.path.join(args.output_dir, f"p{prompt_idx:02d}_s{seed_idx:02d}.mp4")
            if audio_waveform is not None:
                wav = audio_waveform[0].cpu().float()
                write_video(out_path, vid_uint8, fps=24, audio_array=wav, audio_fps=24000, audio_codec="aac")
            else:
                write_video(out_path, vid_uint8, fps=24)

            if is_main:
                print(f"  -> {out_path}")

    dist.barrier()
    if is_main:
        print(f"[Eval] Done. {total_tasks} videos saved to {args.output_dir}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
