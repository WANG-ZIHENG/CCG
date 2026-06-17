#!/usr/bin/env python3
"""
CCG inference — 用真实 FairVLMed 样本生成 fundus 图。
输出 [真实底图 | 真实 cup/disc 掩码 | 生成图] 三联拼图，方便人工对比。

数据集：/root/autodl-tmp/data/fairvlmed10k（论文使用的 FairVLMed 7363 子集所属父集）
prompt 重现 tutorial_dataset.MyDataset 在 fairvlmed10k 上的格式：
    "Age <young/old>, <race>, <ethnicity>, <language>, <maritalstatus>, <note>, <gpt4_summary>"

用法：
    python run_inference.py                          # 默认 ckpt + 3 张
    python run_inference.py <ckpt_path>
    python run_inference.py <ckpt_path> <n_samples>
    python run_inference.py <ckpt_path> <n_samples> <out_dir>
"""
import os
import sys
import numpy as np
import pandas as pd
import einops
import cv2
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


# ---- default checkpoint: CCG ControlNet on SD 2.1 ----
# Download the released weight into `<repo_root>/checkpoints/`
# or pass an explicit path as the first positional argument.
# Both `.safetensors` (fp16, recommended) and `.ckpt` are supported.
DEFAULT_CKPT = os.path.join(HERE, "checkpoints", "ccg_controlnet_sd21.safetensors")
CONFIG       = os.path.join(HERE, "models", "cldm_v21.yaml")
DATA_DIR     = "/root/autodl-tmp/data/fairvlmed10k"
DEFAULT_OUT  = os.path.join(HERE, "gen_out")

DDIM_STEPS   = 30
GUIDANCE     = 9.0
STRENGTH     = 1.0
CROP_SIZE    = 512
SEED         = 42


def pick_samples(n=3):
    """从 data_summary.csv 的 test 集挑 n 个样本，2 个 glaucoma + 1 个 healthy。"""
    csv = pd.read_csv(os.path.join(DATA_DIR, "data_summary.csv"))
    csv = csv[csv["use"] == "test"]
    g = csv[csv["glaucoma"] == "yes"].head(n - 1)
    h = csv[csv["glaucoma"] == "no" ].head(1)
    return pd.concat([g, h]).head(n).reset_index(drop=True)


def build_input(row):
    """复现 tutorial_dataset.MyDataset.__getitem__（fairvlmed10k 分支）。"""
    fname = row["filename"]
    npz = np.load(os.path.join(DATA_DIR, "All", fname), allow_pickle=True)

    # --- target: slo_fundus -> CLAHE -> 3 通道 ---
    fundus = npz["slo_fundus"].astype(np.uint8)
    fundus3 = np.stack([fundus] * 3, axis=-1)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    target = np.stack([clahe.apply(fundus3[:, :, i]) for i in range(3)], axis=-1)

    # --- source: 从 <dataset_dir>/mask/<basename>_predict.png 读取 cup/disc 掩码 ---
    mask_name = fname.replace(".npz", "") + "_predict.png"
    mask_path = os.path.join(DATA_DIR, "mask", mask_name)
    mask_pil = Image.open(mask_path).convert("RGB")
    mask3 = np.array(mask_pil)

    # --- 中心裁剪 512x512 ---
    H, W = target.shape[:2]
    sh = max((H - CROP_SIZE) // 2, 0)
    sw = max((W - CROP_SIZE) // 2, 0)
    target = target[sh:sh + CROP_SIZE, sw:sw + CROP_SIZE]
    mask3  = mask3 [sh:sh + CROP_SIZE, sw:sw + CROP_SIZE]

    # --- BGR -> RGB to match the training-time channel order ---
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    mask3  = cv2.cvtColor(mask3,  cv2.COLOR_BGR2RGB)

    # --- prompt: e.g. "Male, White, Non-Hispanic, with Glaucoma" ---
    gender    = str(row["gender"]).capitalize()
    race      = str(row["race"]).capitalize()
    ethnicity = "-".join(p.capitalize() for p in str(row["ethnicity"]).split("-"))
    glauc     = "with Glaucoma" if row["glaucoma"] == "yes" else "without Glaucoma"
    prompt = f"{gender}, {race}, {ethnicity}, {glauc}"

    return target, mask3, prompt, fname


def generate(model, sampler, mask3_u8, prompt, seed=SEED):
    seed_everything(seed)
    control = mask3_u8.astype(np.float32) / 255.0
    control = einops.rearrange(
        torch.from_numpy(control).float().cuda(), "h w c -> 1 c h w"
    )
    n_prompt = "low quality, blurry, artifacts"
    cond    = {"c_concat": [control],
               "c_crossattn": [model.get_learned_conditioning([prompt])]}
    un_cond = {"c_concat": [control],
               "c_crossattn": [model.get_learned_conditioning([n_prompt])]}
    model.control_scales = [STRENGTH] * 13
    samples, _ = sampler.sample(
        DDIM_STEPS, 1, (4, CROP_SIZE // 8, CROP_SIZE // 8), cond,
        verbose=False,
        unconditional_guidance_scale=GUIDANCE,
        unconditional_conditioning=un_cond,
    )
    x = model.decode_first_stage(samples)
    x = (einops.rearrange(x, "b c h w -> b h w c").cpu().numpy() * 127.5 + 127.5)
    return x.clip(0, 255).astype(np.uint8)[0]


def main():
    # Positional args; empty string falls back to defaults so users can do
    #   python run_inference.py "" 6 ./gen_out  (override n + out_dir only)
    ckpt    = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else DEFAULT_CKPT
    n       = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else 3
    out_dir = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else DEFAULT_OUT
    os.makedirs(out_dir, exist_ok=True)

    assert torch.cuda.is_available(), "this needs a GPU"
    print(f"[ckpt]   {ckpt}")
    print(f"[config] {CONFIG}")
    print(f"[data]   {DATA_DIR}")
    print(f"[out]    {out_dir}")

    print("[1] loading model ...")
    model = create_model(CONFIG).cpu()
    if ckpt.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(ckpt, device="cpu")
    else:
        state_dict = load_state_dict(ckpt, location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model = model.cuda().eval()
    sampler = DDIMSampler(model)
    print(f"    params: {sum(p.numel() for p in model.parameters())/1e9:.2f} B")

    rows = pick_samples(n)
    print(f"[2] selected {len(rows)} test-set samples: {rows['filename'].tolist()}")

    summary_path = os.path.join(out_dir, "manifest.txt")
    with open(summary_path, "w") as mf:
        for i, row in rows.iterrows():
            label = "Glaucoma" if row["glaucoma"] == "yes" else "Health"
            print(f"\n--- sample {i+1}/{len(rows)}: {row['filename']} ({label}) ---")
            target, mask3, prompt, fname = build_input(row)
            print(f"    prompt: {prompt!r}")
            print(f"    generating ...")
            gen = generate(model, sampler, mask3, prompt, seed=SEED + i)

            triptych = np.concatenate([target, mask3, gen], axis=1)
            out_png = os.path.join(out_dir, f"{i:02d}_{fname.replace('.npz','')}_{label}.png")
            Image.fromarray(triptych).save(out_png)
            print(f"    saved  -> {out_png}")
            mf.write(f"{out_png}\tlabel={label}\tprompt={prompt[:200]}\n")

    print(f"\nALL DONE. Manifest: {summary_path}")


if __name__ == "__main__":
    main()
