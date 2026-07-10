provider: Cloud SSH
host: eva02 (via eva01 ProxyJump, both in ~/.ssh/config)
gpu: 1x NVIDIA RTX A6000, 46GB VRAM
driver: 575.64.03, CUDA 12.9
venv: ~/grpo_cipher/.venv on eva02 (python 3.12.3)
stack (pinned, hard-won after CUDA13/driver mismatch + torchvision/torchaudio lockstep issues):
  torch==2.6.0+cu124, torchvision==0.21.0+cu124, torchaudio==2.6.0+cu124,
  transformers==5.13.0, trl==1.7.1, peft==0.19.1, accelerate==1.14.0, datasets==5.0.0,
  bitsandbytes==0.49.2
  NOTE: vllm intentionally NOT installed -- latest vllm forces torch cu130 which needs a newer
  driver than the box has (575.64.03/CUDA 12.9); reinstalling a matching driver on a shared
  multi-user box was out of scope for this run. Using TRL's default HF-`generate()` rollout
  backend instead of vLLM colocate mode -- slower rollouts, but no infra risk to the shared host.
  verl was requested by the user but not found anywhere on eva02 (checked ~15 local venvs +
  system python) -- user confirmed proceeding with TRL instead.
disk: 3.6TB total, ~101GB free after model download (was 109GB free before this run started;
  97-98% full -- be disk-conscious, checkpoints must be cleaned up actively)
model cache: /home/alexw/grpo_cipher/hf_cache (Qwen/Qwen3-4B-Instruct-2507 downloaded, ~7GB)
