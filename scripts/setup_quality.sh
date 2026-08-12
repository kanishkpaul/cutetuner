#!/bin/sh
set -eu

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install uv, then run this script again." >&2
  exit 1
fi

if command -v memory_pressure >/dev/null 2>&1; then
  free_percent="$(memory_pressure | awk -F': ' '/System-wide memory free percentage/ {gsub(/%/, "", $2); print $2}')"
  if [ -n "$free_percent" ] && [ "$free_percent" -lt 55 ]; then
    echo "Quality-model setup paused: only ${free_percent}% memory headroom is free." >&2
    echo "Close memory-heavy apps, then run this script again." >&2
    exit 2
  fi
fi

data_root="${CUTETUNER_DATA_DIR:-$HOME/Library/Application Support/CUTE Tuner}"
runtime_root="$data_root/runtimes"
model_root="$data_root/models"
separation_runtime="$runtime_root/separation"
perception_runtime="$runtime_root/perception"
clap_root="$model_root/clap/larger_clap_music_and_speech"
singmos_root="$model_root/singmos"
singmos_repo="$singmos_root/repository"
singmos_checkpoint="$singmos_root/ft_wav2vec2_large_ll60k_mdf_p1_200epochs_all_192epochs.pth"
s3prl_root="$model_root/s3prl/download"
s3prl_checkpoint="$s3prl_root/70a1c0d7bd4d235fe73bc359f70cc91a165d2405f6608fb01c504b8ba1244ac6.wav2vec_vox_new.pt"

mkdir -p "$runtime_root" "$model_root" "$clap_root" "$singmos_root" "$s3prl_root"
uv sync --extra quality

if [ ! -x "$separation_runtime/bin/python" ]; then
  uv venv "$separation_runtime" --python 3.12
fi
uv pip install --python "$separation_runtime/bin/python" \
  'torch==2.11.0' 'audio-separator[cpu]==0.44.5'

if [ ! -x "$perception_runtime/bin/python" ]; then
  uv venv "$perception_runtime" --python 3.12
fi
uv pip install --python "$perception_runtime/bin/python" \
  'torch==2.8.0' 'torchaudio==2.8.0' 'transformers==5.14.1' \
  'huggingface-hub==1.26.0' 'librosa==0.11.0' 's3prl==0.4.18' 'soundfile==0.13.1'

"$perception_runtime/bin/python" -c \
  'from huggingface_hub import snapshot_download; import sys; snapshot_download(repo_id="laion/larger_clap_music_and_speech", local_dir=sys.argv[1], allow_patterns=["*.json", "*.txt", "*.bin"], max_workers=4)' \
  "$clap_root"

clap_sha="b73c5e596fda5b29b522a1ab3f0842977fe2b147bf8b78c2ec5f7ae6267038a1"
printf '%s  %s\n' "$clap_sha" "$clap_root/pytorch_model.bin" | shasum -a 256 -c -

if [ ! -d "$singmos_repo/.git" ]; then
  git clone --depth 1 --branch v1.1.2 https://github.com/South-Twilight/SingMOS.git "$singmos_repo"
fi
singmos_commit="$(git -C "$singmos_repo" rev-parse HEAD)"
if [ "$singmos_commit" != "e55cf0713d487731ddf1661068a1aa0995e00337" ]; then
  echo "SingMOS repository does not match the pinned v1.1.2 commit." >&2
  exit 3
fi

if [ ! -f "$singmos_checkpoint" ]; then
  curl --fail --location --output "$singmos_checkpoint" \
    'https://github.com/South-Twilight/SingMOS/releases/download/ckpt_v3/ft_wav2vec2_large_ll60k_mdf_p1_200epochs_all_192epochs.pth'
fi
singmos_sha="85be234135b95d0264269d5f5ce00a361ae8fbd5ed4e6551f13cc6ce3b198f6a"
printf '%s  %s\n' "$singmos_sha" "$singmos_checkpoint" | shasum -a 256 -c -

s3prl_bytes=634937511
s3prl_current_bytes=0
if [ -f "$s3prl_checkpoint" ]; then
  s3prl_current_bytes="$(wc -c < "$s3prl_checkpoint" | tr -d ' ')"
fi
if [ "$s3prl_current_bytes" -lt "$s3prl_bytes" ]; then
  curl --fail --location --continue-at - --output "$s3prl_checkpoint" \
    'https://huggingface.co/s3prl/converted_ckpts/resolve/main/wav2vec_vox_new.pt'
fi
s3prl_sha="ba1dd46c4bfc9588b6679b66fa5a1e217d285f260117771dc19e1208fa2421cb"
printf '%s  %s\n' "$s3prl_sha" "$s3prl_checkpoint" | shasum -a 256 -c -

separator_models="$model_root/audio-separator"
mkdir -p "$separator_models"
"$separation_runtime/bin/audio-separator" \
  --model_filename vocals_mel_band_roformer.ckpt \
  --model_file_dir "$separator_models" \
  --download_model_only
separator_sha="87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e"
printf '%s  %s\n' "$separator_sha" "$separator_models/vocals_mel_band_roformer.ckpt" | \
  shasum -a 256 -c -

if ! command -v rubberband >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    brew install rubberband
  else
    echo "Rubber Band is not installed; PSOLA will remain the emergency renderer." >&2
  fi
fi

model_kib="$(du -sk "$model_root" | awk '{print $1}')"
budget_kib=5242880
if [ "$model_kib" -gt "$budget_kib" ]; then
  echo "Model storage exceeds the 5 GB budget: ${model_kib} KiB." >&2
  exit 4
fi

echo "Quality stack installed and verified. Model storage: ${model_kib} KiB / ${budget_kib} KiB."
