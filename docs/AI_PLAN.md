# AI plan: use it only where it earns its keep

Do not begin by training a model to “sound like a better singer.” That has no clear
target and tends to erase the specific character you actually like in your voice.

Phase 1 is the included hybrid chain: full torchcrepe pitch tracking, a time-varying
chord/key map, chord-aware note inference, protected vocal gestures, and a continuous Rubber
Band pitch map. Technical measurements, music-aware CLAP descriptors, and SingMOS-Pro's
naturalness prediction are shown with their source and uncertainty. None is presented as an
artistic verdict. A local LLM can translate the evidence and producer brief into one bounded
tuning profile, but it never receives arbitrary control over the waveform or note targets.

Phase 2 is a small personal denoiser. Record 10-30 minutes of your cleanest dry takes,
split by *song* into train/validation, and train `scripts/train_denoiser.py`. It learns
to remove synthetic low-level noise, not to clone a performer. Keep the best checkpoint
only if held-out recordings sound better in a blind A/B comparison.

Phase 3, only if the full tracker regularly fails on your voice, is a separately validated
replacement such as RMVPE. Measure voiced/unvoiced errors and octave errors on 20 manually
checked clips before changing the default. A more complicated tracker is not automatically
a better vocal.

Never train or run a voice-conversion model on someone else's recordings without their
explicit permission. This repo is designed to keep your recordings and model weights
local and ignored by Git.
