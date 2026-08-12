# First take: make the raw vocal easy to fix

Autotune does not make a vocal interesting. It makes an already convincing take sound
more intentional. Your clear, mature tone is the asset; the correction should be hard
to notice unless you deliberately want the effect.

1. In BandLab, start a song and use headphones. Create a **mono vocal track**. Disable
   vocal presets, echo, mastering, and AutoPitch while recording.
2. Record a dry WAV or FLAC at 44.1 or 48 kHz. Stand 10-15 cm from the mic, slightly
   off-axis, with consistent level. Leave a second of silence before and after the line.
3. Export the vocal alone, not the entire instrumental. Put it in `data/raw/`; Git will
   ignore it, so your takes cannot be pushed by accident.
4. Find the song's key from the instrumental or chords. Do not guess from one vocal
   note. Try a five-second export and listen to a few settings before processing the
   whole song.

For an intimate Radiohead-like ballad, start at `--strength 0.50 --retune-ms 110`.
That leaves gentle approaches to notes and vibrato intact. `--strength 0.90 --retune-ms
25` is an audible effect, not a neutral repair.

After correction, re-import the output to BandLab. A good beginner finishing chain is:

- high-pass EQ around 70-90 Hz;
- light compression (roughly 2:1, aiming for 2-4 dB of gain reduction);
- a short, quiet plate or room reverb;
- no AutoPitch on top until you have compared it against the dry/corrected result.

Always level-match comparisons. Louder almost always sounds “better,” even when it is
less natural.
