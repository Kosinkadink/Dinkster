# dinkster-model-yue2

First-party native YuE2 text-to-music pack. It detects the official combined
YuE2 3B checkpoint, runs score and semantic-token generation, samples the
64-channel acoustic flow latent through the universal sampling engine, and
decodes 48 kHz stereo audio with the checkpoint's AudioOobleck codec.

The pack owns the YuE2 model family, its four workflow nodes, combined
checkpoint planning, autoregressive text runtime, acoustic transformer, and
codec. Generic checkpoint loading, sampling, and audio output remain shared
generation surfaces. The pack decode node preserves YuE2's 64-channel latent,
unscaled AudioOobleck waveform, and 48 kHz rate while the shared audio decode
node remains MiniMax Music 3-specific (Dinkster issue #322).
