# MiniMax H3 vendor drop-in

Copy the upstream MiniMax H3 Python source tree into `official/` without adapting it in place. Preserve its license and provenance
files. The copied tree must never read the checkpoint to choose or execute Python code.

Musubi-specific work stays one level up:

- `../integration.py` maps the official config, models, encoders, VAE, sampler, and checkpoint names to Musubi;
- `../load_options.py` owns CLI loading, FP8/INT8, and block-swap choices;
- `../quantization.py` owns the transformer's FP8 target/exclude key policy;
- cache and training entrypoints remain ordinary `musubi_tuner` modules.

Record the upstream repository URL and exact revision here. Do not claim a working integration until
strict loading reports zero unexplained missing or unexpected keys and a deterministic inference smoke test matches the official
implementation closely enough for its documented precision.
