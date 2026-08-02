# MiniMax H3 vendor drop-in

Copy the upstream MiniMax H3 Python source tree into `official/` without adapting it in place. Preserve its license and provenance
files. The copied tree must never read the checkpoint to choose or execute Python code.

Musubi-specific work stays one level up:

- `../integration.py` maps the official config, models, encoders, VAE, sampler, and checkpoint names to Musubi;
- the component-specific factories in `../backend.py` and `../integration.py` ensure each script loads only what it uses;
- cache and training entrypoints remain ordinary `musubi_tuner` modules.

Add transformer quantization and block swapping only after the released checkpoint keys and transformer block boundaries have been
validated with real forward and backward passes.

Record the upstream repository URL and exact revision here. Do not claim a working integration until
strict loading reports zero unexplained missing or unexpected keys and a deterministic inference smoke test matches the official
implementation closely enough for its documented precision.
