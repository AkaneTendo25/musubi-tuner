# Bundled Kandinsky 6 runtime provenance

- `core/`, `pipeline/`, `runtime/`, `configs/`, `cli.py`, and package entry files are vendored from
  `kandinskylab/kandinsky-6` commit `01c857d4571c2fe9c676f19d1c2d814e66935493`.
- `sr/` is vendored from `kandinskylab/kandinsky-6-sr` commit
  `607f775d9859c026b9138966a5ecd303f60d205d` (`v1.0.0`). The `ports/`
  checkpoint-export templates and build tooling are omitted because runtime loading does not import them.
- Absolute package imports were mechanically relocated below
  `musubi_tuner.kandinsky6.runtime` and `musubi_tuner.kandinsky6.runtime.sr`.
- The base pipeline factory accepts an optional DiT loader callback so Musubi can
  stream ConvRot INT8 weights and attach floating LoRAs before execution wrappers
  and offload registration.

The upstream license texts are preserved as `LICENSE-KANDINSKY6`,
`sr/LICENSE-KANDINSKY6-SR`, and `sr/LICENSE-APACHE`.
