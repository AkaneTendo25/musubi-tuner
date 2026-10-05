import math

import torch

from musubi_tuner.networks.lora import LoRAModule, _profile_scope


class SDSLoRAModule(LoRAModule):
    """Linear SDS-LoRA adapter with detached, periodically refreshed bases."""

    def __init__(self, *args, sds_warmup_steps: int = 10, sds_update_phases: int = 5, **kwargs):
        org_module = args[1] if len(args) > 1 else kwargs.get("org_module")
        if not isinstance(org_module, torch.nn.Linear):
            raise ValueError("SDS-LoRA supports Linear modules only; Conv2d and Conv3d adapters are not supported")
        if kwargs.get("split_dims") is not None:
            raise ValueError("SDS-LoRA does not support split_dims")
        if kwargs.get("nora", "off") != "off":
            raise ValueError("SDS-LoRA is incompatible with nora")
        if kwargs.get("init", "default") != "default":
            raise ValueError("SDS-LoRA is incompatible with init=bimi")
        for name in ("dropout", "rank_dropout", "module_dropout"):
            if kwargs.get(name) is not None:
                raise ValueError(f"SDS-LoRA is incompatible with {name}")
        super().__init__(*args, **kwargs)
        if sds_warmup_steps < 1:
            raise ValueError("sds_warmup_steps must be at least 1")
        if sds_update_phases < 1:
            raise ValueError("sds_update_phases must be at least 1")

        self.sds_warmup_steps = int(sds_warmup_steps)
        self.sds_update_phases = int(sds_update_phases)
        if self.lora_dim > min(self.lora_down.in_features, self.lora_up.out_features):
            raise ValueError(
                "SDS-LoRA rank must not exceed either dimension of the adapted Linear module "
                f"(rank={self.lora_dim}, in={self.lora_down.in_features}, out={self.lora_up.out_features})"
            )
        self.sds_enabled = True
        self.sds_scale = float(self.scale) * math.sqrt(self.lora_dim)
        if not math.isfinite(self.sds_scale) or self.sds_scale <= 0:
            raise ValueError(f"SDS-LoRA requires a positive finite scale, got {self.sds_scale}")
        self.scale = self.sds_scale
        in_dim = self.lora_down.in_features
        out_dim = self.lora_up.out_features
        self.register_buffer("sds_qa", torch.zeros(in_dim, self.lora_dim), persistent=True)
        self.register_buffer("sds_qb", torch.zeros(out_dim, self.lora_dim), persistent=True)
        self.register_buffer("sds_active", torch.tensor(False), persistent=True)
        self._sds_active_cached = False

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        saved_alpha = state_dict.get(prefix + "alpha")
        if saved_alpha is not None:
            saved_scale = float(saved_alpha.item()) / math.sqrt(self.lora_dim)
            if not math.isclose(saved_scale, self.sds_scale, rel_tol=1e-6, abs_tol=1e-8):
                error_msgs.append(f"SDS-LoRA alpha mismatch for {prefix}: resume with the original --network_alpha")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        self._sds_active_cached = bool(self.sds_active.item())

    @torch.no_grad()
    def activate_sds(self):
        """Reparameterize the scaled warmup delta according to Equation 8."""
        with torch.autocast(device_type=self.lora_down.weight.device.type, enabled=False):
            a = self.lora_down.weight.float()
            b = self.lora_up.weight.float()
            qa, ra = torch.linalg.qr(a.t(), mode="reduced")
            qb, rb = torch.linalg.qr(b, mode="reduced")
            core = self.sds_scale * (rb @ ra.t())
            u, singular, vh = torch.linalg.svd(core, full_matrices=False)
            left = qb @ u
            right = qa @ vh.t()
            coefficient = singular / (2.0 * self.sds_scale)
            new_a = coefficient[:, None] * right.t()
            new_b = left * coefficient[None, :]
        self.lora_down.weight.copy_(new_a.to(self.lora_down.weight))
        self.lora_up.weight.copy_(new_b.to(self.lora_up.weight))
        self.sds_qa.copy_(right.to(self.sds_qa))
        self.sds_qb.copy_(left.to(self.sds_qb))
        self.sds_active.fill_(True)
        self._sds_active_cached = True

    @staticmethod
    def _align_qr_signs(new_basis: torch.Tensor, old_basis: torch.Tensor) -> torch.Tensor:
        signs = torch.sign((new_basis * old_basis.float()).sum(dim=0))
        signs.masked_fill_(signs == 0, 1)
        return new_basis * signs

    @torch.no_grad()
    def refresh_sds_bases(self):
        if not self._sds_active_cached:
            return
        with torch.autocast(device_type=self.lora_down.weight.device.type, enabled=False):
            qa = torch.linalg.qr(self.lora_down.weight.float().t(), mode="reduced").Q
            qb = torch.linalg.qr(self.lora_up.weight.float(), mode="reduced").Q
            qa = self._align_qr_signs(qa, self.sds_qa)
            qb = self._align_qr_signs(qb, self.sds_qb)
        self.sds_qa.copy_(qa.to(self.sds_qa))
        self.sds_qb.copy_(qb.to(self.sds_qb))

    def _lora_forward(self, x):
        if not self._sds_active_cached:
            return super()._lora_forward(x)
        with _profile_scope("h3.lora.base"):
            org_forwarded = self.org_forward(x)
        lora_input = self._lora_input(x)
        qa = self.sds_qa.to(device=lora_input.device, dtype=lora_input.dtype)
        qb = self.sds_qb.to(device=lora_input.device, dtype=lora_input.dtype)
        branch_a = torch.nn.functional.linear(self.lora_down(lora_input), qb)
        branch_b = self.lora_up(torch.nn.functional.linear(lora_input, qa.t()))
        delta = branch_a + branch_b
        return self._match_org_dtype(self._fuse_delta(org_forwarded, delta, self.sds_scale), org_forwarded)

    def export_state_dict(self, destination=None, prefix: str = ""):
        if not self._sds_active_cached:
            return {
                prefix + "lora_down.weight": self.lora_down.weight.detach(),
                prefix + "lora_up.weight": self.lora_up.weight.detach(),
                prefix + "alpha": torch.tensor(self.lora_dim * self.sds_scale, device=self.alpha.device, dtype=torch.float32),
            }
        down = torch.cat((self.lora_down.weight.detach(), self.sds_qa.t().to(self.lora_down.weight)), dim=0)
        up = torch.cat((self.sds_qb.to(self.lora_up.weight), self.lora_up.weight.detach()), dim=1)
        return {
            prefix + "lora_down.weight": down,
            prefix + "lora_up.weight": up,
            prefix + "alpha": torch.tensor(2 * self.lora_dim * self.sds_scale, device=self.alpha.device, dtype=torch.float32),
        }

    def export_rank(self) -> int:
        return self.lora_dim * (2 if self._sds_active_cached else 1)

    def export_alpha(self) -> float:
        return self.export_rank() * self.sds_scale

    def delta_norm_sq(self) -> torch.Tensor:
        """Squared Frobenius norm of the effective delta from rank-sized Gram matrices."""
        with torch.autocast(device_type=self.lora_down.weight.device.type, enabled=False):
            a = self.lora_down.weight.float()
            b = self.lora_up.weight.float()
            if self._sds_active_cached:
                qa = self.sds_qa.float()
                qb = self.sds_qb.float()
                value = torch.trace((qb.t() @ qb) @ (a @ a.t()))
                value = value + torch.trace((b.t() @ b) @ (qa.t() @ qa))
                value = value + 2 * torch.trace((qb.t() @ b) @ (a @ qa).t())
            else:
                value = torch.trace((b.t() @ b) @ (a @ a.t()))
            factor = self.multiplier * self.sds_scale
            return (value * (factor * factor)).clamp_min(0)


def clear_sds_optimizer_state(optimizer, modules):
    """Drop moments for reparameterized SDS tensors through optimizer wrappers."""
    while hasattr(optimizer, "optimizer"):
        optimizer = optimizer.optimizer
    state = getattr(optimizer, "state", None)
    if state is None:
        return
    for module in modules:
        state.pop(module.lora_down.weight, None)
        state.pop(module.lora_up.weight, None)
