from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Noll index is 1-based in config/spec; model tensors are 0-based.
DEFAULT_GROUPS_NOLL = {
    "special": [1, 2, 3],
    "low": [4, 5, 6],
    "mid": [7, 8, 9, 10, 11, 12, 13, 14, 15],
    "high": [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
}


def _noll_to_zero_based(indices: List[int]) -> List[int]:
    return [i - 1 for i in indices]


def build_group_indices(groups_noll: Optional[Dict[str, List[int]]] = None) -> Dict[str, List[int]]:
    groups_noll = groups_noll or DEFAULT_GROUPS_NOLL
    groups = {k: _noll_to_zero_based(v) for k, v in groups_noll.items()}
    all_idx = sorted(groups["special"] + groups["low"] + groups["mid"] + groups["high"])
    if all_idx != list(range(36)):
        raise ValueError(f"Grouped indices must exactly cover 36 modes, got: {all_idx}")
    return groups


def _unfold_3x3(x_bchw: torch.Tensor, padding_mode: str = "replicate") -> torch.Tensor:
    # Returns neighborhood tensor with shape [B, C, H, W, 9].
    if padding_mode not in ("replicate", "reflect", "zeros"):
        raise ValueError(f"Unsupported padding mode: {padding_mode}")
    if padding_mode == "zeros":
        x_pad = F.pad(x_bchw, (1, 1, 1, 1), mode="constant", value=0.0)
    else:
        x_pad = F.pad(x_bchw, (1, 1, 1, 1), mode=padding_mode)
    unfolded = F.unfold(x_pad, kernel_size=3, stride=1)
    b, c9, hw = unfolded.shape
    c = x_bchw.shape[1]
    return unfolded.view(b, c, 9, hw).view(b, c, 9, x_bchw.shape[2], x_bchw.shape[3]).permute(0, 1, 3, 4, 2)


def _group_forward_core(
    z_group_bhwc: torch.Tensor,
    bias: torch.Tensor,
    alpha: torch.Tensor,
    amax: torch.Tensor,
    eps: torch.Tensor,
    gss: torch.Tensor,
    p_sat: torch.Tensor,
    padding_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # u keeps sign information. Shape: [B, H, W, Cg]
    u = z_group_bhwc + bias.view(1, 1, 1, 1)

    # Local 3x3 joint energy over neighbors and group channels.
    # sqrt(u^2 + eps) is soft absolute value and sign-aware.
    u_bchw = u.permute(0, 3, 1, 2)
    u_nb = _unfold_3x3(u_bchw, padding_mode=padding_mode)  # [B, Cg, H, W, 9]
    soft_abs = torch.sqrt(u_nb.pow(2) + eps.view(1, 1, 1, 1, 1))
    s = soft_abs.sum(dim=1).sum(dim=-1)  # [B, H, W]

    # NewBP-style shared gain.
    gain = gss.view(1, 1, 1) / (1.0 + s / p_sat.view(1, 1, 1))

    # Bounded output while preserving sign.
    v = alpha.view(1, 1, 1, 1) * gain.unsqueeze(-1) * u
    out = amax.view(1, 1, 1, 1) * torch.tanh(v)
    return out, gain, s


class LocalGroupedZernikeNewBPFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        raw_coeffs,
        special_bias,
        special_alpha,
        special_amax,
        special_eps,
        low_bias,
        low_alpha,
        low_amax,
        low_eps,
        low_gss,
        low_p_sat,
        mid_bias,
        mid_alpha,
        mid_amax,
        mid_eps,
        mid_gss,
        mid_p_sat,
        high_bias,
        high_alpha,
        high_amax,
        high_eps,
        high_gss,
        high_p_sat,
        groups_tensor,
        local_joint_enabled,
        special_use_neighborhood,
        padding_mode_idx,
    ):
        groups = {
            "special": groups_tensor[0].tolist(),
            "low": groups_tensor[1].tolist(),
            "mid": groups_tensor[2].tolist(),
            "high": groups_tensor[3].tolist(),
        }
        groups = {k: [i for i in v if i >= 0] for k, v in groups.items()}
        padding_mode = ["replicate", "reflect", "zeros"][int(padding_mode_idx.item())]

        out = raw_coeffs.clone()

        sidx = groups["special"]
        z_special = raw_coeffs[..., sidx]
        u_special = z_special + special_bias.view(1, 1, 1, 1)
        out_special = special_amax.view(1, 1, 1, 1) * torch.tanh(special_alpha.view(1, 1, 1, 1) * u_special)
        out[..., sidx] = out_special

        if bool(local_joint_enabled.item()):
            for name, bias, alpha, amax, eps, gss, p_sat in (
                ("low", low_bias, low_alpha, low_amax, low_eps, low_gss, low_p_sat),
                ("mid", mid_bias, mid_alpha, mid_amax, mid_eps, mid_gss, mid_p_sat),
                ("high", high_bias, high_alpha, high_amax, high_eps, high_gss, high_p_sat),
            ):
                idx = groups[name]
                gout, _, _ = _group_forward_core(raw_coeffs[..., idx], bias, alpha, amax, eps, gss, p_sat, padding_mode)
                out[..., idx] = gout

        # Save tensors for backward; actual gradient is recomputed by autograd on core ops.
        ctx.save_for_backward(
            raw_coeffs,
            special_bias,
            special_alpha,
            special_amax,
            special_eps,
            low_bias,
            low_alpha,
            low_amax,
            low_eps,
            low_gss,
            low_p_sat,
            mid_bias,
            mid_alpha,
            mid_amax,
            mid_eps,
            mid_gss,
            mid_p_sat,
            high_bias,
            high_alpha,
            high_amax,
            high_eps,
            high_gss,
            high_p_sat,
            groups_tensor,
            local_joint_enabled,
            special_use_neighborhood,
            padding_mode_idx,
        )
        return out

    @staticmethod
    def _forward_no_ctx(
        raw_coeffs,
        special_bias,
        special_alpha,
        special_amax,
        special_eps,
        low_bias,
        low_alpha,
        low_amax,
        low_eps,
        low_gss,
        low_p_sat,
        mid_bias,
        mid_alpha,
        mid_amax,
        mid_eps,
        mid_gss,
        mid_p_sat,
        high_bias,
        high_alpha,
        high_amax,
        high_eps,
        high_gss,
        high_p_sat,
        groups_tensor,
        local_joint_enabled,
        special_use_neighborhood,
        padding_mode_idx,
    ):
        groups = {
            "special": groups_tensor[0].tolist(),
            "low": groups_tensor[1].tolist(),
            "mid": groups_tensor[2].tolist(),
            "high": groups_tensor[3].tolist(),
        }
        groups = {k: [i for i in v if i >= 0] for k, v in groups.items()}
        padding_mode = ["replicate", "reflect", "zeros"][int(padding_mode_idx.item())]

        out = raw_coeffs.clone()

        sidx = groups["special"]
        z_special = raw_coeffs[..., sidx]
        u_special = z_special + special_bias.view(1, 1, 1, 1)
        if bool(special_use_neighborhood.item()) and bool(local_joint_enabled.item()):
            sp_out, _, _ = _group_forward_core(
                z_special,
                special_bias,
                special_alpha,
                special_amax,
                special_eps,
                torch.ones_like(special_amax),
                torch.ones_like(special_amax),
                padding_mode,
            )
            out[..., sidx] = sp_out
        else:
            out_special = special_amax.view(1, 1, 1, 1) * torch.tanh(special_alpha.view(1, 1, 1, 1) * u_special)
            out[..., sidx] = out_special

        if bool(local_joint_enabled.item()):
            for name, bias, alpha, amax, eps, gss, p_sat in (
                ("low", low_bias, low_alpha, low_amax, low_eps, low_gss, low_p_sat),
                ("mid", mid_bias, mid_alpha, mid_amax, mid_eps, mid_gss, mid_p_sat),
                ("high", high_bias, high_alpha, high_amax, high_eps, high_gss, high_p_sat),
            ):
                idx = groups[name]
                gout, _, _ = _group_forward_core(raw_coeffs[..., idx], bias, alpha, amax, eps, gss, p_sat, padding_mode)
                out[..., idx] = gout
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0]
        (
            raw_coeffs,
            special_bias,
            special_alpha,
            special_amax,
            special_eps,
            low_bias,
            low_alpha,
            low_amax,
            low_eps,
            low_gss,
            low_p_sat,
            mid_bias,
            mid_alpha,
            mid_amax,
            mid_eps,
            mid_gss,
            mid_p_sat,
            high_bias,
            high_alpha,
            high_amax,
            high_eps,
            high_gss,
            high_p_sat,
            groups_tensor,
            local_joint_enabled,
            special_use_neighborhood,
            padding_mode_idx,
        ) = ctx.saved_tensors

        # Rebuild differentiable graph and let PyTorch compute exact local-coupled Jacobian.
        with torch.enable_grad():
            req_flags = [
                True,
                bool(special_bias.requires_grad),
                bool(special_alpha.requires_grad),
                bool(special_amax.requires_grad),
                bool(special_eps.requires_grad),
                bool(low_bias.requires_grad),
                bool(low_alpha.requires_grad),
                bool(low_amax.requires_grad),
                bool(low_eps.requires_grad),
                bool(low_gss.requires_grad),
                bool(low_p_sat.requires_grad),
                bool(mid_bias.requires_grad),
                bool(mid_alpha.requires_grad),
                bool(mid_amax.requires_grad),
                bool(mid_eps.requires_grad),
                bool(mid_gss.requires_grad),
                bool(mid_p_sat.requires_grad),
                bool(high_bias.requires_grad),
                bool(high_alpha.requires_grad),
                bool(high_amax.requires_grad),
                bool(high_eps.requires_grad),
                bool(high_gss.requires_grad),
                bool(high_p_sat.requires_grad),
            ]
            inputs = [
                raw_coeffs.detach().requires_grad_(True),
                special_bias.detach().requires_grad_(True),
                special_alpha.detach().requires_grad_(True),
                special_amax.detach().requires_grad_(True),
                special_eps.detach().requires_grad_(True),
                low_bias.detach().requires_grad_(True),
                low_alpha.detach().requires_grad_(True),
                low_amax.detach().requires_grad_(True),
                low_eps.detach().requires_grad_(True),
                low_gss.detach().requires_grad_(True),
                low_p_sat.detach().requires_grad_(True),
                mid_bias.detach().requires_grad_(True),
                mid_alpha.detach().requires_grad_(True),
                mid_amax.detach().requires_grad_(True),
                mid_eps.detach().requires_grad_(True),
                mid_gss.detach().requires_grad_(True),
                mid_p_sat.detach().requires_grad_(True),
                high_bias.detach().requires_grad_(True),
                high_alpha.detach().requires_grad_(True),
                high_amax.detach().requires_grad_(True),
                high_eps.detach().requires_grad_(True),
                high_gss.detach().requires_grad_(True),
                high_p_sat.detach().requires_grad_(True),
            ]

            out = LocalGroupedZernikeNewBPFunction._forward_no_ctx(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                inputs[5],
                inputs[6],
                inputs[7],
                inputs[8],
                inputs[9],
                inputs[10],
                inputs[11],
                inputs[12],
                inputs[13],
                inputs[14],
                inputs[15],
                inputs[16],
                inputs[17],
                inputs[18],
                inputs[19],
                inputs[20],
                inputs[21],
                inputs[22],
                groups_tensor,
                local_joint_enabled,
                special_use_neighborhood,
                padding_mode_idx,
            )

            grads = torch.autograd.grad(
                outputs=out,
                inputs=inputs,
                grad_outputs=grad_output,
                allow_unused=True,
                retain_graph=False,
            )

        grads = tuple(g if g is not None else None for g in grads)
        grads = tuple(grads[i] if req_flags[i] else None for i in range(len(grads)))
        return (
            grads[0],
            grads[1],
            grads[2],
            grads[3],
            grads[4],
            grads[5],
            grads[6],
            grads[7],
            grads[8],
            grads[9],
            grads[10],
            grads[11],
            grads[12],
            grads[13],
            grads[14],
            grads[15],
            grads[16],
            grads[17],
            grads[18],
            grads[19],
            grads[20],
            grads[21],
            grads[22],
            None,
            None,
            None,
            None,
        )


class LocalGroupedZernikeNewBP(nn.Module):
    """Local joint NewBP layer for Zernike coefficients.

    Why this exists:
    It constrains AberrationNet raw coefficients with local 3x3 coupled gain
    competition and group-wise energy allocation, while keeping the original
    Zernike->PSF physics model unchanged.
    """

    def __init__(
        self,
        groups_noll: Optional[Dict[str, List[int]]] = None,
        local_joint_enabled: bool = True,
        kernel_size: int = 3,
        padding_mode: str = "replicate",
        implementation: str = "native_autograd",
        separate_special_group: bool = True,
        special_use_neighborhood: bool = False,
        params: Optional[Dict[str, Dict[str, float]]] = None,
        learnable: Optional[Dict[str, bool]] = None,
    ):
        super().__init__()
        if kernel_size != 3:
            raise ValueError("Only 3x3 kernel is supported for local joint NewBP.")
        if padding_mode not in ("replicate", "reflect", "zeros"):
            raise ValueError(f"Unsupported padding_mode: {padding_mode}")
        if implementation not in ("native_autograd", "custom_autograd"):
            raise ValueError(f"Unsupported implementation: {implementation}")

        self.groups = build_group_indices(groups_noll)
        self.local_joint_enabled = bool(local_joint_enabled)
        self.padding_mode = padding_mode
        self.implementation = implementation
        self.separate_special_group = bool(separate_special_group)
        self.special_use_neighborhood = bool(special_use_neighborhood)

        default_params = {
            "special": {"alpha": 1.0, "amax": 0.35, "bias": 0.0, "eps": 1e-6, "gss": 1.0, "p_sat": 4.0},
            "low": {"alpha": 1.0, "amax": 0.30, "bias": 0.0, "eps": 1e-6, "gss": 1.0, "p_sat": 4.0},
            "mid": {"alpha": 1.0, "amax": 0.18, "bias": 0.0, "eps": 1e-6, "gss": 1.0, "p_sat": 2.5},
            "high": {"alpha": 1.0, "amax": 0.10, "bias": 0.0, "eps": 1e-6, "gss": 1.0, "p_sat": 1.2},
        }
        if params is not None:
            for g in default_params:
                if g in params:
                    default_params[g].update(params[g])

        default_learnable = {
            "bias": True,
            "alpha": False,
            "amax": False,
            "gss": False,
            "p_sat": False,
            "eps": False,
        }
        if learnable is not None:
            default_learnable.update(learnable)

        self._register_group_params(default_params, default_learnable)

        # Fixed-shape tensor for custom function argument packing.
        groups_tensor = torch.full((4, 21), -1, dtype=torch.long)
        for gi, name in enumerate(["special", "low", "mid", "high"]):
            idx = self.groups[name]
            groups_tensor[gi, : len(idx)] = torch.tensor(idx, dtype=torch.long)
        self.register_buffer("groups_tensor", groups_tensor)

        self.last_stats: Dict[str, float] = {}

    def _make_scalar(self, value: float, learnable: bool) -> nn.Parameter:
        p = nn.Parameter(torch.tensor(float(value), dtype=torch.float32), requires_grad=learnable)
        return p

    def _register_group_params(self, params: Dict[str, Dict[str, float]], learnable: Dict[str, bool]) -> None:
        for name in ["special", "low", "mid", "high"]:
            gp = params[name]
            setattr(self, f"{name}_bias", self._make_scalar(gp["bias"], learnable["bias"]))
            setattr(self, f"{name}_alpha", self._make_scalar(gp["alpha"], learnable["alpha"]))
            setattr(self, f"{name}_amax", self._make_scalar(gp["amax"], learnable["amax"]))
            setattr(self, f"{name}_eps", self._make_scalar(max(gp["eps"], 1e-12), learnable["eps"]))
            setattr(self, f"{name}_gss", self._make_scalar(gp.get("gss", 1.0), learnable["gss"]))
            setattr(self, f"{name}_p_sat", self._make_scalar(max(gp.get("p_sat", 1e-6), 1e-6), learnable["p_sat"]))

    def _forward_native(self, raw_coeffs_bhwc: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        out = raw_coeffs_bhwc.clone()
        stats_tensors: Dict[str, torch.Tensor] = {}

        # Special group: conservative bounded mapping by default.
        sidx = self.groups["special"]
        z_special = raw_coeffs_bhwc[..., sidx]
        u_special = z_special + self.special_bias.view(1, 1, 1, 1)
        if self.special_use_neighborhood and self.local_joint_enabled:
            sp_out, sp_gain, sp_s = _group_forward_core(
                z_special,
                self.special_bias,
                self.special_alpha,
                self.special_amax,
                self.special_eps,
                self.special_gss,
                self.special_p_sat,
                self.padding_mode,
            )
            out[..., sidx] = sp_out
            stats_tensors["gain_special"] = sp_gain
            stats_tensors["energy_special"] = sp_s
        else:
            out[..., sidx] = self.special_amax.view(1, 1, 1, 1) * torch.tanh(self.special_alpha.view(1, 1, 1, 1) * u_special)

        if self.local_joint_enabled:
            for name in ["low", "mid", "high"]:
                idx = self.groups[name]
                gout, gain, s = _group_forward_core(
                    raw_coeffs_bhwc[..., idx],
                    getattr(self, f"{name}_bias"),
                    getattr(self, f"{name}_alpha"),
                    getattr(self, f"{name}_amax"),
                    getattr(self, f"{name}_eps"),
                    getattr(self, f"{name}_gss"),
                    getattr(self, f"{name}_p_sat"),
                    self.padding_mode,
                )
                out[..., idx] = gout
                stats_tensors[f"gain_{name}"] = gain
                stats_tensors[f"energy_{name}"] = s

        return out, stats_tensors

    def _update_stats(self, raw: torch.Tensor, out: torch.Tensor, stats_tensors: Dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            st: Dict[str, float] = {
                "raw_mean": float(raw.mean().item()),
                "raw_std": float(raw.std().item()),
                "raw_abs_mean": float(raw.abs().mean().item()),
                "raw_max_abs": float(raw.abs().max().item()),
                "out_mean": float(out.mean().item()),
                "out_std": float(out.std().item()),
                "out_abs_mean": float(out.abs().mean().item()),
                "out_max_abs": float(out.abs().max().item()),
            }
            for name in ["special", "low", "mid", "high"]:
                idx = self.groups[name]
                g = out[..., idx]
                abs_g = g.abs().reshape(-1)
                st[f"group_{name}_mean"] = float(g.mean().item())
                st[f"group_{name}_abs_mean"] = float(abs_g.mean().item())
                st[f"group_{name}_p50"] = float(torch.quantile(abs_g, 0.50).item())
                st[f"group_{name}_p90"] = float(torch.quantile(abs_g, 0.90).item())
                st[f"group_{name}_p99"] = float(torch.quantile(abs_g, 0.99).item())
            for key, value in stats_tensors.items():
                st[f"{key}_mean"] = float(value.mean().item())
                st[f"{key}_std"] = float(value.std().item())
                st[f"{key}_max"] = float(value.max().item())
            self.last_stats = st

    def get_last_stats(self) -> Dict[str, float]:
        return dict(self.last_stats)

    def forward(self, raw_coeffs_bhwc: torch.Tensor) -> torch.Tensor:
        if raw_coeffs_bhwc.dim() != 4 or raw_coeffs_bhwc.shape[-1] != 36:
            raise ValueError(f"Expected [B, H, W, 36], got {tuple(raw_coeffs_bhwc.shape)}")

        if self.implementation == "custom_autograd":
            out = LocalGroupedZernikeNewBPFunction.apply(
                raw_coeffs_bhwc,
                self.special_bias,
                self.special_alpha,
                self.special_amax,
                self.special_eps,
                self.low_bias,
                self.low_alpha,
                self.low_amax,
                self.low_eps,
                self.low_gss,
                self.low_p_sat,
                self.mid_bias,
                self.mid_alpha,
                self.mid_amax,
                self.mid_eps,
                self.mid_gss,
                self.mid_p_sat,
                self.high_bias,
                self.high_alpha,
                self.high_amax,
                self.high_eps,
                self.high_gss,
                self.high_p_sat,
                self.groups_tensor,
                torch.tensor(self.local_joint_enabled, device=raw_coeffs_bhwc.device),
                torch.tensor(self.special_use_neighborhood, device=raw_coeffs_bhwc.device),
                torch.tensor({"replicate": 0, "reflect": 1, "zeros": 2}[self.padding_mode], device=raw_coeffs_bhwc.device),
            )
            # Collect stats with the native path (no grad) for consistent logging.
            _, stats_tensors = self._forward_native(raw_coeffs_bhwc.detach())
            if out is None:
                raise RuntimeError("custom_autograd NewBP produced None output")
            self._update_stats(raw_coeffs_bhwc.detach(), out.detach(), stats_tensors)
            return out

        out, stats_tensors = self._forward_native(raw_coeffs_bhwc)
        self._update_stats(raw_coeffs_bhwc, out, stats_tensors)
        return out
