import math

import torch
from safetensors.torch import load_file as safetensors_load_file

from .cached_enc_dec import CachedEncoder3D, CachedDecoder3D
from .ctx.enc_dec import Encoder3D, Decoder3D
from dataclasses import dataclass
from .regularizers import DiagonalGaussianRegularizer


@dataclass
class DecoderOutput:
    sample: torch.Tensor


class CausalVAE(torch.nn.Module):
    def __init__(self, encoder_conf, decoder_conf, scaling_factor=None, mean=None, ckpt_path=None):
        super().__init__()
        self.conf = {"enc": encoder_conf, "dec": decoder_conf}
        self.encoder = Encoder3D(**encoder_conf)
        self.decoder = Decoder3D(**decoder_conf)
        self.regularizer = DiagonalGaussianRegularizer(sample=False)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, path):
        sd = torch.load(path, map_location="cpu")["state_dict"]

        # Fix checkpoint if starting from new style
        replace_keys = dict()
        delete_keys = list()
        for k in sd:
            if k.startswith("loss."):
                delete_keys.append(k)
                continue

        for old_k, new_k in replace_keys.items():
            sd[new_k] = sd[old_k]
            del sd[old_k]
        for k in delete_keys:
            del sd[k]

        self.load_state_dict(sd, strict=True)

    @staticmethod
    def normalize_data(data):
        return data / 128 - 1.0

    @staticmethod
    def denormalize_data(data):
        return (data + 1) * 128

    def spacial_downsample_ratio(self):
        return 2 ** (self.encoder.num_resolutions - 1)

    def temporal_downsample_ratio(self):
        return self.conf["enc"]["temporal_compress_times"]

    def encode(self, x):

        latent_and_params = self.encoder(x)
        latent = self.regularizer.get_sample(latent_and_params)

        return latent, None

    def decode(self, z):

        out = self.decoder(z)

        return DecoderOutput(sample=out)

    def forward(self, x):
        latent = self.encode(x)
        recs = self.decode(latent)
        return recs


class CachedCausalVAE(CausalVAE):
    def __init__(self, encoder_conf, decoder_conf, scaling_factor=None, mean=None, ckpt_path=None):
        super(CausalVAE, self).__init__()
        self.conf = {"enc": encoder_conf, "dec": decoder_conf}
        self.encoder = CachedEncoder3D(**encoder_conf)
        self.decoder = CachedDecoder3D(**decoder_conf)
        self.regularizer = DiagonalGaussianRegularizer(sample=False)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, path):
        """Load a KVAE checkpoint in safetensors or training-checkpoint format."""
        if str(path).endswith(".safetensors"):
            # Release checkpoints are flat safetensors state dicts already in
            # the new-style key naming.
            self.load_state_dict(safetensors_load_file(str(path)), strict=True)
            return
        sd = torch.load(path, map_location="cpu")["state_dict"]

        # Fix checkpoint if starting from new style
        replace_keys = dict()
        delete_keys = list()
        for k in sd:
            if k.startswith("loss."):
                delete_keys.append(k)
                continue
            if "encoder.down" in k and "downsample.temporal_conv.conv" in k:
                continue

            if k.startswith("decoder"):
                if "upsample" in k:
                    continue
                elif ".conv_b.conv" in k:
                    replace_keys[k] = k.replace(".conv_b.conv", ".conv_b")
                    continue
                elif ".conv_y.conv" in k:
                    replace_keys[k] = k.replace(".conv_y.conv", ".conv_y")
                    continue

            if k.endswith("sample.temporal_conv.conv.weight") or k.endswith("sample.temporal_conv.conv.bias"):
                replace_keys[k] = k.replace(".temporal_conv.conv.", ".temporal_conv.")
        for old_k, new_k in replace_keys.items():
            sd[new_k] = sd[old_k]
            del sd[old_k]
        for k in delete_keys:
            del sd[k]

        self.load_state_dict(sd, strict=True)

    def make_empty_cache(self, block: str):
        """Create empty causal-convolution and normalization caches."""

        def make_dict(name, p=None):
            if name == "conv":
                return {"padding": None}

            layer, module = name.split("_")
            if layer == "norm":
                if module == "enc":
                    return {"mean": None, "var": None}
                else:
                    return {"norm": make_dict("norm_enc"), "add_conv": make_dict("conv")}
            elif layer == "resblock":
                return {
                    "norm1": make_dict(f"norm_{module}"),
                    "norm2": make_dict(f"norm_{module}"),
                    "conv1": make_dict("conv"),
                    "conv2": make_dict("conv"),
                    "conv_shortcut": make_dict("conv"),
                }
            elif layer.isdigit():
                out_dict = {"down": [make_dict("conv"), make_dict("conv")], "up": make_dict("conv")}
                for i in range(p):
                    out_dict[i] = make_dict(f"resblock_{module}")

                return out_dict

        cache = {
            "conv_in": make_dict("conv"),
            "mid_1": make_dict(f"resblock_{block}"),
            "mid_2": make_dict(f"resblock_{block}"),
            "norm_out": make_dict(f"norm_{block}"),
            "conv_out": make_dict("conv"),
        }
        for i in range(len(self.conf[block].get("ch_mult", [1, 2, 4, 8]))):
            cache[i] = make_dict(f"{i}_block", p=self.conf[block]["num_res_blocks"] + 1)
        return cache

    def encode(self, x, seg_len=16):
        """Encode a video in temporal segments and return latents and segment sizes.

        Args:
            x (`torch.Tensor`): Video tensor in ``(batch, channels, frames, height, width)`` format.
            seg_len (`int`, *optional*, defaults to 16): Number of non-initial
                frames processed in each segment.

        Returns:
            `tuple[torch.Tensor, list[int]]`: Encoded latents and the pixel-space
            segment sizes needed by :meth:`decode`.
        """
        cache = self.make_empty_cache("enc")

        # Compute segment sizes.
        split_list = [seg_len + 1]
        n_frames = x.size(2) - (seg_len + 1)
        while n_frames > 0:
            split_list.append(seg_len)
            n_frames -= seg_len

        split_list[-1] += n_frames

        # Encode each segment.
        latent = []
        for chunk in torch.split(x, split_list, dim=2):
            l = self.encoder(chunk, cache)
            sample = self.regularizer.get_sample(l)
            latent.append(sample)

        latent = torch.cat(latent, dim=2)
        return latent, split_list

    def decode(self, z, split_list=None):
        """Decode latent segments while reusing the causal decoder cache.

        Args:
            z (`torch.Tensor`): Latent tensor in ``(batch, channels, frames, height, width)`` format.
            split_list (`list[int]`, *optional*): Pixel-space segment sizes
                returned by :meth:`encode`.

        Returns:
            `DecoderOutput`: Decoded video in ``sample``.
        """
        cache = self.make_empty_cache("dec")

        # Compute latent segment sizes.
        if split_list is None:
            default_split_size = 16 // self.conf["enc"]["temporal_compress_times"]
            time_dim = z.shape[2]
            if time_dim == 1:
                # image
                split_list = [1]
            else:
                splits_num = (time_dim - 1) // default_split_size
                split_list = [default_split_size] * splits_num
                if (time_dim - 1) % default_split_size != 0:
                    split_list.append((time_dim - 1) % default_split_size)
                split_list[0] += 1
        else:
            split_list = [math.ceil(size / self.conf["enc"]["temporal_compress_times"]) for size in split_list]

        # Decode each segment.
        recs = []
        for chunk in torch.split(z, split_list, dim=2):
            out = self.decoder(chunk, cache)
            recs.append(out)

        recs = torch.cat(recs, dim=2)
        return DecoderOutput(sample=recs)

    def forward(self, x, seg_len: int = 16):
        """Encode and decode a video in one call.

        Args:
            x (`torch.Tensor`): Video tensor in ``(batch, channels, frames, height, width)`` format.
            seg_len (`int`, *optional*, defaults to 16): Number of non-initial
                frames processed in each segment.

        Returns:
            `DecoderOutput`: Reconstructed video in ``sample``.
        """
        latent, split_list = self.encode(x, seg_len)
        recs = self.decode(latent, split_list)
        return recs
