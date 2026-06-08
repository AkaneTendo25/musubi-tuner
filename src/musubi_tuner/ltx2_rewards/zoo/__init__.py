"""Built-in reward plugins. Importing this package registers them.

Each plugin lazy-imports its heavy model deps inside ``setup()``, so importing this
package never requires optional reward dependencies. The real model code is vendored
under ``ltx2_rewards/vendor/`` — the zoo no longer depends on the external hpsv3
packages.

Reward routes: ``video`` (hpsv3, videoreward, videoscore2, anti_noise, iqa_quality),
``audio`` (clap, audiobox), ``sync`` (av_align, av_desync, imagebind). Audio/sync rewards become
usable once AV generation feeds ``audio_waveform``/``audio_file`` into the rollout samples.
"""

from __future__ import annotations

from . import anti_noise  # noqa: F401  (registers "anti_noise"; model-free speckle/flicker guardrail)
from . import audiobox  # noqa: F401  (registers "audiobox"; audio aesthetics)
from . import av_align  # noqa: F401  (registers "av_align"; algorithmic AV peak-IoU)
from . import av_desync  # noqa: F401  (registers "av_desync"; Synchformer AV sync)
from . import clap  # noqa: F401  (registers "clap"; audio-text similarity)
from . import hpsv3  # noqa: F401  (registers "hpsv3"; vendored Qwen2-VL preference)
from . import imagebind  # noqa: F401  (registers "imagebind"; multimodal similarity)
from . import iqa_quality  # noqa: F401  (registers "iqa_quality"; IQA-PyTorch perceptual quality/detail)
from . import videoreward  # noqa: F401  (registers "videoreward"; VideoAlign VQ/MQ/TA)
from . import videoscore2  # noqa: F401  (registers "videoscore2"; VQ/TA/PC physics head)

__all__ = [
    "anti_noise",
    "audiobox",
    "av_align",
    "av_desync",
    "clap",
    "hpsv3",
    "imagebind",
    "iqa_quality",
    "videoreward",
    "videoscore2",
]
