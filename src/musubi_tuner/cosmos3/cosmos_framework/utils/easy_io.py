# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Small easy_io subset for local Cosmos tokenizer loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


class _EasyIO:
    def load(self, path: str, backend_args: dict[str, Any] | None = None, map_location: str | torch.device | None = None):
        if backend_args is not None:
            raise ValueError("Object-store easy_io backends are not supported for local Cosmos loading.")

        suffix = Path(path).suffix.lower()
        if suffix == ".json":
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return torch.load(path, map_location=map_location)


easy_io = _EasyIO()
