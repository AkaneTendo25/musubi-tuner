import argparse
import copy
from dataclasses import (
    asdict,
    dataclass,
    fields,
)
import functools
import os
import random
from textwrap import dedent, indent
import json
from datetime import datetime, timezone
from pathlib import Path

from typing import List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized

SharedEpoch = Optional["Synchronized[int]"]


import toml
import voluptuous
from voluptuous import Any, ExactSequence, MultipleInvalid, Object, Optional as VOptional, Schema

from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_LTX2,
    ARCHITECTURE_LTX2_FULL,
    AudioDataset,
    DatasetGroup,
    ImageDataset,
    VideoDataset,
)

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class BaseDatasetParams:
    resolution: Tuple[int, int] = (960, 544)
    enable_bucket: bool = False
    bucket_no_upscale: bool = False
    caption_extension: Optional[str] = None
    batch_size: int = 1
    num_repeats: int = 1
    video_loss_weight: Optional[float] = None
    audio_loss_weight: Optional[float] = None
    cache_directory: Optional[str] = None
    reference_cache_directory: Optional[str] = None
    reference_cache_directories: Optional[Sequence[str]] = None
    reference_audio_cache_directory: Optional[str] = None
    reference_audio_cache_directories: Optional[Sequence[str]] = None
    separate_audio_buckets: bool = False
    cache_only: bool = False
    debug_dataset: bool = False
    architecture: str = "no_default"  # short style like "hv" or "wan"


@dataclass
class ImageDatasetParams(BaseDatasetParams):
    image_directory: Optional[str] = None
    image_jsonl_file: Optional[str] = None
    control_directory: Optional[str] = None
    multiple_target: Optional[bool] = False

    # FramePack dependent parameters
    fp_latent_window_size: Optional[int] = 9
    fp_1f_clean_indices: Optional[Sequence[int]] = None
    fp_1f_target_index: Optional[int] = None
    fp_1f_no_post: Optional[bool] = False

    no_resize_control: Optional[bool] = False  # if True, control images are not resized to target resolution
    control_resolution: Optional[Tuple[int, int]] = None  # if set, control images are resized to this resolution


@dataclass
class VideoDatasetParams(BaseDatasetParams):
    video_directory: Optional[str] = None
    video_jsonl_file: Optional[str] = None
    control_directory: Optional[str] = None
    reference_directory: Optional[str] = None
    reference_directories: Optional[Sequence[str]] = None
    reference_audio_directory: Optional[str] = None
    reference_audio_directories: Optional[Sequence[str]] = None
    target_frames: Sequence[int] = (1,)
    frame_extraction: Optional[str] = "head"
    frame_stride: Optional[int] = 1
    frame_sample: Optional[int] = 1
    max_frames: Optional[int] = 129
    source_fps: Optional[float] = None
    target_fps: Optional[float] = None

    # FramePack dependent parameters
    fp_latent_window_size: Optional[int] = 9


@dataclass
class AudioDatasetParams(BaseDatasetParams):
    audio_directory: Optional[str] = None
    audio_jsonl_file: Optional[str] = None
    audio_bucket_strategy: str = "pad"  # "pad" (default) or "truncate"
    audio_bucket_interval: float = 2.0  # bucket step in seconds


@dataclass
class DatasetBlueprint:
    dataset_type: str  # "image", "video", "audio"
    params: Union[ImageDatasetParams, VideoDatasetParams, AudioDatasetParams]


@dataclass
class DatasetGroupBlueprint:
    datasets: Sequence[DatasetBlueprint]


@dataclass
class Blueprint:
    dataset_group: DatasetGroupBlueprint


class ConfigSanitizer:
    # @curry
    @staticmethod
    def __validate_and_convert_twodim(klass, value: Sequence) -> Tuple:
        Schema(ExactSequence([klass, klass]))(value)
        return tuple(value)

    # @curry
    @staticmethod
    def __validate_and_convert_scalar_or_twodim(klass, value: Union[float, Sequence]) -> Tuple:
        Schema(Any(klass, ExactSequence([klass, klass])))(value)
        try:
            Schema(klass)(value)
            return (value, value)
        except:
            return ConfigSanitizer.__validate_and_convert_twodim(klass, value)

    # datasets schema
    DATASET_ASCENDABLE_SCHEMA = {
        "caption_extension": str,
        "batch_size": int,
        "num_repeats": int,
        "resolution": functools.partial(__validate_and_convert_scalar_or_twodim.__func__, int),
        "enable_bucket": bool,
        "bucket_no_upscale": bool,
        "video_loss_weight": float,
        "audio_loss_weight": float,
        "cache_directory": str,
        "reference_cache_directory": str,
        "reference_cache_directories": [str],
        "reference_audio_cache_directory": str,
        "reference_audio_cache_directories": [str],
        "separate_audio_buckets": bool,
        "cache_only": bool,
    }
    IMAGE_DATASET_DISTINCT_SCHEMA = {
        "image_directory": str,
        "image_jsonl_file": str,
        "control_directory": str,
        "multiple_target": bool,
        "fp_latent_window_size": int,
        "fp_1f_clean_indices": [int],
        "fp_1f_target_index": int,
        "fp_1f_no_post": bool,
        "no_resize_control": bool,
        "control_resolution": functools.partial(__validate_and_convert_scalar_or_twodim.__func__, int),
    }
    AUDIO_DATASET_DISTINCT_SCHEMA = {
        "audio_directory": str,
        "audio_jsonl_file": str,
        "audio_bucket_strategy": str,
        "audio_bucket_interval": float,
    }
    VIDEO_DATASET_DISTINCT_SCHEMA = {
        "video_directory": str,
        "video_jsonl_file": str,
        "control_directory": str,
        "reference_directory": str,
        "reference_directories": [str],
        "reference_audio_directory": str,
        "reference_audio_directories": [str],
        "target_frames": [int],
        "frame_extraction": str,
        "frame_stride": int,
        "frame_sample": int,
        "max_frames": int,
        "source_fps": float,
        "target_fps": float,
        "fp_latent_window_size": int,
    }

    # options handled by argparse but not handled by user config
    ARGPARSE_SPECIFIC_SCHEMA = {
        "debug_dataset": bool,
    }

    def __init__(self) -> None:
        self.image_dataset_schema = self.__merge_dict(
            self.DATASET_ASCENDABLE_SCHEMA,
            self.IMAGE_DATASET_DISTINCT_SCHEMA,
        )
        self.audio_dataset_schema = self.__merge_dict(
            self.DATASET_ASCENDABLE_SCHEMA,
            self.AUDIO_DATASET_DISTINCT_SCHEMA,
        )
        self.video_dataset_schema = self.__merge_dict(
            self.DATASET_ASCENDABLE_SCHEMA,
            self.VIDEO_DATASET_DISTINCT_SCHEMA,
        )

        def validate_flex_dataset(dataset_config: dict):
            if "audio_directory" in dataset_config or "audio_jsonl_file" in dataset_config:
                return Schema(self.audio_dataset_schema)(dataset_config)
            if "video_directory" in dataset_config or "video_jsonl_file" in dataset_config:
                return Schema(self.video_dataset_schema)(dataset_config)
            else:
                return Schema(self.image_dataset_schema)(dataset_config)

        self.dataset_schema = validate_flex_dataset

        self.general_schema = self.__merge_dict(
            self.DATASET_ASCENDABLE_SCHEMA,
        )
        self.user_config_validator = Schema(
            {
                "general": self.general_schema,
                "datasets": [self.dataset_schema],
                VOptional("validation_datasets"): [self.dataset_schema],
            }
        )
        self.argparse_schema = self.__merge_dict(
            self.ARGPARSE_SPECIFIC_SCHEMA,
        )
        self.argparse_config_validator = Schema(Object(self.argparse_schema), extra=voluptuous.ALLOW_EXTRA)

    def sanitize_user_config(self, user_config: dict) -> dict:
        try:
            return self.user_config_validator(user_config)
        except MultipleInvalid:
            # TODO: clarify the error message
            logger.error("Invalid user config / ユーザ設定の形式が正しくないようです")
            raise

    # NOTE: In nature, argument parser result is not needed to be sanitize
    #   However this will help us to detect program bug
    def sanitize_argparse_namespace(self, argparse_namespace: argparse.Namespace) -> argparse.Namespace:
        try:
            return self.argparse_config_validator(argparse_namespace)
        except MultipleInvalid:
            # XXX: this should be a bug
            logger.error(
                "Invalid cmdline parsed arguments. This should be a bug. / コマンドラインのパース結果が正しくないようです。プログラムのバグの可能性が高いです。"
            )
            raise

    # NOTE: value would be overwritten by latter dict if there is already the same key
    @staticmethod
    def __merge_dict(*dict_list: dict) -> dict:
        merged = {}
        for schema in dict_list:
            # merged |= schema
            for k, v in schema.items():
                merged[k] = v
        return merged


class BlueprintGenerator:
    BLUEPRINT_PARAM_NAME_TO_CONFIG_OPTNAME = {}

    def __init__(self, sanitizer: ConfigSanitizer):
        self.sanitizer = sanitizer

    # runtime_params is for parameters which is only configurable on runtime, such as tokenizer
    def generate(self, user_config: dict, argparse_namespace: argparse.Namespace, **runtime_params) -> Blueprint:
        normalized_user_config = self._normalize_runtime_specific_user_config(user_config, runtime_params)
        sanitized_user_config = self.sanitizer.sanitize_user_config(normalized_user_config)
        sanitized_argparse_namespace = self.sanitizer.sanitize_argparse_namespace(argparse_namespace)

        argparse_config = {k: v for k, v in vars(sanitized_argparse_namespace).items() if v is not None}
        general_config = sanitized_user_config.get("general", {})

        dataset_blueprints = []
        for dataset_config in sanitized_user_config.get("datasets", []):
            is_audio_dataset = "audio_directory" in dataset_config or "audio_jsonl_file" in dataset_config
            is_image_dataset = "image_directory" in dataset_config or "image_jsonl_file" in dataset_config
            if is_audio_dataset:
                dataset_params_klass = AudioDatasetParams
                dataset_type = "audio"
            elif is_image_dataset:
                dataset_params_klass = ImageDatasetParams
                dataset_type = "image"
            else:
                dataset_params_klass = VideoDatasetParams
                dataset_type = "video"

            params = self.generate_params_by_fallbacks(
                dataset_params_klass, [dataset_config, general_config, argparse_config, runtime_params]
            )
            dataset_blueprints.append(DatasetBlueprint(dataset_type, params))

        dataset_group_blueprint = DatasetGroupBlueprint(dataset_blueprints)

        return Blueprint(dataset_group_blueprint)

    @staticmethod
    def _normalize_runtime_specific_user_config(user_config: dict, runtime_params: dict) -> dict:
        architecture = runtime_params.get("architecture")
        if architecture not in {ARCHITECTURE_LTX2, ARCHITECTURE_LTX2_FULL}:
            return user_config

        normalized_user_config = copy.deepcopy(user_config)

        for section_name in ("datasets", "validation_datasets"):
            dataset_entries = normalized_user_config.get(section_name)
            if not isinstance(dataset_entries, list):
                continue

            for i, dataset_config in enumerate(dataset_entries):
                if not isinstance(dataset_config, dict):
                    continue

                is_image_dataset = "image_directory" in dataset_config or "image_jsonl_file" in dataset_config
                if not is_image_dataset:
                    continue

                reference_directories = dataset_config.get("reference_directories")
                if reference_directories is not None and not isinstance(reference_directories, list):
                    raise ValueError(f"{section_name}[{i}] reference_directories must be a list of strings.")
                if reference_directories:
                    if len(reference_directories) != 1:
                        raise ValueError(
                            f"{section_name}[{i}] uses reference_directories on an image dataset, but image IC-LoRA "
                            "currently supports only one reference directory. Use a single entry."
                        )
                    dataset_config["reference_directory"] = reference_directories[0]
                reference_directory = dataset_config.get("reference_directory")
                control_directory = dataset_config.get("control_directory")
                has_reference_directory = reference_directory is not None
                has_control_directory = control_directory is not None
                reference_cache_directories = dataset_config.get("reference_cache_directories")
                if reference_cache_directories is not None and not isinstance(reference_cache_directories, list):
                    raise ValueError(f"{section_name}[{i}] reference_cache_directories must be a list of strings.")
                if reference_cache_directories:
                    if len(reference_cache_directories) != 1:
                        raise ValueError(
                            f"{section_name}[{i}] uses reference_cache_directories on an image dataset, but image IC-LoRA "
                            "currently supports only one reference cache directory. Use a single entry."
                        )
                    dataset_config["reference_cache_directory"] = reference_cache_directories[0]
                has_reference_cache = dataset_config.get("reference_cache_directory") is not None

                if has_reference_directory and not has_reference_cache:
                    raise ValueError(
                        f"{section_name}[{i}] uses reference_directory on an image dataset without "
                        "reference_cache_directory. For LTX image IC-LoRA datasets, set both "
                        "reference_directory and reference_cache_directory."
                    )

                if not has_reference_cache:
                    continue

                if has_reference_directory and has_control_directory:
                    raise ValueError(
                        f"{section_name}[{i}] sets both reference_directory and control_directory. "
                        "For LTX image IC-LoRA datasets, use reference_directory only."
                    )

                if has_control_directory:
                    raise ValueError(
                        f"{section_name}[{i}] uses control_directory, but LTX image IC-LoRA datasets must use "
                        "reference_directory when reference_cache_directory is set."
                    )

                if not has_reference_directory:
                    raise ValueError(
                        f"{section_name}[{i}] sets reference_cache_directory, but LTX image IC-LoRA datasets "
                        "also require reference_directory."
                    )

                dataset_config["control_directory"] = dataset_config.pop("reference_directory")

        return normalized_user_config

    @staticmethod
    def generate_params_by_fallbacks(param_klass, fallbacks: Sequence[dict]):
        name_map = BlueprintGenerator.BLUEPRINT_PARAM_NAME_TO_CONFIG_OPTNAME
        search_value = BlueprintGenerator.search_value
        default_params = asdict(param_klass())
        param_names = default_params.keys()

        params = {name: search_value(name_map.get(name, name), fallbacks, default_params.get(name)) for name in param_names}

        return param_klass(**params)

    @staticmethod
    def search_value(key: str, fallbacks: Sequence[dict], default_value=None):
        for cand in fallbacks:
            value = cand.get(key)
            if value is not None:
                return value

        return default_value


# if training is True, it will return a dataset group for training, otherwise for caching
def generate_dataset_group_by_blueprint(
    dataset_group_blueprint: DatasetGroupBlueprint,
    training: bool = False,
    num_timestep_buckets: Optional[int] = None,
    shared_epoch: SharedEpoch = None,
) -> DatasetGroup:
    datasets: List[Union[ImageDataset, VideoDataset, AudioDataset]] = []

    for dataset_blueprint in dataset_group_blueprint.datasets:
        if dataset_blueprint.dataset_type == "audio":
            dataset_klass = AudioDataset
        elif dataset_blueprint.dataset_type == "image":
            dataset_klass = ImageDataset
        else:
            dataset_klass = VideoDataset

        dataset = dataset_klass(**asdict(dataset_blueprint.params))
        datasets.append(dataset)

    # warn about missing data directories
    for i, dataset in enumerate(datasets):
        data_dir = getattr(dataset, "image_directory", None) or getattr(dataset, "video_directory", None) or getattr(dataset, "audio_directory", None)
        if data_dir is not None and not os.path.isdir(data_dir):
            logger.warning(
                "Dataset [%d]: data directory does not exist: %s — this dataset will produce zero items",
                i, data_dir,
            )

    # assertion
    cache_directories = [dataset.cache_directory for dataset in datasets]
    num_of_unique_cache_directories = len(set(cache_directories))
    if num_of_unique_cache_directories != len(cache_directories):
        raise ValueError(
            "cache directory should be unique for each dataset (note that cache directory is image/video directory if not specified)"
            + " / cache directory は各データセットごとに異なる必要があります（指定されていない場合はimage/video directoryが使われるので注意）"
        )

    # print info
    info = ""
    for i, dataset in enumerate(datasets):
        is_image_dataset = isinstance(dataset, ImageDataset)
        is_audio_dataset = isinstance(dataset, AudioDataset)
        info += dedent(
            f"""\
      [Dataset {i}]
        dataset_type: {"audio" if is_audio_dataset else "image" if is_image_dataset else "video"}
        resolution: {dataset.resolution}
        batch_size: {dataset.batch_size}
        num_repeats: {dataset.num_repeats}
        video_loss_weight: {getattr(dataset, "video_loss_weight", None)}
        audio_loss_weight: {getattr(dataset, "audio_loss_weight", None)}
        caption_extension: "{dataset.caption_extension}"
        enable_bucket: {dataset.enable_bucket}
        bucket_no_upscale: {dataset.bucket_no_upscale}
        separate_audio_buckets: {getattr(dataset, "separate_audio_buckets", False)}
        cache_only: {getattr(dataset, "cache_only", False)}
        cache_directory: "{dataset.cache_directory}"
        reference_cache_directory: "{getattr(dataset, 'reference_cache_directory', None)}"
        reference_cache_directories: {getattr(dataset, "reference_cache_directories", None)}
        debug_dataset: {dataset.debug_dataset}
    """
        )

        if is_audio_dataset:
            info += indent(
                dedent(
                    f"""\
        audio_directory: "{dataset.audio_directory}"
        audio_jsonl_file: "{dataset.audio_jsonl_file}"
        audio_bucket_strategy: {getattr(dataset, "audio_bucket_strategy", "pad")}
        audio_bucket_interval: {getattr(dataset, "audio_bucket_interval", 2.0)}
    \n"""
                ),
                "    ",
            )
        elif is_image_dataset:
            info += indent(
                dedent(
                    f"""\
        image_directory: "{dataset.image_directory}"
        image_jsonl_file: "{dataset.image_jsonl_file}"
        control_directory: "{dataset.control_directory}"
        multiple_target: {dataset.multiple_target}
        fp_latent_window_size: {dataset.fp_latent_window_size}
        fp_1f_clean_indices: {dataset.fp_1f_clean_indices}
        fp_1f_target_index: {dataset.fp_1f_target_index}
        fp_1f_no_post: {dataset.fp_1f_no_post}
        no_resize_control: {dataset.no_resize_control}
        control_resolution: {dataset.control_resolution}
    \n"""
                ),
                "    ",
            )
        else:
            info += indent(
                dedent(
                    f"""\
        video_directory: "{dataset.video_directory}"
        video_jsonl_file: "{dataset.video_jsonl_file}"
        control_directory: "{dataset.control_directory}"
        reference_directory: "{getattr(dataset, 'reference_directory', None)}"
        reference_directories: {getattr(dataset, "reference_directories", None)}
        reference_audio_directory: "{getattr(dataset, 'reference_audio_directory', None)}"
        reference_audio_directories: {getattr(dataset, "reference_audio_directories", None)}
        reference_audio_cache_directory: "{getattr(dataset, 'reference_audio_cache_directory', None)}"
        reference_audio_cache_directories: {getattr(dataset, "reference_audio_cache_directories", None)}
        target_frames: {dataset.target_frames}
        frame_extraction: {dataset.frame_extraction}
        frame_stride: {dataset.frame_stride}
        frame_sample: {dataset.frame_sample}
        max_frames: {dataset.max_frames}
        source_fps: {dataset.source_fps}
        target_fps: {getattr(dataset, "target_fps", None)}
        fp_latent_window_size: {dataset.fp_latent_window_size}
    \n"""
                ),
                "    ",
            )
    logger.info(f"{info}")

    # make buckets first because it determines the length of dataset
    # and set the same seed for all datasets
    seed = random.randint(0, 2**31)  # actual seed is seed + epoch_no
    for i, dataset in enumerate(datasets):
        # logger.info(f"[Dataset {i}]")
        dataset.set_seed(seed, shared_epoch)
        if training:
            dataset.prepare_for_training(num_timestep_buckets=num_timestep_buckets)

    return DatasetGroup(datasets)


def _manifest_params_with_cache_only(dataset_type: str, params: dict) -> dict:
    params = dict(params)

    if not params.get("cache_directory"):
        if dataset_type == "audio":
            params["cache_directory"] = params.get("audio_directory")
        elif dataset_type == "image":
            params["cache_directory"] = params.get("image_directory")
        else:
            params["cache_directory"] = params.get("video_directory")

    if not params.get("cache_directory"):
        raise ValueError(
            f"cache_directory is required to create a cache-only manifest for {dataset_type} datasets. "
            "Set cache_directory in dataset config."
        )

    params["cache_only"] = True

    # Strip source references to guarantee source-free training from manifest.
    if dataset_type == "audio":
        params["audio_directory"] = None
        params["audio_jsonl_file"] = None
    elif dataset_type == "image":
        params["image_directory"] = None
        params["image_jsonl_file"] = None
        params["control_directory"] = None
        params["multiple_target"] = False
    else:
        params["video_directory"] = None
        params["video_jsonl_file"] = None
        params["control_directory"] = None
        params["reference_directory"] = None
        params["reference_directories"] = None
        params["reference_audio_directory"] = None
        params["reference_audio_directories"] = None

    return params


def _blueprint_to_manifest_entries(dataset_group_blueprint: DatasetGroupBlueprint) -> list[dict]:
    entries: list[dict] = []
    for dataset_blueprint in dataset_group_blueprint.datasets:
        params = _manifest_params_with_cache_only(dataset_blueprint.dataset_type, asdict(dataset_blueprint.params))
        entries.append(
            {
                "dataset_type": dataset_blueprint.dataset_type,
                "params": params,
            }
        )
    return entries


def create_cache_only_dataset_manifest(
    user_config: dict,
    argparse_namespace: argparse.Namespace,
    architecture: str,
    source_dataset_config: Optional[Union[str, Path]] = None,
) -> dict:
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    blueprint = blueprint_generator.generate(user_config, argparse_namespace, architecture=architecture)

    manifest: dict = {
        "format": "musubi_tuner_dataset_manifest",
        "version": 1,
        "architecture": architecture,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": _blueprint_to_manifest_entries(blueprint.dataset_group),
    }

    if source_dataset_config is not None:
        manifest["source_dataset_config"] = str(source_dataset_config)

    if user_config.get("validation_datasets"):
        validation_user_config = {
            "general": user_config.get("general", {}),
            "datasets": user_config.get("validation_datasets", []),
        }
        validation_blueprint = blueprint_generator.generate(
            validation_user_config,
            argparse_namespace,
            architecture=architecture,
        )
        manifest["validation_datasets"] = _blueprint_to_manifest_entries(validation_blueprint.dataset_group)

    return manifest


def save_dataset_manifest(manifest: dict, manifest_path: Union[str, Path]) -> Path:
    path = Path(manifest_path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path


def load_dataset_manifest(manifest_path: Union[str, Path]) -> dict:
    path = Path(manifest_path)
    if not path.is_file():
        raise ValueError(f"dataset manifest not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        raise ValueError(f"failed to load dataset manifest: {path}") from e

    if not isinstance(manifest, dict):
        raise ValueError(f"invalid dataset manifest format: {path}")
    if manifest.get("version") != 1:
        raise ValueError(f"unsupported dataset manifest version: {manifest.get('version')}")
    if not isinstance(manifest.get("datasets"), list):
        raise ValueError(f"dataset manifest must contain a datasets array: {path}")

    return manifest


def _normalize_manifest_params(params: dict) -> dict:
    normalized = dict(params)
    if isinstance(normalized.get("resolution"), list):
        normalized["resolution"] = tuple(normalized["resolution"])
    if isinstance(normalized.get("control_resolution"), list):
        normalized["control_resolution"] = tuple(normalized["control_resolution"])
    if isinstance(normalized.get("target_frames"), list):
        normalized["target_frames"] = tuple(normalized["target_frames"])
    return normalized


def _manifest_entries_to_blueprint(entries: Sequence[dict], default_architecture: Optional[str] = None) -> DatasetGroupBlueprint:
    dataset_blueprints: list[DatasetBlueprint] = []
    for i, entry in enumerate(entries):
        dataset_type = entry.get("dataset_type")
        params = entry.get("params")
        if dataset_type not in {"audio", "image", "video"}:
            raise ValueError(f"invalid dataset_type in manifest entry {i}: {dataset_type}")
        if not isinstance(params, dict):
            raise ValueError(f"manifest entry {i} has invalid params")

        if dataset_type == "audio":
            dataset_params_klass = AudioDatasetParams
        elif dataset_type == "image":
            dataset_params_klass = ImageDatasetParams
        else:
            dataset_params_klass = VideoDatasetParams

        normalized_params = _normalize_manifest_params(params)
        normalized_params["cache_only"] = True
        if default_architecture and normalized_params.get("architecture") in {None, "no_default"}:
            normalized_params["architecture"] = default_architecture

        valid_fields = {f.name for f in fields(dataset_params_klass)}
        filtered_params = {k: v for k, v in normalized_params.items() if k in valid_fields}
        dataset_params = dataset_params_klass(**filtered_params)
        dataset_blueprints.append(DatasetBlueprint(dataset_type, dataset_params))

    return DatasetGroupBlueprint(dataset_blueprints)


def generate_dataset_group_by_manifest(
    manifest: dict,
    split: str = "train",
    training: bool = False,
    num_timestep_buckets: Optional[int] = None,
    shared_epoch: SharedEpoch = None,
) -> Optional[DatasetGroup]:
    if split not in {"train", "validation"}:
        raise ValueError(f"invalid manifest split: {split}")

    key = "datasets" if split == "train" else "validation_datasets"
    entries = manifest.get(key, [])
    if not entries:
        return None

    default_architecture = manifest.get("architecture")
    dataset_group_blueprint = _manifest_entries_to_blueprint(entries, default_architecture=default_architecture)
    return generate_dataset_group_by_blueprint(
        dataset_group_blueprint,
        training=training,
        num_timestep_buckets=num_timestep_buckets,
        shared_epoch=shared_epoch,
    )


def load_user_config(file: str) -> dict:
    file: Path = Path(file)
    if not file.is_file():
        raise ValueError(f"file not found / ファイルが見つかりません: {file}")

    if file.name.lower().endswith(".json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            logger.error(
                f"Error on parsing JSON config file. Please check the format. / JSON 形式の設定ファイルの読み込みに失敗しました。文法が正しいか確認してください。: {file}"
            )
            raise
    elif file.name.lower().endswith(".toml"):
        try:
            config = toml.load(file)
        except Exception:
            logger.error(
                f"Error on parsing TOML config file. Please check the format. / TOML 形式の設定ファイルの読み込みに失敗しました。文法が正しいか確認してください。: {file}"
            )
            raise
    else:
        raise ValueError(f"not supported config file format / 対応していない設定ファイルの形式です: {file}")

    deprecated_key_map = {
        "flux_kontext_no_resize_control": "no_resize_control",
        "qwen_image_edit_no_resize_control": "no_resize_control",
        "qwen_image_edit_control_resolution": "control_resolution",
    }

    def normalize_deprecated_keys(section: dict, section_name: str) -> None:
        for old_key, new_key in deprecated_key_map.items():
            if old_key not in section:
                continue
            if new_key in section:
                logger.warning(
                    f"Deprecated config key '{old_key}' is ignored because '{new_key}' is already set in {section_name}."
                )
            else:
                section[new_key] = section[old_key]
                logger.warning(f"Deprecated config key '{old_key}' found in {section_name}; use '{new_key}' instead.")
            del section[old_key]

    general_config = config.get("general")
    if isinstance(general_config, dict):
        normalize_deprecated_keys(general_config, "general")

    datasets_config = config.get("datasets", [])
    if isinstance(datasets_config, list):
        for idx, dataset_config in enumerate(datasets_config):
            if isinstance(dataset_config, dict):
                normalize_deprecated_keys(dataset_config, f"datasets[{idx}]")

    return config


# for config test
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_config")
    config_args, remain = parser.parse_known_args()

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug_dataset", action="store_true")
    argparse_namespace = parser.parse_args(remain)

    logger.info("[argparse_namespace]")
    logger.info(f"{vars(argparse_namespace)}")

    user_config = load_user_config(config_args.dataset_config)

    logger.info("")
    logger.info("[user_config]")
    logger.info(f"{user_config}")

    sanitizer = ConfigSanitizer()
    sanitized_user_config = sanitizer.sanitize_user_config(user_config)

    logger.info("")
    logger.info("[sanitized_user_config]")
    logger.info(f"{sanitized_user_config}")

    blueprint = BlueprintGenerator(sanitizer).generate(user_config, argparse_namespace)

    logger.info("")
    logger.info("[blueprint]")
    logger.info(f"{blueprint}")

    dataset_group = generate_dataset_group_by_blueprint(blueprint.dataset_group)
