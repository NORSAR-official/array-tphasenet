#!/usr/bin/env python3
"""Preflight checks for Array-TPhaseNet workflows.

This script validates configuration consistency and required files before
running training/prediction/evaluation scripts.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional

try:
    from omegaconf import OmegaConf
except ModuleNotFoundError:
    print(
        "[FAIL] Missing dependency 'omegaconf'. Install the project environment first "
        "(`conda env create -f environment.yml`).",
        file=sys.stderr,
    )
    sys.exit(2)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FALSE_LIKE = {"", "0", "false", "none", "null", "no", "off"}


class Reporter:
    def __init__(self, strict: bool = False) -> None:
        self.strict = strict
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def info(self, msg: str) -> None:
        print(f"[INFO] {msg}")

    def ok(self, msg: str) -> None:
        print(f"[ OK ] {msg}")

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"[WARN] {msg}")

    def error(self, msg: str) -> None:
        self.errors.append(msg)
        print(f"[FAIL] {msg}")

    def finalize(self) -> int:
        if self.errors:
            print(f"\nPreflight failed with {len(self.errors)} error(s).")
            return 1
        if self.strict and self.warnings:
            print(f"\nPreflight failed in strict mode ({len(self.warnings)} warning(s)).")
            return 1
        print(
            f"\nPreflight passed with {len(self.warnings)} warning(s)."
            if self.warnings
            else "\nPreflight passed."
        )
        return 0


def select(cfg: Any, key: str, default: Any = None) -> Any:
    value = OmegaConf.select(cfg, key, default=default)
    return value


def to_list(value: Any) -> List[Any]:
    if value is None:
        return []

    # OmegaConf returns ListConfig for YAML list values. Treat those the same as
    # regular Python lists/tuples.
    if OmegaConf.is_list(value) or isinstance(value, (list, tuple)):
        raw_items = list(value)
    else:
        raw_items = [value]

    normalized: List[Any] = []
    for item in raw_items:
        if OmegaConf.is_list(item) or isinstance(item, (list, tuple)):
            normalized.extend(list(item))
            continue

        # Handle stringified list input such as "[2022]" from ad-hoc overrides.
        if isinstance(item, str):
            s = item.strip()
            if s.startswith("[") and s.endswith("]"):
                inner = s[1:-1].strip()
                if not inner:
                    continue
                parts = [part.strip().strip("'\"") for part in inner.split(",") if part.strip()]
                normalized.extend(parts)
                continue

        normalized.append(item)

    return normalized


def is_false_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return not value
    if isinstance(value, str):
        return value.strip().lower() in FALSE_LIKE
    return False


def optional_str(value: Any) -> Optional[str]:
    if is_false_like(value):
        return None
    return str(value)


def as_abs_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def as_abs_pattern(pattern_value: str) -> str:
    if os.path.isabs(pattern_value):
        return pattern_value
    return str(PROJECT_ROOT / pattern_value)


def has_match(pattern: str) -> bool:
    return len(glob.glob(pattern)) > 0


def check_required_keys(cfg: Any, report: Reporter, keys: Iterable[str]) -> None:
    for key in keys:
        if select(cfg, key, None) is None:
            report.error(f"Missing required config key: '{key}'")


def check_common_config(cfg: Any, report: Reporter) -> None:
    check_required_keys(
        cfg,
        report,
        [
            "run.outputdir",
            "data.inputdir",
            "data.input_dataset_name",
            "data.input_datatype",
            "data.test_years",
            "model.type",
            "prediction.output_dir",
            "prediction.stacking",
        ],
    )

    years = to_list(select(cfg, "data.test_years", []))
    if not years:
        report.error("data.test_years is empty.")
    else:
        report.ok(f"Configured test years: {years}")

    outdir = as_abs_path(str(select(cfg, "run.outputdir")))
    if outdir.exists():
        report.ok(f"run.outputdir exists: {outdir}")
    else:
        report.warn(f"run.outputdir does not exist yet: {outdir}")

    pred_outdir = as_abs_path(str(select(cfg, "prediction.output_dir")))
    if pred_outdir.exists():
        report.ok(f"prediction.output_dir exists: {pred_outdir}")
    else:
        report.warn(f"prediction.output_dir does not exist yet: {pred_outdir}")

    picks = select(cfg, "evaluation.picks", None)
    if isinstance(picks, str) and picks.endswith(".data"):
        dat_alt = picks[:-5] + ".dat"
        dat_alt_abs = as_abs_path(dat_alt)
        if dat_alt_abs.exists():
            report.warn(
                "evaluation.picks points to '.data' while '.dat' exists. "
                f"Did you mean '{dat_alt}'?"
            )


def dataset_files_exist(cfg: Any, years: List[Any], report: Reporter, label: str) -> None:
    input_dir_raw = str(select(cfg, "data.inputdir"))
    if not input_dir_raw.endswith("/"):
        input_dir_raw += "/"

    dataset = str(select(cfg, "data.input_dataset_name"))
    datatype = str(select(cfg, "data.input_datatype"))
    noise_enabled = not is_false_like(select(cfg, "data.noise_waveforms", False))

    for year in years:
        y = str(year)
        data_pattern = as_abs_pattern(f"{input_dir_raw}{dataset}_{y}_{datatype}.hdf5")
        label_pattern_exact = as_abs_pattern(
            f"{input_dir_raw}{dataset}_{y}_labels_phase_detection.hdf5"
        )
        label_pattern_year = as_abs_pattern(
            f"{input_dir_raw}{dataset}_{y[:4]}_labels_phase_detection.hdf5"
        )

        if has_match(data_pattern):
            report.ok(f"{label}: found data for year '{y}'")
        else:
            report.error(
                f"{label}: missing data file for year '{y}'. Expected pattern: {data_pattern}"
            )

        if has_match(label_pattern_exact) or has_match(label_pattern_year):
            report.ok(f"{label}: found labels for year '{y}'")
        else:
            report.error(
                f"{label}: missing label file for year '{y}'. Expected one of: "
                f"{label_pattern_exact} or {label_pattern_year}"
            )

        if noise_enabled:
            noise_pattern = as_abs_pattern(f"{input_dir_raw}{dataset}_{y}_{datatype}_noise.hdf5")
            if not has_match(noise_pattern):
                report.error(
                    f"{label}: data.noise_waveforms is enabled but no noise file matched: "
                    f"{noise_pattern}"
                )


def check_metadata_files(cfg: Any, report: Reporter) -> None:
    years = to_list(select(cfg, "data.test_years", []))
    dataset = str(select(cfg, "data.input_dataset_name"))
    input_dir_raw = str(select(cfg, "data.inputdir"))
    if not input_dir_raw.endswith("/"):
        input_dir_raw += "/"

    needs_metadata = (
        not is_false_like(select(cfg, "evaluation.vs_metadata", False))
        or not is_false_like(select(cfg, "evaluation.theoretical_arrivals", False))
        or not is_false_like(select(cfg, "evaluation.snr_threshold", False))
        or str(select(cfg, "prediction.combine_array_stations", "")).strip().lower() == "beam"
    )

    if not needs_metadata:
        return

    for year in years:
        y4 = str(year)[:4]
        arrivals_pattern = as_abs_pattern(f"{input_dir_raw}{dataset}_{y4}_arrivals*.csv")
        metadata_file = as_abs_pattern(f"{input_dir_raw}metadata_{y4}.csv")
        if has_match(arrivals_pattern):
            report.ok(f"Found arrivals metadata for year {y4}")
        else:
            report.error(f"Missing arrivals metadata. Expected pattern: {arrivals_pattern}")
        if has_match(metadata_file):
            report.ok(f"Found event metadata file for year {y4}")
        else:
            report.error(f"Missing event metadata file: {metadata_file}")


def training_setting(cfg: Any) -> str:
    dataset = str(select(cfg, "data.input_dataset_name"))
    model_type = str(select(cfg, "model.type"))
    setting = f"{dataset}_{model_type}"

    if not is_false_like(select(cfg, "data.holdout", False)):
        setting = f"{dataset}_holdout_{model_type}"
    if str(select(cfg, "normalization.channel_mode", "")).strip().lower() == "local":
        setting = f"{dataset}_localnorm_{model_type}"
    if not is_false_like(select(cfg, "data.extract_array_channels", False)):
        setting = f"array{select(cfg, 'data.setname')}_{model_type}"

    custom_outname = optional_str(select(cfg, "run.custom_outname", False))
    if custom_outname:
        setting = f"{dataset}_{custom_outname}_{model_type}"
    if custom_outname and not is_false_like(select(cfg, "data.extract_array_channels", False)):
        setting = f"array{select(cfg, 'data.setname')}_{custom_outname}_{model_type}"
    return setting


def predict_test_model_setting(cfg: Any) -> str:
    model_type = str(select(cfg, "model.type"))
    model_override = optional_str(select(cfg, "run.predict_with_other_model", False))
    dataset = model_override or str(select(cfg, "data.input_dataset_name"))
    setting = f"{dataset}_{model_type}"
    if not is_false_like(select(cfg, "data.extract_array_channels", False)):
        setting = f"array{select(cfg, 'data.setname')}_{model_type}"
    return setting


def predict_test_output_setting(cfg: Any) -> str:
    model_type = str(select(cfg, "model.type"))
    model_override = optional_str(select(cfg, "run.predict_with_other_model", False))
    dataset = model_override or str(select(cfg, "data.input_dataset_name"))
    setting = f"{dataset}_{model_type}"
    if not is_false_like(select(cfg, "data.extract_array_channels", False)):
        setting = f"array{select(cfg, 'data.setname')}_{model_type}"

    custom_outname = optional_str(select(cfg, "run.custom_outname", False))
    if custom_outname:
        setting = f"{dataset}_{custom_outname}_{model_type}"
    if model_override:
        setting += f"_{model_override}"
    return setting


def eval_test_setting(cfg: Any) -> str:
    dataset = str(select(cfg, "data.input_dataset_name"))
    model_type = str(select(cfg, "model.type"))
    setting = f"{dataset}_{model_type}"
    if not is_false_like(select(cfg, "data.extract_array_channels", False)):
        setting = f"array{select(cfg, 'data.setname')}_{model_type}"
    if not is_false_like(select(cfg, "data.holdout", False)):
        setting = f"{dataset}_holdout2_{model_type}"
    if str(select(cfg, "normalization.channel_mode", "")).strip().lower() == "local":
        setting = f"{dataset}_localnorm_{model_type}"

    custom_outname = optional_str(select(cfg, "run.custom_outname", False))
    if custom_outname:
        setting = f"{dataset}_{custom_outname}_{model_type}"
    if custom_outname and not is_false_like(select(cfg, "data.extract_array_channels", False)):
        setting = f"array{select(cfg, 'data.setname')}_{custom_outname}_{model_type}"

    model_override = optional_str(select(cfg, "run.predict_with_other_model", False))
    if model_override:
        setting += f"_{model_override}"

    years = to_list(select(cfg, "data.test_years", []))
    if len(years) > 1:
        setting += f"_{years[0]}"
    return setting


def continuous_model_setting(cfg: Any) -> str:
    dataset = str(select(cfg, "data.input_dataset_name"))
    model_type = str(select(cfg, "model.type"))
    if not is_false_like(select(cfg, "data.extract_array_channels", False)):
        setting = f"array{select(cfg, 'data.setname')}_{model_type}"
    else:
        setting = f"{dataset}_{model_type}"

    custom_outname = optional_str(select(cfg, "run.custom_outname", False))
    if custom_outname:
        setting = f"{dataset}_{custom_outname}_{model_type}"
    if custom_outname and not is_false_like(select(cfg, "data.extract_array_channels", False)):
        setting = f"array{select(cfg, 'data.setname')}_{custom_outname}_{model_type}"
    return setting


def check_stage_train(cfg: Any, report: Reporter) -> None:
    train_years = to_list(select(cfg, "data.train_years", []))
    valid_years = to_list(select(cfg, "data.valid_years", []))
    test_years = to_list(select(cfg, "data.test_years", []))
    if not train_years:
        report.error("data.train_years is empty but required for stage 'train'.")
    if not valid_years:
        report.error("data.valid_years is empty but required for stage 'train'.")
    dataset_files_exist(cfg, train_years, report, "train")
    dataset_files_exist(cfg, valid_years, report, "valid")
    dataset_files_exist(cfg, test_years, report, "test")


def check_stage_predict_test(cfg: Any, report: Reporter) -> None:
    dataset_files_exist(cfg, to_list(select(cfg, "data.test_years", [])), report, "test")
    output_dir = as_abs_path(str(select(cfg, "run.outputdir")))
    model_setting = predict_test_model_setting(cfg)
    model_path = output_dir / "models" / f"saved_model_{model_setting}.tf"
    if model_path.exists():
        report.ok(f"Found model for test prediction: {model_path}")
    else:
        report.error(f"Missing model for test prediction: {model_path}")


def check_stage_evaluate_test(cfg: Any, report: Reporter) -> None:
    check_metadata_files(cfg, report)
    output_dir = as_abs_path(str(select(cfg, "run.outputdir")))
    eval_setting = eval_test_setting(cfg)
    pred_file = output_dir / "predictions" / f"predictions_{eval_setting}.npz"
    if pred_file.exists():
        report.ok(f"Found predictions for evaluation: {pred_file}")
        return

    candidates = [
        output_dir / "predictions" / f"predictions_{training_setting(cfg)}.npz",
        output_dir / "predictions" / f"predictions_{predict_test_output_setting(cfg)}.npz",
    ]
    existing_candidates = [str(p) for p in candidates if p.exists()]
    if existing_candidates:
        report.error(
            "evaluate_on_testdata.py will look for a different prediction file name. "
            f"Expected: {pred_file}. Existing candidate(s): {existing_candidates}"
        )
    else:
        report.error(f"Missing predictions for evaluation: {pred_file}")


def check_stage_predict_continuous(cfg: Any, report: Reporter) -> None:
    output_dir = as_abs_path(str(select(cfg, "run.outputdir")))
    model_setting = continuous_model_setting(cfg)
    model_path = output_dir / "models" / f"saved_model_{model_setting}.tf"
    if model_path.exists():
        report.ok(f"Found model for continuous prediction: {model_path}")
    else:
        report.error(f"Missing model for continuous prediction: {model_path}")

    report.warn(
        "Continuous prediction depends on external waveform access via the "
        "UIB-NORSAR FDSN service."
    )


def check_stage_evaluate_continuous(cfg: Any, report: Reporter) -> None:
    picks_file = select(cfg, "evaluation.picks", None)
    if picks_file is None:
        report.error("Missing evaluation.picks in config for continuous evaluation.")
    else:
        picks_path = as_abs_path(str(picks_file))
        if picks_path.exists():
            report.ok(f"Found manual picks file: {picks_path}")
        else:
            report.error(f"Missing manual picks file: {picks_path}")

    out_dir = as_abs_path(str(select(cfg, "prediction.output_dir")))
    model_setting = continuous_model_setting(cfg)
    stations = to_list(select(cfg, "prediction.stations", []))
    station_str = str(stations[0]) if stations else "UNKNOWN"
    stacking = str(select(cfg, "prediction.stacking", "median"))
    combine_array = optional_str(select(cfg, "prediction.combine_array_stations", False))
    combine_beams = not is_false_like(select(cfg, "prediction.combine_beams", False))

    if combine_array:
        pattern = str(
            out_dir / f"{model_setting}_{station_str}_{stacking}_combined_{combine_array}*.npz"
        )
    elif combine_beams:
        pattern = str(out_dir / f"{model_setting}_{station_str}_maxbeam*.npz")
    else:
        pattern = str(out_dir / f"{model_setting}_*_{stacking}_2*.npz")

    if has_match(pattern):
        report.ok(f"Found continuous prediction files matching: {pattern}")
    else:
        report.error(f"Missing continuous prediction files. Expected pattern: {pattern}")


def check_fdsn_connectivity(report: Reporter) -> None:
    try:
        from obspy import UTCDateTime
        from obspy.clients.fdsn import Client

        client = Client("UIB-NORSAR")
        client.get_stations(
            network="*",
            station="*",
            starttime=UTCDateTime("2020-01-01T00:00:00"),
            endtime=UTCDateTime("2020-01-01T00:01:00"),
            level="station",
        )
        report.ok("UIB-NORSAR FDSN service probe succeeded.")
    except Exception as exc:  # pragma: no cover - network-dependent
        report.error(f"UIB-NORSAR FDSN probe failed: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate config and required files before running the pipeline."
    )
    parser.add_argument(
        "--config",
        default="config_1stat.yaml",
        help="Path to config file relative to repo root (default: config_1stat.yaml)",
    )
    parser.add_argument(
        "--stage",
        choices=[
            "benchmark",
            "train",
            "predict-test",
            "evaluate-test",
            "predict-continuous",
            "evaluate-continuous",
        ],
        default="benchmark",
        help="Which workflow stage to validate",
    )
    parser.add_argument(
        "--check-fdsn",
        action="store_true",
        help="Probe UIB-NORSAR FDSN availability (network required)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    report = Reporter(strict=args.strict)
    report.info(f"Project root: {PROJECT_ROOT}")
    report.info(f"Config: {config_path}")
    report.info(f"Stage: {args.stage}")

    if not config_path.exists():
        report.error(f"Config file does not exist: {config_path}")
        return report.finalize()

    try:
        cfg = OmegaConf.load(str(config_path))
    except Exception as exc:
        report.error(f"Failed to parse config '{config_path}': {exc}")
        return report.finalize()

    check_common_config(cfg, report)

    if args.stage == "benchmark":
        check_stage_train(cfg, report)
        check_metadata_files(cfg, report)
    elif args.stage == "train":
        check_stage_train(cfg, report)
    elif args.stage == "predict-test":
        check_stage_predict_test(cfg, report)
    elif args.stage == "evaluate-test":
        check_stage_evaluate_test(cfg, report)
    elif args.stage == "predict-continuous":
        check_stage_predict_continuous(cfg, report)
    elif args.stage == "evaluate-continuous":
        check_stage_evaluate_continuous(cfg, report)

    if args.check_fdsn:
        check_fdsn_connectivity(report)

    return report.finalize()


if __name__ == "__main__":
    sys.exit(main())
