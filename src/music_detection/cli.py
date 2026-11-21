"""Command-line entry point for both pipelines: python -m music_detection.cli <command> ..."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .m6.preprocessing.build_manifest import build_manifest
from .m6.preprocessing.demo_data import create_demo
from .shared.data.manifest import load_manifest
from .shared.evaluation.shortcut_checks import sample_rate_baseline
from .shared.features.spectral import BANDLIMITED_SAMPLE_RATE
from .shared.output.predict import predict_classical_files, predict_modern_files
from .shared.training.classical import run as run_classical
from .shared.training.modern import run as run_modern
from .shared.tuning.calibrate import calibrate_checkpoint_threshold, calibrate_svm_threshold
from .shared.tuning.ensemble import fit_ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-generated music detection pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("make-demo-data", help="Create synthetic demo dataset")
    demo.add_argument("--output", required=True)

    classical = subparsers.add_parser("classical", help="Train and evaluate classical SVM baseline")
    classical.add_argument("--manifest", required=True)
    classical.add_argument("--output", required=True)
    classical.add_argument(
        "--threshold-criterion",
        default="low_fpr",
        choices=["low_fpr", "f1", "youden", "cost_sensitive"],
        help="Criterion for tuning decision threshold",
    )
    classical.add_argument("--target-fpr", type=float, default=0.02, help="Target max FPR for low_fpr criterion")
    classical.add_argument(
        "--sample-rate",
        type=int,
        default=BANDLIMITED_SAMPLE_RATE,
        help="Resample all audio to this rate (16000 removes the M6 bandwidth shortcut; 44100 = original pipeline)",
    )
    classical.add_argument("--train-crops", type=int, default=2, help="Random 5 s windows per training clip")
    classical.add_argument("--jobs", type=int, default=-1, help="Parallel feature-extraction processes")

    modern = subparsers.add_parser("modern", help="Train and evaluate ResNet-18 modern branch")
    modern.add_argument("--manifest", required=True)
    modern.add_argument("--output", required=True)
    modern.add_argument("--epochs", type=int, default=5)
    modern.add_argument("--batch-size", type=int, default=32)
    modern.add_argument("--workers", type=int, default=8)
    modern.add_argument("--cache-dir")
    modern.add_argument("--resume")
    modern.add_argument("--no-pretrained", action="store_true")
    modern.add_argument(
        "--augment",
        action="store_true",
        help="Enable data augmentations (SpecAugment frequency & time masking)",
    )
    modern.add_argument(
        "--audio-augment",
        action="store_true",
        help="Enable full online audio-domain augmentations (volume, noise, MP3, pitch/time)",
    )
    modern.add_argument(
        "--threshold-criterion",
        default="low_fpr",
        choices=["low_fpr", "f1", "youden", "cost_sensitive"],
        help="Criterion for tuning decision threshold on validation set",
    )
    modern.add_argument("--target-fpr", type=float, default=0.02, help="Target max FPR for low_fpr criterion")
    modern.add_argument(
        "--sample-rate",
        type=int,
        default=BANDLIMITED_SAMPLE_RATE,
        help="Resample all audio to this rate (16000 removes the M6 bandwidth shortcut; 44100 = original pipeline)",
    )
    modern.add_argument("--lr", type=float, default=1e-4)
    modern.add_argument("--seed", type=int, default=42)

    ensemble = subparsers.add_parser("ensemble", help="Fit modern/classical blend weight + threshold on val, report test")
    ensemble.add_argument("--classical", required=True, help="Classical run directory (with predictions_*.csv)")
    ensemble.add_argument("--modern", required=True, help="Modern run directory (with predictions_*.csv)")
    ensemble.add_argument("--output", required=True)
    ensemble.add_argument("--criterion", default="low_fpr", choices=["low_fpr", "f1", "youden", "cost_sensitive"])
    ensemble.add_argument("--target-fpr", type=float, default=0.02)

    baseline = subparsers.add_parser("sr-baseline", help="Shortcut check: predict the label from native sample rate only")
    baseline.add_argument("--manifest", required=True)
    baseline.add_argument("--output", required=True, help="JSON file to write")

    tune = subparsers.add_parser("tune-threshold", help="Tune decision threshold on validation data for an existing model")
    tune.add_argument("--manifest", required=True)
    tune.add_argument("--model", required=True)
    tune.add_argument("--model-type", choices=["modern", "classical"], default="modern")
    tune.add_argument("--split", default="val", help="Split to tune threshold on (val or train)")
    tune.add_argument("--criterion", default="low_fpr", choices=["low_fpr", "f1", "youden", "cost_sensitive"])
    tune.add_argument("--target-fpr", type=float, default=0.02)

    predict = subparsers.add_parser("predict", help="Predict on audio tracks with full-track window aggregation")
    predict.add_argument("--model", required=True)
    predict.add_argument("--model-type", choices=["modern", "classical"], default="modern")
    predict.add_argument("--audio", nargs="+", required=True)
    predict.add_argument("--window-duration", type=float, default=5.0)
    predict.add_argument("--hop-duration", type=float, default=5.0)
    predict.add_argument("--aggregation", choices=["mean", "median", "max"], default="mean")
    predict.add_argument("--threshold", type=float, default=None)
    predict.add_argument("--detailed", action="store_true")

    manifest = subparsers.add_parser("build-m6-manifest", help="Build manifest from M6 database")
    manifest.add_argument("--root", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--include-duration", action="store_true")

    args = parser.parse_args()

    if args.command == "make-demo-data":
        print(create_demo(args.output))
    elif args.command == "classical":
        print(
            run_classical(
                args.manifest,
                args.output,
                threshold_criterion=args.threshold_criterion,
                target_fpr=args.target_fpr,
                sample_rate=args.sample_rate,
                train_crops=args.train_crops,
                jobs=args.jobs,
            )
        )
    elif args.command == "modern":
        print(
            run_modern(
                args.manifest,
                args.output,
                args.epochs,
                batch_size=args.batch_size,
                pretrained=not args.no_pretrained,
                workers=args.workers,
                cache_dir=args.cache_dir,
                resume=args.resume,
                augment=args.augment,
                audio_augment=args.audio_augment,
                threshold_criterion=args.threshold_criterion,
                target_fpr=args.target_fpr,
                sample_rate=args.sample_rate,
                learning_rate=args.lr,
                seed=args.seed,
            )
        )
    elif args.command == "ensemble":
        result = fit_ensemble(args.classical, args.modern, args.output, args.criterion, args.target_fpr)
        print(json.dumps({k: result[k] for k in ("modern_weight", "threshold", "val", "test")}, indent=2))
    elif args.command == "sr-baseline":
        result = sample_rate_baseline(load_manifest(args.manifest))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result["test"], indent=2))
    elif args.command == "tune-threshold":
        if args.model_type == "modern":
            result = calibrate_checkpoint_threshold(
                args.model,
                args.manifest,
                split=args.split,
                criterion=args.criterion,
                target_fpr=args.target_fpr,
            )
        else:
            result = calibrate_svm_threshold(
                args.model,
                args.manifest,
                split=args.split,
                criterion=args.criterion,
                target_fpr=args.target_fpr,
            )
        print("Threshold tuning result:")
        print(f"  Optimal threshold: {result['optimal_threshold']:.3f}")
        print(f"  Criterion: {result['criterion']}")
        print(f"  Metrics at 0.5: {result['metrics_at_0_5']}")
        print(f"  Metrics at optimal: {result['metrics_at_optimal']}")
        print(f"  FP reduction: {result.get('fp_reduction', 0)}")
    elif args.command == "predict":
        predict_fn = predict_modern_files if args.model_type == "modern" else predict_classical_files
        df = predict_fn(
            args.model,
            args.audio,
            window_duration=args.window_duration,
            hop_duration=args.hop_duration,
            aggregation=args.aggregation,
            threshold=args.threshold,
            detailed=args.detailed,
        )
        if args.detailed:
            for _, row in df.iterrows():
                print(f"File: {row['filename']}")
                print(f"  AI Probability ({args.aggregation}): {row['ai_probability']*100:.2f}% -> {row['prediction']}")
                print(f"  Confidence: {row['confidence']*100:.1f}%")
                print(f"  Flagged Windows: {row['flagged_windows']}/{row['total_windows']}")
                print("  Segments:")
                for seg in row["segments"]:
                    flag = "[FLAGGED]" if seg["flagged"] else "[OK]"
                    print(f"    {seg['start_sec']:6.2f}s - {seg['end_sec']:6.2f}s: {seg['probability']*100:5.1f}% {flag}")
        else:
            print(df.to_string(index=False))
    else:
        print(build_manifest(args.root, args.output, args.include_duration))


if __name__ == "__main__":
    main()
