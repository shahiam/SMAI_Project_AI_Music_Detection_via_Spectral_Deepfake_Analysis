from __future__ import annotations

import argparse

from .classical import run as run_classical
from .modern import run as run_modern
from .synthetic import create_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-generated music detection pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("make-demo-data")
    demo.add_argument("--output", required=True)
    classical = subparsers.add_parser("classical")
    classical.add_argument("--manifest", required=True)
    classical.add_argument("--output", required=True)
    modern = subparsers.add_parser("modern")
    modern.add_argument("--manifest", required=True)
    modern.add_argument("--output", required=True)
    modern.add_argument("--epochs", type=int, default=5)
    modern.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    if args.command == "make-demo-data":
        print(create_demo(args.output))
    elif args.command == "classical":
        print(run_classical(args.manifest, args.output))
    else:
        print(run_modern(args.manifest, args.output, args.epochs, pretrained=not args.no_pretrained))


if __name__ == "__main__":
    main()
