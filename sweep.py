#!/usr/bin/env python3
"""Run paired mask/auxiliary ablations, using identical seeds and validation masks."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from exp2.runner import Config, train, json_write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="results/ablation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Print all configs without training or writing")
    args = parser.parse_args()
    cfg = Config(**json.loads(Path(args.config).read_text()))
    variants = [
        ("mixture", {}),
        ("random20", {"mixture": (1, 0, 0), "random_ratio": 0.2}),
        ("random40", {"mixture": (1, 0, 0), "random_ratio": 0.4}),
        ("random50", {"mixture": (1, 0, 0), "random_ratio": 0.5}),
        ("random60", {"mixture": (1, 0, 0), "random_ratio": 0.6}),
        ("module", {"mixture": (0, 0, 1)}),
    ]
    results = []
    for seed in args.seeds:
        for name, changes in variants:
            for aux in (True, False):
                output = str(Path(args.output) / f"{name}_aux{int(aux)}_seed{seed}")
                run = replace(cfg, **changes, seed=seed, device=args.device, output=output,
                              lambda_msg=cfg.lambda_msg if aux else 0.0,
                              lambda_adv=cfg.lambda_adv if aux else 0.0)
                if args.dry_run:
                    from dataclasses import asdict
                    print(json.dumps(asdict(run)))
                    continue
                train(run)
                import torch
                checkpoint = torch.load(Path(output) / "checkpoint.pt", map_location="cpu", weights_only=True)
                results.append({"variant": name, "aux": aux, "seed": seed, "output": output,
                                "best_epoch": checkpoint["epoch"], "validation": checkpoint["validation"]})
                json_write(Path(args.output) / "summary.json", results)


if __name__ == "__main__":
    main()
