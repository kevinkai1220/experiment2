#!/usr/bin/env python3
"""Train or evaluate Exp2; configuration files are portable JSON."""
import argparse
from dataclasses import asdict
import json
from exp2.runner import Config, train, evaluate_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--config", help="JSON configuration; CLI values override it")
    defaults = asdict(Config())
    bools = {"synthetic", "assume_single_tissue", "checkpoint_senders"}
    ints = {"seed", "epochs", "batch_size", "hidden", "neighbors", "n_modules", "shuffle_repeats", "cpu_threads", "sender_chunk_size"}
    strings = {"data", "coordinates", "type_key", "perturbation_key", "tissue_key", "control_label",
               "transform", "output", "device", "graph_mode", "distance_mode"}
    for name in defaults:
        option = "--" + name.replace("_", "-")
        if name in bools:
            training.add_argument(option, action=argparse.BooleanOptionalAction, default=None)
        elif name in ("mixture", "validation_ratios"):
            training.add_argument(option, type=float, nargs="+", default=None)
        else:
            training.add_argument(option, type=int if name in ints else str if name in strings else float,
                                  default=None)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--device", default="auto")
    evaluation.add_argument("--data")
    evaluation.add_argument("--coordinates")
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "evaluate":
        args["data_path"] = args.pop("data")
        evaluate_checkpoint(**args)
    else:
        config_path = args.pop("config")
        config = json.loads(open(config_path).read()) if config_path else {}
        config.update({key: value for key, value in args.items() if value is not None})
        train(Config(**config))


if __name__ == "__main__":
    main()
