from pointbnn.engines.defaults import (
    default_argument_parser,
    default_config_parser,
)
import os
from pointbnn.datasets import build_dataset

def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)
    train_dataset = build_dataset(cfg.data.train)
    train_dataset.visualize_dataset(save_path=cfg.save_path)

if __name__ == "__main__":
    main()