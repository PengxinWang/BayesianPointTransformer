from weaver.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from weaver.engines.train import TRAINERS
from weaver.engines.launch import launch


def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    print(f'-------------trainer successfully built----------------')
    trainer.train()
    print(f'-------------train method successfully called----------------')


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()