from weaver.engines.defaults import (
    default_argument_parser,
    default_config_parser,
)
from weaver.engines.vis import VISUALIZERS

def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)
    visualizer = VISUALIZERS.build(dict(type=cfg.vis.type, cfg=cfg))
    visualizer.vis(0)

if __name__ == "__main__":
    main()