import argparse
from litgpt.config import Config
from pathlib import Path
import yaml

from saws.utils import get_mup_shape_base


KEYS_TO_EXCLUDE = [
    "max_micro_batch_size",
]


def quicker(
    base_path: Path,
    base: int,
    target: int,
    depth: int = 6,
    block: int = 1024,
    prefix: str = "width-only"
) -> None:
    with open(base_path / f"{prefix}_block={block}_depth={depth}_scale{base}.yaml", "rb") as f:
        _base_config = yaml.safe_load(f)
    for k in KEYS_TO_EXCLUDE:
        _base_config.pop(k, None)
    base_config = Config(**_base_config)
    # base_config = Config.from_file(
    #     base_path / f"{prefix}_block={block}_depth={depth}_scale{base}.yaml"
    # )

    with open(base_path / f"{prefix}_block={block}_depth={depth}_scale{target}.yaml", "rb") as f:
        _target_config = yaml.safe_load(f)
    for k in KEYS_TO_EXCLUDE:
        _target_config.pop(k, None)
    target_config = Config(**_target_config)
    # target_config = Config.from_file(
    #     base_path / f"{prefix}_block={block}_depth={depth}_scale{target}.yaml"
    # )
    get_mup_shape_base(
        base_config, target_config, base_path / "mup" / f"{prefix}_block={block}_depth={depth}_scale{target}.bsh",  # f"{prefix}_scale{base}.bsh",
        verbose=True
    )


def get_args():
    parser = argparse.ArgumentParser(description="Parser for generating MuP base files")

    parser.add_argument("--depth", type=int, default=6, help="n_layer choice")
    parser.add_argument("--block", type=int, default=1024, help="block_size choice")
    parser.add_argument("--prefix", type=str, default="width-only", help="file name prefix")

    return parser.parse_args()



if __name__ == "__main__":
    args = get_args()

    # check `scaling_exps/configs/width_only/` to make sense of upper index of scale available
    # NOTE: the map will likely require custom changes, not a scalable script/code
    scale_map = {
        # `n_layers/depth`: list(range(`smallest scale integer`, `largest scale integer + 1`))
        3: list(range(0, 7)),
    }

    scale_index = scale_map[args.depth]
    for base in scale_index:
        _scale_index = scale_index[min(base+1, len(scale_index)):]
        if len(_scale_index):
            print(base, _scale_index[0])
            quicker(
                # NOTE: the file path strings will likely require custom changes, not a scalable script/code
                Path(__file__).parent.parent.parent / "configs" / "width_only" / "types",
                base,
                _scale_index[0],
                depth=args.depth,
                block=args.block,
                prefix=args.prefix,  # f"{args.prefix}_block={args.block}_depth={args.depth}"
            )
        print()
