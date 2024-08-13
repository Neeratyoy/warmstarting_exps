import argparse
from litgpt.config import Config
from pathlib import Path

from scales.utils import get_mup_shape_base


def quicker(base_path: Path, base: int, target: int, depth: int = 6, prefix: str = "width-only"):
    base_config = Config.from_file(base_path / f"width-only_depth={depth}_scale{base}.yaml")
    target_config = Config.from_file(base_path / f"width-only_depth={depth}_scale{target}.yaml")
    get_mup_shape_base(
        base_config, target_config, base_path / "mup" / f"{prefix}_scale{base}.bsh",
        verbose=True
    )


def get_args():
    parser = argparse.ArgumentParser(description="Parser for generating MuP base files")

    parser.add_argument("--depth", type=int, default=6, help="n_layer choice")

    return parser.parse_args()



if __name__ == "__main__":
    args = get_args()

    # check `scaling_exps/configs/width_only/` to make sense of upper index of scale available
    scale_map = {
        6: list(range(0, 9)),
        12: list(range(0, 5)),
    }
    
    scale_index = scale_map[args.depth]
    for base in scale_index:
        _scale_index = scale_index[min(base+1, len(scale_index)):]
        if len(_scale_index):
            print(base, _scale_index[0])
            quicker(
                Path("/work/dlclarge1/mallik-scaling/scaling_exps/configs/width_only/dev"),
                base,
                _scale_index[0],
                depth=args.depth,
                prefix=f"width-only_depth={args.depth}"
            )
        print()
