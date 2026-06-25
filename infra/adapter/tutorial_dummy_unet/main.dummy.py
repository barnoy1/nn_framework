from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infra.adapter.tutorial_dummy_unet.manifest import create_manifest
from infra.adapter.tutorial_dummy_unet.model_builder import TutorialDummyUNetModelBuilder
from infra.adapter.tutorial_dummy_unet.runtime import (
    apply_single_channel_adapter_policy,
    build_runtime_api,
    load_tutorial_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tutorial helper: trace 3ch->1ch adapter transformation"
    )
    parser.add_argument(
        "--config",
        default="infra/adapter/tutorial_dummy_unet/configs/tutorial_dummy_unet.yaml",
        help="Path to tutorial dummy UNet config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    before_payload = load_tutorial_payload(config_path)
    after_payload = apply_single_channel_adapter_policy(before_payload)
    runtime_api = build_runtime_api(after_payload)
    first_conv_shape = tuple(runtime_api.model.enc1.layers[0].weight.shape)

    manifest = create_manifest(builder_factory=TutorialDummyUNetModelBuilder)
    print("Dummy adapter manifest summary")
    print(f"name={manifest.name}")
    print(f"config_subdir={manifest.config_subdir}")
    print(f"override_order={manifest.override_order}")
    print()
    print("3ch -> 1ch adaptation trace")
    print(f"config requested in_channels={before_payload.get('in_channels')}")
    print(f"adapter effective in_channels={after_payload.get('in_channels')}")
    print(f"policy marker={after_payload.get('adapter_channel_policy')}")
    print(f"runtime first-conv shape={first_conv_shape}")
    print(f"runtime metadata={runtime_api.metadata}")
    print()
    print("To use this adapter in full framework flows:")
    print("1) register TutorialDummyUNetModelBuilder in infra/adapter/core/registry.py")
    print("2) set app_config.adapter.name to:")
    print(f"   {manifest.name}")
    print("3) run normal CLI flows (train/eval/inference) with that app config")


if __name__ == "__main__":
    main()
