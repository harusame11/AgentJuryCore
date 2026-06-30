import argparse
import contextlib
import datetime
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from Lib.api_utils import (
    API_MODEL_MAP,
    AgentJury_alg_enhance,
    AgentJury_alg_enhance_noGT,
)
from Lib.api_utils_4_handcrafted import AgentJury_hc_enhance
from Lib.alg_melting import (
    AgentJury_ablation_no_probe,
    AgentJury_ablation_no_spotlight,
    AgentJury_ablation_static_experts,
)


PUBLIC_METHODS = {
    "agentjury_alg_enhance": AgentJury_alg_enhance,
    "agentjury_alg_enhance_nogt": AgentJury_alg_enhance_noGT,
    "agentjury_hc_enhance": AgentJury_hc_enhance,
    "ablation_no_spotlight": AgentJury_ablation_no_spotlight,
    "ablation_static_experts": AgentJury_ablation_static_experts,
    "ablation_no_probe": AgentJury_ablation_no_probe,
}

HC_METHODS = {"agentjury_hc_enhance"}
ALG_ABLATION_METHODS = {
    "ablation_no_spotlight",
    "ablation_static_experts",
    "ablation_no_probe",
}


def _str_to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _build_client(args: argparse.Namespace) -> OpenAI:
    api_key = args.api_key or os.environ.get("SILICON_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "Error: API key is required. Set SILICON_API_KEY or pass --api_key.",
            file=sys.stderr,
        )
        sys.exit(1)

    return OpenAI(
        api_key=api_key,
        base_url=args.base_url,
    )


def _run_method(
    method: str,
    client: OpenAI,
    args: argparse.Namespace,
    model_id: str,
):
    func = PUBLIC_METHODS[method]

    common_kwargs = {
        "client": client,
        "directory_path": args.directory_path,
        "model": model_id,
        "max_tokens": args.max_tokens,
    }

    if method in HC_METHODS:
        return func(
            **common_kwargs,
            target_seg_size=args.target_seg_size,
            max_probe_k=args.max_probe_k,
        )

    if method in {"agentjury_alg_enhance", "agentjury_alg_enhance_nogt", "ablation_static_experts"}:
        return func(
            **common_kwargs,
            probe_k=args.probe_k,
        )

    return func(**common_kwargs)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run AgentJury failure attribution experiments."
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=sorted(PUBLIC_METHODS),
        help="Public experiment entry. Historical baselines are kept in Lib/ but are not exposed here.",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(API_MODEL_MAP),
        help=f"Model alias. Available aliases: {', '.join(sorted(API_MODEL_MAP))}",
    )
    parser.add_argument(
        "--directory_path",
        required=True,
        help="Directory containing dataset JSON files.",
    )
    parser.add_argument(
        "--is_handcrafted",
        default="False",
        choices=["True", "False", "true", "false"],
        help="Set True when evaluating the Hand-Crafted split.",
    )
    parser.add_argument(
        "--api_key",
        default=os.environ.get("SILICON_API_KEY", ""),
        help="OpenAI-compatible API key. Defaults to SILICON_API_KEY.",
    )
    parser.add_argument(
        "--base_url",
        default=os.environ.get("SILICON_BASE_URL", "https://api.siliconflow.cn/v1"),
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1500,
        help="Maximum output tokens for judge calls.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Directory for run logs.",
    )
    parser.add_argument(
        "--probe_k",
        type=int,
        default=4,
        help="K-partition probe count for Algorithm-Generated methods.",
    )
    parser.add_argument(
        "--target_seg_size",
        type=int,
        default=15,
        help="Adaptive probe target segment size for the Hand-Crafted method.",
    )
    parser.add_argument(
        "--max_probe_k",
        type=int,
        default=8,
        help="Maximum adaptive probe count for the Hand-Crafted method.",
    )
    args = parser.parse_args()

    args.is_handcrafted = _str_to_bool(args.is_handcrafted)

    if args.method in HC_METHODS and not args.is_handcrafted:
        print("[Config] --is_handcrafted was False; enabling it for the HC method.")
        args.is_handcrafted = True

    if args.method in ALG_ABLATION_METHODS and args.is_handcrafted:
        print("[Config] Algorithm ablations are defined for Algorithm-Generated data; setting is_handcrafted=False.")
        args.is_handcrafted = False

    os.makedirs(args.output_dir, exist_ok=True)
    dataset_name = os.path.basename(os.path.normpath(args.directory_path))
    split_suffix = "_hand" if args.is_handcrafted else "_alg"
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    output_filename = f"{args.method}_{args.model}{split_suffix}_{dataset_name}_{timestamp}.txt"
    output_path = os.path.join(args.output_dir, output_filename)

    model_id = API_MODEL_MAP[args.model]
    client = _build_client(args)

    print(f"Analysis method: {args.method}")
    print(f"Model alias: {args.model} ({model_id})")
    print(f"Input directory: {args.directory_path}")
    print(f"Output will be saved to: {output_path}")

    with open(output_path, "w", encoding="utf-8", buffering=1) as output_file:
        with contextlib.redirect_stdout(output_file):
            print(f"--- Starting Analysis: {args.method} ---")
            print(f"Timestamp: {datetime.datetime.now()}")
            print(f"Model Used: {model_id}")
            print(f"Input Directory: {args.directory_path}")
            print(f"Is Handcrafted: {args.is_handcrafted}")
            print("-" * 20)
            _run_method(args.method, client, args, model_id)

    print(f"Analysis complete. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
