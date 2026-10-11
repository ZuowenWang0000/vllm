# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare LoopCoder's dense-diagonal and headwise gate computations."""

import argparse

import torch
import torch.nn.functional as F
from torch.utils.benchmark import Timer


def flatten_gate(logits, head_dim):
    heads, tokens, _ = logits.shape
    return (
        logits.sigmoid()
        .transpose(0, 1)
        .expand(-1, -1, head_dim)
        .reshape(tokens, heads * head_dim)
    )


def dense_gate(query, weight, bias):
    heads, tokens, head_dim = query.shape
    logits = F.linear(query.reshape(-1, head_dim), weight, bias)
    logits = logits.view(heads, tokens, heads).diagonal(dim1=0, dim2=2)
    return flatten_gate(logits.transpose(0, 1).unsqueeze(-1), head_dim)


def headwise_gate(query, weight, bias):
    logits = torch.baddbmm(bias.view(-1, 1, 1), query, weight.unsqueeze(-1))
    return flatten_gate(logits, query.shape[-1])


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16"
    )
    parser.add_argument("--num-heads", type=int, default=32, help="Heads per TP rank")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 16, 128, 1024, 4096]
    )
    parser.add_argument("--min-run-time", type=float, default=1.0)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    dtype = getattr(torch, args.dtype)
    heads, head_dim = args.num_heads, args.head_dim
    weight = (
        torch.randn(heads, head_dim, dtype=dtype, device=args.device) / head_dim**0.5
    )
    bias = torch.randn(heads, dtype=dtype, device=args.device)
    functions = [dense_gate, headwise_gate]
    if args.compile:
        functions = [
            torch.compile(fn, fullgraph=True, dynamic=True) for fn in functions
        ]

    print(
        f"torch={torch.__version__}, device={args.device}, dtype={args.dtype}, "
        f"heads={heads}, head_dim={head_dim}, compile={args.compile}"
    )
    print("tokens,dense_us,headwise_us,speedup,dense_iqr_us,headwise_iqr_us")
    for tokens in args.tokens:
        qkv = torch.randn(tokens, heads * head_dim * 3, dtype=dtype, device=args.device)
        query = qkv[:, : heads * head_dim].view(tokens, heads, head_dim).transpose(0, 1)
        torch.testing.assert_close(
            functions[1](query, weight, bias), functions[0](query, weight, bias)
        )
        results = [
            Timer(
                "fn(query, weight, bias)",
                globals={"fn": fn, "query": query, "weight": weight, "bias": bias},
            ).blocked_autorange(min_run_time=args.min_run_time)
            for fn in functions
        ]
        dense, headwise = results
        print(
            f"{tokens},{dense.median * 1e6:.2f},{headwise.median * 1e6:.2f},"
            f"{dense.median / headwise.median:.2f},"
            f"{dense.iqr * 1e6:.2f},{headwise.iqr * 1e6:.2f}"
        )


if __name__ == "__main__":
    main()
