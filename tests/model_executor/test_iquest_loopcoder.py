# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm import envs
from vllm.model_executor import parameter
from vllm.model_executor.layers import linear
from vllm.model_executor.layers.utils import cpu_unquantized_gemm
from vllm.model_executor.models import iquest_loopcoder
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype

# These unit tests simulate TP ranks and never initialize process groups.
pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def gate_inputs(monkeypatch, request):
    tp_size, tp_rank, num_tokens, dtype, device = request.param
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    # Simulate a TP rank without requiring a multi-GPU process group.
    for module in (parameter, linear, iquest_loopcoder):
        monkeypatch.setattr(
            module, "get_tensor_model_parallel_world_size", lambda: tp_size
        )
    for module in (parameter, linear):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: tp_rank)

    torch.manual_seed(0)
    num_heads, head_dim = 32, 128
    weight = (
        torch.randn(num_heads, head_dim, dtype=dtype, device=device) / head_dim**0.5
    )
    bias = torch.randn(num_heads, dtype=dtype, device=device)
    with set_default_torch_dtype(dtype):
        gate = iquest_loopcoder.LoopGateProjection(num_heads, head_dim).to(device)
    gate.gate_proj.weight.weight_loader(gate.gate_proj.weight, weight)
    gate.gate_proj.bias.weight_loader(gate.gate_proj.bias, bias)

    local_heads = num_heads // tp_size
    # Q is a strided slice of the QKV projection in the model's attention path.
    qkv = torch.randn(
        num_tokens, local_heads * head_dim * 3, dtype=dtype, device=device
    )
    query = qkv[:, : local_heads * head_dim].view(num_tokens, local_heads, head_dim)
    query = query.transpose(0, 1)
    logits = F.linear(
        query.reshape(-1, head_dim), gate.gate_proj.weight, gate.gate_proj.bias
    )
    logits = logits.view(local_heads, num_tokens, local_heads).diagonal(dim1=0, dim2=2)
    expected = logits.sigmoid().unsqueeze(-1).expand(-1, -1, head_dim)
    return gate, query, expected.reshape(num_tokens, local_heads * head_dim)


_CASES = [
    pytest.param(
        (tp_size, tp_rank, num_tokens, dtype, device),
        marks=pytest.mark.skipif(
            device == "cuda"
            and (
                not current_platform.is_cuda_alike()
                or dtype not in current_platform.supported_dtypes
            ),
            reason="A CUDA or ROCm GPU supporting this dtype is required",
        ),
    )
    for tp_size, tp_rank in [(1, 0), (2, 1), (4, 3)]
    for num_tokens in [0, 1, 17]
    for dtype in [torch.float32, torch.float16, torch.bfloat16]
    for device in ["cpu", "cuda"]
]


@pytest.mark.parametrize("gate_inputs", _CASES, indirect=True)
@torch.inference_mode()
def test_loop_gate_matches_dense_diagonal(gate_inputs, monkeypatch):
    """Preserve head alignment, TP weight loading, dtype and QKV strides."""
    gate, query, expected = gate_inputs
    monkeypatch.setattr(current_platform, "is_cpu", lambda: False)
    gate.gate_proj.register_forward_pre_hook(
        lambda *_: pytest.fail("Unquantized headwise gates must not run a dense GEMM")
    )
    actual = gate(query)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "gate_inputs", [(2, 1, 17, torch.float32, "cpu")], indirect=True
)
@torch.inference_mode()
def test_loop_gate_preserves_specialized_linear_dispatch(gate_inputs, monkeypatch):
    """A specialized linear method must not be bypassed by raw-weight access."""
    gate, query, expected = gate_inputs
    monkeypatch.setattr(current_platform, "is_cpu", lambda: False)
    calls = []

    class SpecializedMethod(linear.UnquantizedLinearMethod):
        def apply(self, layer, x, bias=None):
            calls.append(x.shape)
            return F.linear(x, layer.weight, bias)

    gate.gate_proj.quant_method = SpecializedMethod()
    torch.testing.assert_close(gate(query), expected)
    assert calls == [torch.Size([16 * 17, 128])]


@pytest.mark.parametrize(
    "gate_inputs", [(1, 0, 17, torch.float32, "cpu")], indirect=True
)
@torch.inference_mode()
def test_loop_gate_preserves_cpu_packed_weight_dispatch(gate_inputs, monkeypatch):
    """CPU methods can retain packed weights after removing the raw parameter."""
    gate, query, expected = gate_inputs
    weight = gate.gate_proj.weight.detach().clone()
    gate.gate_proj.cpu_linear = lambda x, unused_weight, bias: F.linear(x, weight, bias)
    gate.gate_proj.quant_method._gemm_impl = cpu_unquantized_gemm
    gate.gate_proj.weight.data = torch.empty(0)
    monkeypatch.setattr(current_platform, "is_cpu", lambda: True)
    torch.testing.assert_close(gate(query), expected)


@pytest.mark.parametrize(
    "gate_inputs", [(1, 0, 17, torch.float32, "cpu")], indirect=True
)
@torch.inference_mode()
def test_loop_gate_preserves_batch_invariant_dispatch(gate_inputs, monkeypatch):
    gate, query, expected = gate_inputs
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(current_platform, "is_cpu", lambda: False)
    monkeypatch.setattr(current_platform, "is_cuda_alike", lambda: False)
    monkeypatch.setattr(current_platform, "is_xpu", lambda: False)
    calls = []
    gate.gate_proj.register_forward_pre_hook(lambda *_: calls.append(True))
    torch.testing.assert_close(gate(query), expected)
    assert calls == [True]


@pytest.mark.parametrize(
    "gate_inputs",
    [
        pytest.param(
            (1, 0, 17, torch.float32, device),
            marks=pytest.mark.skipif(
                device == "cuda" and not current_platform.is_cuda_alike(),
                reason="CUDA or ROCm is required",
            ),
        )
        for device in ["cpu", "cuda"]
    ],
    indirect=True,
)
@torch.inference_mode()
def test_loop_gate_compiles_with_dynamic_tokens(gate_inputs, monkeypatch):
    gate, query, expected = gate_inputs
    monkeypatch.setattr(current_platform, "is_cpu", lambda: False)
    backend = "aot_eager" if query.device.type == "cpu" else "inductor"
    compiled = torch.compile(gate, backend=backend, fullgraph=True, dynamic=True)
    for num_tokens in [1, 17]:
        torch.testing.assert_close(
            compiled(query[:, :num_tokens]), expected[:num_tokens]
        )
