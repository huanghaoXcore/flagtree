import pytest
import torch
import random

import triton
import triton.language as tl


@pytest.mark.parametrize("num_warps", [1, 4])
@pytest.mark.parametrize("grid", [16, 32])
def test_scatter(num_warps, grid, device):

    @triton.jit
    def kernel(
        num_tokens,
        expert_counters,  # Counter for each expert (atomic operation)
        input_data,
        input_data_stride0,
        expert_ids,  # Expert ID assigned to each token
        expert_ids_stride0,
        output_data,
        output_stride0,
        output_indices,  # Record output position
        output_indices_stride0,
        token_per_expert: tl.constexpr,  # token number per expert
        topk: tl.constexpr,
        HIDDEN_SIZE: tl.constexpr,
    ):
        """
        Simplified version of the scatter operator, retaining the core double loop structure and atomic_add operation
        """
        start_token = tl.program_id(0)
        grid_size = tl.num_programs(0)

        # Prepare offset for data loading
        offset = tl.arange(0, HIDDEN_SIZE)

        # Outer loop: iterate over tokens
        for token_id in range(start_token, num_tokens, grid_size):

            data_to_copy = tl.load(input_data + token_id * input_data_stride0 + offset)

            # Inner loop: process topk experts for each token
            for k_idx in tl.range(0, topk, 1):
                # Get current expert ID
                expert_id = tl.load(expert_ids + token_id * expert_ids_stride0 + k_idx)

                # Key operation: atomic add to get output position
                # There may be concurrency issues here
                local_idx = tl.atomic_add(expert_counters + expert_id, 1)
                global_idx = expert_id * token_per_expert + local_idx

                tl.store(output_indices + token_id * output_indices_stride0 + k_idx, global_idx)
                output_ptr = output_data + global_idx * output_stride0
                tl.store(output_ptr + offset, data_to_copy)

    num_tokens = 32  # number of tokens
    num_experts = 8  # number of experts
    hidden_size = 512  # hidden size (features)
    topk = 2  # each token is assigned to topk experts
    token_per_expert = num_tokens * topk // num_experts

    # input data
    input_data = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.float32)

    # expert ids
    expert_ids = list(range(num_experts)) * token_per_expert
    random.shuffle(expert_ids)
    expert_ids = torch.tensor(expert_ids, device=device, dtype=torch.int32).reshape(num_tokens, topk)

    # expert counters
    expert_counters = torch.zeros(num_experts, device=device, dtype=torch.int32)

    # output data and indices
    output_data = torch.zeros(num_tokens * topk, hidden_size, device=device, dtype=torch.float32)
    output_indices = torch.full((num_tokens, topk), -1, device=device, dtype=torch.int32)

    kernel[(grid, )](
        num_tokens=num_tokens,
        expert_counters=expert_counters,
        input_data=input_data,
        input_data_stride0=input_data.stride(0),
        expert_ids=expert_ids,
        expert_ids_stride0=expert_ids.stride(0),
        output_data=output_data,
        output_stride0=output_data.stride(0),
        output_indices=output_indices,
        output_indices_stride0=output_indices.stride(0),
        token_per_expert=token_per_expert,
        topk=topk,
        HIDDEN_SIZE=hidden_size,
        num_warps=num_warps,
    )

    for i in range(num_experts):
        expected_count = (expert_ids == i).sum().item()
        actual_count = expert_counters[i].item()
        assert expected_count == actual_count

    valid_indices = output_indices[output_indices >= 0]
    unique_indices = torch.unique(valid_indices)
    assert unique_indices.numel() == num_tokens * topk

    for token_id in range(num_tokens):
        for k in range(topk):
            out_idx = output_indices[token_id, k].item()
            input_vec = input_data[token_id, :hidden_size]
            output_vec = output_data[out_idx, :hidden_size]
            torch.testing.assert_close(input_vec, output_vec)
