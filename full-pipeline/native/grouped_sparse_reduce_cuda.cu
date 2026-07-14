#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void sparse_reduce_forward_kernel(
    const scalar_t* __restrict__ flat_q,
    const int64_t* __restrict__ sparse_indices,
    const int64_t* __restrict__ sparse_counts,
    scalar_t* __restrict__ out,
    int64_t batch_size,
    int64_t num_local_slots,
    int64_t max_sparse_points,
    int64_t flat_size) {
    int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_outputs = batch_size * num_local_slots;
    if (linear_idx >= total_outputs) {
        return;
    }

    int64_t batch_idx = linear_idx / num_local_slots;
    int64_t local_id = linear_idx % num_local_slots;
    if (local_id == 0) {
        out[linear_idx] = static_cast<scalar_t>(0);
        return;
    }

    int64_t count = sparse_counts[linear_idx];
    scalar_t acc = static_cast<scalar_t>(0);

    int64_t sparse_base = linear_idx * max_sparse_points;
    int64_t flat_base = batch_idx * flat_size;
    for (int64_t point_idx = 0; point_idx < count; ++point_idx) {
        int64_t flat_offset = sparse_indices[sparse_base + point_idx];
        acc = acc + flat_q[flat_base + flat_offset];
    }

    out[linear_idx] = acc;
}

torch::Tensor sparse_reduce_forward_cuda(
    torch::Tensor flat_q,
    torch::Tensor sparse_indices,
    torch::Tensor sparse_counts) {
    TORCH_CHECK(flat_q.is_cuda(), "flat_q must be a CUDA tensor");
    TORCH_CHECK(sparse_indices.is_cuda(), "sparse_indices must be a CUDA tensor");
    TORCH_CHECK(sparse_counts.is_cuda(), "sparse_counts must be a CUDA tensor");
    TORCH_CHECK(flat_q.dim() == 2, "flat_q must have shape [B, N]");
    TORCH_CHECK(sparse_indices.dim() == 3, "sparse_indices must have shape [B, L, P]");
    TORCH_CHECK(sparse_counts.dim() == 2, "sparse_counts must have shape [B, L]");
    TORCH_CHECK(flat_q.is_contiguous(), "flat_q must be contiguous");
    TORCH_CHECK(sparse_indices.is_contiguous(), "sparse_indices must be contiguous");
    TORCH_CHECK(sparse_counts.is_contiguous(), "sparse_counts must be contiguous");
    TORCH_CHECK(sparse_indices.scalar_type() == torch::kLong, "sparse_indices must use torch.long");
    TORCH_CHECK(sparse_counts.scalar_type() == torch::kLong, "sparse_counts must use torch.long");
    TORCH_CHECK(flat_q.scalar_type() == torch::kFloat16 || flat_q.scalar_type() == torch::kFloat32,
                "flat_q must use float16 or float32");
    TORCH_CHECK(flat_q.size(0) == sparse_indices.size(0), "Batch size mismatch between flat_q and sparse_indices");
    TORCH_CHECK(flat_q.size(0) == sparse_counts.size(0), "Batch size mismatch between flat_q and sparse_counts");
    TORCH_CHECK(sparse_indices.size(1) == sparse_counts.size(1),
                "Conductor slot mismatch between sparse_indices and sparse_counts");

    auto batch_size = flat_q.size(0);
    auto flat_size = flat_q.size(1);
    auto num_local_slots = sparse_indices.size(1);
    auto max_sparse_points = sparse_indices.size(2);
    auto out = torch::zeros({batch_size, num_local_slots}, flat_q.options());

    if (batch_size == 0 || num_local_slots == 0) {
        return out;
    }

    const int threads = 256;
    const int64_t total_outputs = batch_size * num_local_slots;
    const int blocks = static_cast<int>((total_outputs + threads - 1) / threads);

    c10::cuda::CUDAGuard device_guard(flat_q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(flat_q.get_device()).stream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(flat_q.scalar_type(), "sparse_reduce_forward_cuda", [&] {
        sparse_reduce_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            flat_q.data_ptr<scalar_t>(),
            sparse_indices.data_ptr<int64_t>(),
            sparse_counts.data_ptr<int64_t>(),
            out.data_ptr<scalar_t>(),
            batch_size,
            num_local_slots,
            max_sparse_points,
            flat_size);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
