#include <stdexcept>
#include <torch/extension.h>

torch::Tensor sparse_reduce_forward_cuda(
    torch::Tensor flat_q,
    torch::Tensor sparse_indices,
    torch::Tensor sparse_counts);

torch::Tensor sparse_reduce_forward(
    torch::Tensor flat_q,
    torch::Tensor sparse_indices,
    torch::Tensor sparse_counts) {
    if (!flat_q.is_cuda() || !sparse_indices.is_cuda() || !sparse_counts.is_cuda()) {
        throw std::runtime_error("sparse_reduce_forward expects CUDA tensors");
    }
    return sparse_reduce_forward_cuda(flat_q, sparse_indices, sparse_counts);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_reduce_forward", &sparse_reduce_forward, "Grouped sparse per-conductor reduction (CUDA)");
}
