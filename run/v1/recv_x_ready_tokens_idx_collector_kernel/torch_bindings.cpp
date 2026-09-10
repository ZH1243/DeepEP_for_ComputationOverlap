#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <limits>

#include "collector.cuh"

long long get_device_clock_rate_khz() {
    int device = -1;
    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    TORCH_CHECK(properties.major >= 9, "the collector targets Hopper (SM90) or newer");
    return static_cast<long long>(properties.clockRate);
}

void launch(
    const torch::Tensor& topk,
    const torch::Tensor& begin, const torch::Tensor& end, const torch::Tensor& ready,
    const torch::Tensor& indices, const torch::Tensor& counts,
    const torch::Tensor& consumed, const torch::Tensor& status,
    unsigned long long timeout_cycles) {
    TORCH_CHECK(topk.is_cuda() && topk.is_contiguous() && topk.dim() == 2,
                "recv_topk_idx must be a contiguous CUDA matrix");
    TORCH_CHECK(topk.scalar_type() == torch::kInt32 || topk.scalar_type() == torch::kInt64,
                "recv_topk_idx must have dtype int32 or int64");
    for (const auto& tensor : {begin, end, ready, indices, counts, consumed, status}) {
        TORCH_CHECK(tensor.is_cuda() && tensor.device() == topk.device() &&
                    tensor.is_contiguous() && tensor.scalar_type() == torch::kInt32,
                    "state tensors must be contiguous int32 tensors on recv_topk_idx's GPU");
    }
    constexpr auto max_int = std::numeric_limits<int>::max();
    TORCH_CHECK(topk.size(0) <= max_int && topk.size(1) >= 1 && topk.size(1) <= 32,
                "require num_rows <= INT_MAX and 1 <= num_topk <= 32");
    TORCH_CHECK(begin.dim() == 1 && begin.numel() >= 1 && begin.numel() <= max_int &&
                end.sizes() == begin.sizes() && ready.sizes() == begin.sizes() &&
                consumed.sizes() == begin.sizes(), "range arrays must have matching 1D shapes");
    TORCH_CHECK(indices.dim() == 2 && indices.size(0) == recv_x_ready::kExperts &&
                indices.size(1) <= max_int, "indices must have shape [8, capacity <= INT_MAX]");
    TORCH_CHECK(counts.dim() == 1 && counts.numel() == recv_x_ready::kExperts &&
                status.dim() == 1 && status.numel() == 1, "invalid count/status shape");

    const c10::cuda::CUDAGuard guard(topk.device());
    C10_CUDA_CHECK(recv_x_ready::launch_collector(
        topk.data_ptr(), topk.scalar_type() == torch::kInt64,
        static_cast<int>(topk.size(0)), static_cast<int>(topk.size(1)),
        begin.data_ptr<int>(), end.data_ptr<int>(), ready.data_ptr<int>(),
        static_cast<int>(begin.numel()), indices.data_ptr<int>(),
        static_cast<int>(indices.size(1)), counts.data_ptr<int>(), consumed.data_ptr<int>(),
        status.data_ptr<int>(), timeout_cycles,
        c10::cuda::getCurrentCUDAStream(topk.get_device()).stream()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("get_device_clock_rate_khz", &get_device_clock_rate_khz,
               "Return the current CUDA device clock rate after validating SM90+");
    module.def("launch", &launch, "Launch one-CTA ready-index collector on the current CUDA stream");
}
