/**
 * vllm-ascend binding for cloud_qsa_indexer_fused_v2 (aclnn, dlsym at runtime).
 *
 * Fuses: int8 quant matmul (AIC) + K group compression + RMSNorm + mRoPE (AIV).
 * All aclnn symbols are resolved at runtime from libopapi.so / libcust_opapi.so
 * by pytorch_npu_helper.hpp, so no aclnn op headers are needed at compile time.
 */
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

using namespace at;

std::tuple<at::Tensor, at::Tensor, at::Tensor> CloudQsaIndexerFusedV2ImplNpu(
    const at::Tensor& activation, const at::Tensor& weight, const at::Tensor& weight_scale,
    const at::Tensor& token_scale, const at::Tensor& positions, const at::Tensor& q_gamma,
    const at::Tensor& k_gamma, const at::Tensor& cos_sin, const at::Tensor& compressor_state_cache,
    const at::Tensor& compressor_state_block_table, const at::Tensor& token_to_req,
    const at::Tensor& query_start_loc, const at::Tensor& logical_positions,
    const at::Tensor& compressed_slots, const at::Tensor& rope_cache, int64_t q_head_num,
    int64_t k_head_num, int64_t head_dim, double eps, int64_t rotary_dim, int64_t compress_ratio,
    int64_t pos_offset)
{
    float epsFloat = static_cast<float>(eps);
    auto batchsize = activation.sizes()[0];
    at::Tensor q_out = at::empty(
        {batchsize, q_head_num * head_dim}, torch::TensorOptions().dtype(at::kBFloat16).device(activation.device()));
    at::Tensor k_out = at::empty(
        {batchsize, k_head_num * head_dim}, torch::TensorOptions().dtype(at::kBFloat16).device(activation.device()));
    // Dequantized raw K rows; the caller stores them into the raw key ring cache.
    at::Tensor raw_k_out = at::empty(
        {batchsize, k_head_num * head_dim}, torch::TensorOptions().dtype(at::kBFloat16).device(activation.device()));

    EXEC_NPU_CMD(aclnnCloudQsaIndexerFusedV2, activation, weight, weight_scale, token_scale, positions,
        q_gamma, k_gamma, cos_sin, compressor_state_cache, compressor_state_block_table,
        token_to_req, query_start_loc, logical_positions, compressed_slots, rope_cache,
        q_head_num, k_head_num, head_dim, epsFloat, rotary_dim, compress_ratio, pos_offset, q_out, k_out,
        raw_k_out);
    return std::make_tuple(q_out, k_out, raw_k_out);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> CloudQsaIndexerFusedV2ImplMeta(
    const at::Tensor& activation, const at::Tensor& weight, const at::Tensor& weight_scale,
    const at::Tensor& token_scale, const at::Tensor& positions, const at::Tensor& q_gamma,
    const at::Tensor& k_gamma, const at::Tensor& cos_sin, const at::Tensor& compressor_state_cache,
    const at::Tensor& compressor_state_block_table, const at::Tensor& token_to_req,
    const at::Tensor& query_start_loc, const at::Tensor& logical_positions,
    const at::Tensor& compressed_slots, const at::Tensor& rope_cache, int64_t q_head_num,
    int64_t k_head_num, int64_t head_dim, double eps, int64_t rotary_dim, int64_t compress_ratio,
    int64_t pos_offset)
{
    auto batchsize = activation.sym_size(0);
    auto q_out = at::empty_symint(
        {batchsize, q_head_num * head_dim}, torch::TensorOptions().dtype(at::kBFloat16).device(activation.device()));
    auto k_out = at::empty_symint(
        {batchsize, k_head_num * head_dim}, torch::TensorOptions().dtype(at::kBFloat16).device(activation.device()));
    auto raw_k_out = at::empty_symint(
        {batchsize, k_head_num * head_dim}, torch::TensorOptions().dtype(at::kBFloat16).device(activation.device()));
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(q_out, k_out, raw_k_out);
}

TORCH_LIBRARY(vllm_ascend_qsa_fused, m)
{
    m.def("cloud_qsa_indexer_fused_v2("
          "Tensor activation, "
          "Tensor weight, "
          "Tensor weight_scale, "
          "Tensor token_scale, "
          "Tensor positions, "
          "Tensor q_gamma, "
          "Tensor k_gamma, "
          "Tensor cos_sin, "
          "Tensor compressor_state_cache, "
          "Tensor compressor_state_block_table, "
          "Tensor token_to_req, "
          "Tensor query_start_loc, "
          "Tensor logical_positions, "
          "Tensor compressed_slots, "
          "Tensor rope_cache, "
          "int q_head_num, "
          "int k_head_num, "
          "int head_dim, "
          "float eps, "
          "int rotary_dim, "
          "int compress_ratio, "
          "int pos_offset"
          ") -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(vllm_ascend_qsa_fused, PrivateUse1, m)
{
    m.impl("cloud_qsa_indexer_fused_v2", &CloudQsaIndexerFusedV2ImplNpu);
}

TORCH_LIBRARY_IMPL(vllm_ascend_qsa_fused, Meta, m)
{
    m.impl("cloud_qsa_indexer_fused_v2", &CloudQsaIndexerFusedV2ImplMeta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("cloud_qsa_indexer_fused_v2", &CloudQsaIndexerFusedV2ImplNpu,
        "cloud_qsa_indexer_fused_v2 (int8 quant matmul + compress + norm + mrope fusion)");
}
