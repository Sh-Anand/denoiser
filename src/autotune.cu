#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "conv.h"
#include "cuda_defs.h"
#include "exr.h"
#include "model.h"
#include "tza.h"

struct ConvSpec {
    std::string label;
    bool is_decoder;
    size_t layer_idx;
    size_t conv_idx;
    size_t in_h;
    size_t in_w;
    size_t in_c;
    size_t out_c;
    const half* weights;
    const half* bias;
};

struct ConvResult {
    ConvSpec spec;
    dim3 best_block;
    float best_time_ms;
};

static size_t compute_padded(size_t value, size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

static void collect_encoder_specs(const UNetModel& model,
                                  size_t& h,
                                  size_t& w,
                                  size_t& c,
                                  std::vector<ConvSpec>& specs) {
    for (size_t layer_idx = 0; layer_idx < model.encoder_layers.size(); ++layer_idx) {
        const Layer& layer = model.encoder_layers[layer_idx];
        for (size_t conv_idx = 0; conv_idx < layer.num_convs; ++conv_idx) {
            ConvSpec spec;
            spec.label = "enc" + std::to_string(layer_idx) + "_conv" + std::to_string(conv_idx);
            spec.is_decoder = false;
            spec.layer_idx = layer_idx;
            spec.conv_idx = conv_idx;
            spec.in_h = h;
            spec.in_w = w;
            spec.in_c = c;
            spec.out_c = layer.out_channels[conv_idx];
            const size_t weight_offset = layer.weight_idxs[conv_idx];
            const size_t bias_offset = layer.bias_idxs[conv_idx];
            spec.weights = model.weights->half_data + weight_offset;
            spec.bias = model.weights->half_data + bias_offset;
            specs.push_back(spec);
            c = spec.out_c;
        }

        if (layer.post_op == LayerPostOp::MAX_POOL) {
            h /= 2;
            w /= 2;
        } else if (layer.post_op == LayerPostOp::NN_UPSAMPLE) {
            h *= 2;
            w *= 2;
        }
    }
}

static void collect_decoder_specs(const UNetModel& model,
                                  size_t c0,
                                  size_t& h,
                                  size_t& w,
                                  size_t& c,
                                  std::vector<ConvSpec>& specs) {
    int skip_idx = std::max(0, static_cast<int>(model.encoder_layers.size()) - 3);
    for (size_t layer_idx = 0; layer_idx < model.decoder_layers.size(); ++layer_idx) {
        if (skip_idx >= 0) {
            size_t skip_c = (skip_idx == 0)
                                ? c0
                                : model.encoder_layers[skip_idx].out_channels[0];
            c += skip_c;
            skip_idx--;
        }

        const Layer& layer = model.decoder_layers[layer_idx];
        for (size_t conv_idx = 0; conv_idx < layer.num_convs; ++conv_idx) {
            ConvSpec spec;
            spec.label = "dec" + std::to_string(layer_idx) + "_conv" + std::to_string(conv_idx);
            spec.is_decoder = true;
            spec.layer_idx = layer_idx;
            spec.conv_idx = conv_idx;
            spec.in_h = h;
            spec.in_w = w;
            spec.in_c = c;
            spec.out_c = layer.out_channels[conv_idx];
            const size_t weight_offset = layer.weight_idxs[conv_idx];
            const size_t bias_offset = layer.bias_idxs[conv_idx];
            spec.weights = model.weights->half_data + weight_offset;
            spec.bias = model.weights->half_data + bias_offset;
            specs.push_back(spec);
            c = spec.out_c;
        }

        if (layer.post_op == LayerPostOp::NN_UPSAMPLE) {
            h *= 2;
            w *= 2;
        } else if (layer.post_op == LayerPostOp::MAX_POOL) {
            h /= 2;
            w /= 2;
        }
    }
}

static std::vector<dim3> build_candidate_blocks() {
    const unsigned int xy_vals[] = {2, 4, 8, 16, 32};
    const unsigned int z_vals[] = {4, 8, 16, 32};
    std::vector<dim3> candidates;
    for (unsigned int x : xy_vals) {
        for (unsigned int y : xy_vals) {
            for (unsigned int z : z_vals) {
                const unsigned long long threads =
                    static_cast<unsigned long long>(x) * y * z;
                if (threads <= 1024) {
                    candidates.emplace_back(x, y, z);
                }
            }
        }
    }
    return candidates;
}

static size_t shared_mem_for_block(const dim3& block) {
    const size_t block_hw = static_cast<size_t>(block.x) * block.y;
    const size_t tile_a = block_hw * CONV_IM2COL_TILE_K;
    const size_t tile_b = static_cast<size_t>(block.z) * CONV_IM2COL_TILE_K;
    return (tile_a + tile_b) * sizeof(half);
}

static bool is_candidate_valid(const dim3& block, size_t max_shared_mem) {
    if (block.x == 0 || block.y == 0 || block.z == 0)
        return false;
    const unsigned long long threads =
        static_cast<unsigned long long>(block.x) * block.y * block.z;
    if (threads == 0 || threads > 1024)
        return false;
    return shared_mem_for_block(block) <= max_shared_mem;
}

static dim3 compute_grid(const dim3& block,
                         size_t in_h,
                         size_t in_w,
                         size_t out_c) {
    dim3 grid;
    grid.x = (in_h + block.x - 1) / block.x;
    grid.y = (in_w + block.y - 1) / block.y;
    grid.z = (out_c + block.z - 1) / block.z;
    if (grid.x == 0)
        grid.x = 1;
    if (grid.y == 0)
        grid.y = 1;
    if (grid.z == 0)
        grid.z = 1;
    return grid;
}

static float launch_and_time_conv(const ConvSpec& spec,
                                  const dim3& block,
                                  const half* input,
                                  half* output,
                                  size_t max_shared_mem,
                                  int repeats) {
    if (!is_candidate_valid(block, max_shared_mem)) {
        return std::numeric_limits<float>::infinity();
    }

    const dim3 grid = compute_grid(block, spec.in_h, spec.in_w, spec.out_c);
    const size_t shared_mem = shared_mem_for_block(block);

    cudaEvent_t start, stop;
    CUDA_ERR(cudaEventCreate(&start));
    CUDA_ERR(cudaEventCreate(&stop));

    CUDA_ERR(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        conv_relu_nchw_oihw_cuda<<<grid, block, shared_mem>>>(
            input,
            output,
            spec.in_h,
            spec.in_w,
            spec.in_c,
            spec.out_c,
            spec.weights,
            spec.bias);
    }
    CUDA_ERR(cudaEventRecord(stop));
    CUDA_ERR(cudaEventSynchronize(stop));
    CUDA_ERR(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_ERR(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_ERR(cudaEventDestroy(start));
    CUDA_ERR(cudaEventDestroy(stop));

    const float avg_ms = total_ms / static_cast<float>(repeats);
    return avg_ms;
}

static ConvResult autotune_conv(const ConvSpec& spec,
                                const std::vector<dim3>& candidates,
                                const half* input,
                                half* output,
                                size_t max_shared_mem,
                                int repeats) {
    ConvResult result;
    result.spec = spec;
    result.best_block = dim3(1, 1, 1);
    result.best_time_ms = std::numeric_limits<float>::infinity();

    for (const dim3& block : candidates) {
        const float time_ms =
            launch_and_time_conv(spec, block, input, output, max_shared_mem, repeats);
        if (time_ms < result.best_time_ms) {
            result.best_time_ms = time_ms;
            result.best_block = block;
        }
    }

    return result;
}

static void print_block_dims_initializer(const std::vector<std::vector<dim3>>& blocks) {
    printf("{\n");
    for (size_t i = 0; i < blocks.size(); ++i) {
        printf("    {");
        for (size_t j = 0; j < blocks[i].size(); ++j) {
            const dim3& blk = blocks[i][j];
            printf("dim3(%u, %u, %u)", blk.x, blk.y, blk.z);
            if (j + 1 < blocks[i].size())
                printf(", ");
        }
        printf("}%s\n", (i + 1 < blocks.size()) ? "," : "");
    }
    printf("};\n");
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        printf("Usage: %s <input_exr> [model_name=rt_hdr]\n", argv[0]);
        return 1;
    }

    const char* input_file = argv[1];
    const std::string model_name = (argc >= 3) ? argv[2] : "rt_hdr";
    const std::string weights_path = "../weights/" + model_name + ".tza";

    EXR::Image exr;
    exr.load(input_file, {"R", "G", "B"});

    TzaFile weights;
    loadTza(weights_path, weights);
    UNetModel model = createUNetModel(model_name, weights, true);

    const size_t alignment = 16;
    const size_t h_padded = compute_padded(exr.height, alignment);
    const size_t w_padded = compute_padded(exr.width, alignment);
    size_t h = h_padded;
    size_t w = w_padded;
    size_t c = exr.loaded_channels.size();

    std::vector<ConvSpec> specs;
    collect_encoder_specs(model, h, w, c, specs);
    collect_decoder_specs(model, exr.loaded_channels.size(), h, w, c, specs);

    if (specs.empty()) {
        printf("No convolution layers found for model %s\n", model_name.c_str());
        freeUNetModel(model, true);
        return 0;
    }

    size_t max_in_elems = 0;
    size_t max_out_elems = 0;
    for (const auto& spec : specs) {
        max_in_elems = std::max(max_in_elems, spec.in_h * spec.in_w * spec.in_c);
        max_out_elems = std::max(max_out_elems, spec.in_h * spec.in_w * spec.out_c);
    }

    half* d_input = nullptr;
    half* d_output = nullptr;
    CUDA_ERR(cudaMalloc(&d_input, max_in_elems * sizeof(half)));
    CUDA_ERR(cudaMalloc(&d_output, max_out_elems * sizeof(half)));
    CUDA_ERR(cudaMemset(d_input, 0, max_in_elems * sizeof(half)));
    CUDA_ERR(cudaMemset(d_output, 0, max_out_elems * sizeof(half)));

    int device = 0;
    CUDA_ERR(cudaGetDevice(&device));
    int shared_default = 0;
    int shared_optin = 0;
    CUDA_ERR(cudaDeviceGetAttribute(&shared_default, cudaDevAttrMaxSharedMemoryPerBlock, device));
    CUDA_ERR(cudaDeviceGetAttribute(&shared_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    const size_t max_shared_mem = static_cast<size_t>(std::max(shared_default, shared_optin));

    const std::vector<dim3> candidates = build_candidate_blocks();
    const int repeats = 3;

    std::vector<ConvResult> results;
    results.reserve(specs.size());
    printf("Autotuning %zu convolution kernels...\n", specs.size());
    for (const auto& spec : specs) {
        ConvResult result =
            autotune_conv(spec, candidates, d_input, d_output, max_shared_mem, repeats);
        printf("  %-12s -> block (%u, %u, %u) : %.4f ms\n",
               result.spec.label.c_str(),
               result.best_block.x,
               result.best_block.y,
               result.best_block.z,
               result.best_time_ms);
        results.push_back(result);
    }

    std::vector<std::vector<dim3>> encoder_blocks(model.encoder_layers.size());
    std::vector<std::vector<dim3>> decoder_blocks(model.decoder_layers.size());
    for (const auto& result : results) {
        if (result.spec.is_decoder) {
            decoder_blocks[result.spec.layer_idx].push_back(result.best_block);
        } else {
            encoder_blocks[result.spec.layer_idx].push_back(result.best_block);
        }
    }

    printf("\nSuggested encoder block dims:\n");
    print_block_dims_initializer(encoder_blocks);

    printf("\nSuggested decoder block dims:\n");
    print_block_dims_initializer(decoder_blocks);

    CUDA_ERR(cudaFree(d_input));
    CUDA_ERR(cudaFree(d_output));
    freeUNetModel(model, true);
    return 0;
}
