#include "unet.h"
#include "transfer.h"
#include "conv.h"
#include "layer.h"
#include "tza.h"
#include <cuda_fp16.h>
#include <cstring>

__global__ static void cuda_apply_hdr_transfer_function(const float* input_img, size_t h0, size_t w0, 
                                                         size_t c0, half* output, size_t h_padded, 
                                                         size_t w_padded, float norm_scale) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int ch = blockIdx.z * blockDim.z + threadIdx.z;

    if (y >= h_padded || x >= w_padded || ch >= c0)
        return;

    if (y < h0 && x < w0) {
        size_t src_idx = (y * w0 + x) * c0 + ch;
        size_t dst_idx = (y * w_padded + x) * c0 + ch;
        float val = Transfer::PU::forward(input_img[src_idx]);
        output[dst_idx] = __float2half(val * norm_scale);
    } else {
        size_t dst_idx = (y * w_padded + x) * c0 + ch;
        output[dst_idx] = 0.0f;
    }
}

__global__ static void cuda_apply_inverse_hdr_transfer_function(const half* input, size_t h0, size_t w0,
                                                                  size_t c, size_t w_padded, float* output, 
                                                                  float rcp_norm_scale) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int ch = blockIdx.z * blockDim.z + threadIdx.z;

    if (y >= h0 || x >= w0 || ch >= c)
        return;

    size_t src_idx = (y * w_padded + x) * c + ch;
    size_t dst_idx = (y * w0 + x) * c + ch;
    output[dst_idx] = __float2half(Transfer::PU::inverse(__half2float(input[src_idx]) * rcp_norm_scale));
}

__global__ static void cuda_concat_skip(const half* input, size_t c_input, 
                                        const half* skip, size_t c_skip,
                                        half* output, size_t h, size_t w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_pixels = h * w;
    
    if (idx >= total_pixels)
        return;

    for (size_t ch = 0; ch < c_input; ch++) {
        output[idx * (c_input + c_skip) + ch] = input[idx * c_input + ch];
    }
    
    for (size_t ch = 0; ch < c_skip; ch++) {
        output[idx * (c_input + c_skip) + c_input + ch] = skip[idx * c_skip + ch];
    }
}

static void apply_convolutions(const Layer& layer, const UNetModel& model, half*& d_input, half*& d_output, size_t& h, size_t& w, size_t& c, 
                                const dim3& block, dim3& grid) {
    const size_t conv_shared_mem = (block.x * block.y * CONV_IM2COL_TILE_K +
                                    CONV_IM2COL_TILE_K * block.z) * sizeof(half);

    for (size_t i = 0; i < layer.num_convs; i++) {
        size_t out_c = layer.out_channels[i];
        
        grid = dim3((h + block.x - 1) / block.x, (w + block.y - 1) / block.y, (out_c + block.z - 1) / block.z);
        conv_relu_nhwc_oihw_cuda<<<grid, block, conv_shared_mem>>>(
            d_input, d_output, h, w, 3, 3, c, out_c,
            model.weights->half_data + layer.weight_idxs[i],
            model.weights->half_data + layer.bias_idxs[i]);
        CUDA_ERR(cudaGetLastError()); 
        CUDA_ERR(cudaDeviceSynchronize());
        
        c = out_c;
        std::swap(d_input, d_output);
    }
}

static void apply_post_op(LayerPostOp post_op, half*& d_input, half*& d_output, size_t& h, size_t& w, size_t c,
                          const dim3& block, dim3& grid) {
    if (post_op == LayerPostOp::MAX_POOL) {
        size_t out_h = h / 2;
        size_t out_w = w / 2;
        
        grid = dim3((out_h + block.x - 1) / block.x, (out_w + block.y - 1) / block.y, (c + block.z - 1) / block.z);
        max_pool_nhwc_cuda<<<grid, block>>>(d_input, d_output, h, w, c);
        CUDA_ERR(cudaGetLastError());
        CUDA_ERR(cudaDeviceSynchronize());
        
        std::swap(d_input, d_output);
        h = out_h;
        w = out_w;
    } else if (post_op == LayerPostOp::NN_UPSAMPLE) {
        size_t out_h = h * 2;
        size_t out_w = w * 2;
        grid = dim3((out_h + block.x - 1) / block.x, (out_w + block.y - 1) / block.y, (c + block.z - 1) / block.z);
        nn_upsample_nhwc_cuda<<<grid, block>>>(d_input, d_output, h, w, c);
        CUDA_ERR(cudaGetLastError());
        CUDA_ERR(cudaDeviceSynchronize());
        
        std::swap(d_input, d_output);
        h = out_h;
        w = out_w;
    }
}

void oidn_unet_cuda(EXR::Image& input_img, UNetModel& model, float*& output_img) {
    const size_t h0 = input_img.height;
    const size_t w0 = input_img.width;
    const size_t c0 = input_img.loaded_channels.size();
    
    constexpr size_t alignment = 16;
    const size_t h_padded = ((h0 + alignment - 1) / alignment) * alignment;
    const size_t w_padded = ((w0 + alignment - 1) / alignment) * alignment;

    size_t h = h_padded;
    size_t w = w_padded;
    size_t c = c0;

    // precompute largest matrix size and encoder skip buffer requirements
    const size_t h_pad = h;
    const size_t w_pad = w;
    size_t hx = h, wx = w, cx = c;
    size_t max_sz = hx * wx * cx * 2;
    const size_t padded_input_elems = h_pad * w_pad * c0;
    const size_t base_encode_size = padded_input_elems;
    size_t* encode_output_offset = new size_t[model.num_encoder_layers];
    encode_output_offset[0] = 0;
    size_t next_encode_offset = base_encode_size;
    for (int i = 0; i< model.num_encoder_layers; i++) {
        for (int j = 0; j < model.encoder_layers[i].num_convs; j++) {
            cx = model.encoder_layers[i].out_channels[j];
            max_sz = max(max_sz, hx * wx * cx);
        }
        if (model.encoder_layers[i].post_op == LayerPostOp::MAX_POOL) {
            hx /= 2;
            wx /= 2;
        } else if (model.encoder_layers[i].post_op == LayerPostOp::NN_UPSAMPLE) {
            hx *= 2;
            wx *= 2;
        }

        if (i < model.num_encoder_layers - 1) {
            size_t layer_output_sz = hx * wx * cx;
            encode_output_offset[i + 1] = next_encode_offset;
            next_encode_offset += layer_output_sz;
        }

        max_sz = max(max_sz, hx * wx * cx);
    }
    int skip_idx = std::max(0, static_cast<int>(model.num_encoder_layers) - 3);
    for (int i = 0; i< model.num_decoder_layers; i++) {
        if (skip_idx >= 0) {
            size_t skip_c = (skip_idx == 0) ? c0 : model.encoder_layers[skip_idx].out_channels[0];
            cx += skip_c;
            max_sz = max(max_sz, hx * wx * cx);
            skip_idx--;
        }
        for (int j = 0; j < model.decoder_layers[i].num_convs; j++) {
            cx = model.decoder_layers[i].out_channels[j];
            max_sz = max(max_sz, hx * wx * cx);
        }
        if (model.decoder_layers[i].post_op == LayerPostOp::NN_UPSAMPLE) {
            hx *= 2;
            wx *= 2;
        }
        max_sz = max(max_sz, hx * wx * cx);
    }
    size_t tot_encode_outputs_sz = next_encode_offset;

    const float normScale = Transfer::compute_norm_scale();
    const float rcpNormScale = Transfer::compute_rcp_norm_scale();

    half* d_encode_outputs;
    CUDA_ERR(cudaMalloc(&d_encode_outputs, tot_encode_outputs_sz * sizeof(half)));

    half* d_buf0, *d_buf1;
    CUDA_ERR(cudaMalloc(&d_buf0, max_sz * sizeof(half)));
    CUDA_ERR(cudaMalloc(&d_buf1, max_sz * sizeof(half)));

    CUDA_ERR(cudaMemcpy(d_buf0, input_img.tensor.data(), h0 * w0 * c0 * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(8, 8, 4);
    dim3 grid((h + block.x - 1) / block.x, (w + block.y - 1) / block.y, (c + block.z - 1) / block.z);
    cuda_apply_hdr_transfer_function<<<grid, block>>>((float *)d_buf0, h0, w0, c0, d_buf1, h, w, normScale);
    CUDA_ERR(cudaGetLastError());
    CUDA_ERR(cudaDeviceSynchronize());

    CUDA_ERR(cudaMemcpy(d_encode_outputs, d_buf1, padded_input_elems * sizeof(half), cudaMemcpyDeviceToDevice));

    std::swap(d_buf0, d_buf1);    
    
    for (size_t layer_idx = 0; layer_idx < model.num_encoder_layers; layer_idx++) {
        const Layer& layer = model.encoder_layers[layer_idx];
        apply_convolutions(layer, model, d_buf0, d_buf1, h, w, c, block, grid);
        apply_post_op(layer.post_op, d_buf0, d_buf1, h, w, c, block, grid);

        if (layer_idx < model.num_encoder_layers - 1) {
            size_t layer_elements = h * w * c;
            half* dst = d_encode_outputs + encode_output_offset[layer_idx + 1];
            CUDA_ERR(cudaMemcpy(dst, d_buf0, layer_elements * sizeof(half), cudaMemcpyDeviceToDevice));
        }
    }
    
    skip_idx = std::max(0, static_cast<int>(model.num_encoder_layers) - 3);
    for (size_t layer_idx = 0; layer_idx < model.num_decoder_layers; layer_idx++) {
        const Layer& layer = model.decoder_layers[layer_idx];
        
        if (skip_idx >= 0) {
            const half* d_skip = (skip_idx == 0)
                ? d_encode_outputs
                : d_encode_outputs + encode_output_offset[skip_idx + 1];
            size_t skip_c = (skip_idx == 0) ? c0 : model.encoder_layers[skip_idx].out_channels[0];
            
            dim3 concat_block(256);
            dim3 concat_grid((h * w + concat_block.x - 1) / concat_block.x);
            cuda_concat_skip<<<concat_grid, concat_block>>>(d_buf0, c, d_skip, skip_c, d_buf1, h, w);
            CUDA_ERR(cudaGetLastError());
            CUDA_ERR(cudaDeviceSynchronize());
            
            std::swap(d_buf0, d_buf1);
            c = c + skip_c;
            skip_idx--;
        }
        
        apply_convolutions(layer, model, d_buf0, d_buf1, h, w, c, block, grid);
        apply_post_op(layer.post_op, d_buf0, d_buf1, h, w, c, block, grid);
    }
    
    grid = dim3((h0 + block.x - 1) / block.x, (w0 + block.y - 1) / block.y, (c + block.z - 1) / block.z);
    cuda_apply_inverse_hdr_transfer_function<<<grid, block>>>(d_buf0, h0, w0, c, w, (float *)d_buf1, rcpNormScale);
    CUDA_ERR(cudaGetLastError());
    CUDA_ERR(cudaDeviceSynchronize());
    
    output_img = new float[h0 * w0 * c];
    CUDA_ERR(cudaMemcpy(output_img, (float *)d_buf1, h0 * w0 * c * sizeof(float), cudaMemcpyDeviceToHost));
    
    CUDA_ERR(cudaFree(d_buf0));
    CUDA_ERR(cudaFree(d_buf1));
    CUDA_ERR(cudaFree(d_encode_outputs));
    delete[] encode_output_offset;
    freeUNetModel(model, true);
}
