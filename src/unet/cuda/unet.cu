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

static void apply_convolutions(const Layer& layer, const UNetModel& model, half*& d_input, size_t& h, size_t& w, size_t& c, 
                                const dim3& block, dim3& grid) {
    for (size_t i = 0; i < layer.num_convs; i++) {
        size_t out_c = layer.out_channels[i];
        half* d_output;
        CUDA_ERR(cudaMalloc(&d_output, h * w * out_c * sizeof(half)));
        
        grid = dim3((h + block.x - 1) / block.x, (w + block.y - 1) / block.y, (out_c + block.z - 1) / block.z);
        conv_relu_nhwc_oihw_cuda<<<grid, block>>>(
            d_input, d_output, h, w, 3, 3, c, out_c,
            model.weights->half_data + layer.weight_idxs[i],
            model.weights->half_data + layer.bias_idxs[i]);
        CUDA_ERR(cudaGetLastError()); 
        CUDA_ERR(cudaDeviceSynchronize());
        
        CUDA_ERR(cudaFree(d_input));
        d_input = d_output;
        c = out_c;
    }
}

static void apply_post_op(LayerPostOp post_op, half*& d_input, size_t& h, size_t& w, size_t c,
                          const dim3& block, dim3& grid) {
    if (post_op == LayerPostOp::MAX_POOL) {
        size_t out_h = h / 2;
        size_t out_w = w / 2;
        half* d_output;
        CUDA_ERR(cudaMalloc(&d_output, out_h * out_w * c * sizeof(half)));
        
        grid = dim3((out_h + block.x - 1) / block.x, (out_w + block.y - 1) / block.y, (c + block.z - 1) / block.z);
        max_pool_nhwc_cuda<<<grid, block>>>(d_input, d_output, h, w, c);
        CUDA_ERR(cudaGetLastError());
        CUDA_ERR(cudaDeviceSynchronize());
        
        CUDA_ERR(cudaFree(d_input));
        d_input = d_output;
        h = out_h;
        w = out_w;
    } else if (post_op == LayerPostOp::NN_UPSAMPLE) {
        size_t out_h = h * 2;
        size_t out_w = w * 2;
        half* d_output;
        CUDA_ERR(cudaMalloc(&d_output, out_h * out_w * c * sizeof(half)));
        
        grid = dim3((out_h + block.x - 1) / block.x, (out_w + block.y - 1) / block.y, (c + block.z - 1) / block.z);
        nn_upsample_nhwc_cuda<<<grid, block>>>(d_input, d_output, h, w, c);
        CUDA_ERR(cudaGetLastError());
        CUDA_ERR(cudaDeviceSynchronize());
        
        CUDA_ERR(cudaFree(d_input));
        d_input = d_output;
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

    // precompute largest matrix size
    size_t hx = h, wx = w, cx = c;
    size_t max_sz = hx * wx * cx * 2;
    size_t tot_encode_outputs_sz = 0;
    for (int i = 0; i< model.num_encoder_layers; i++) {
        for (int j = 0; j < model.encoder_layers[i].num_convs; j++) {
            cx = model.encoder_layers[i].out_channels[j];
            max_sz = max(max_sz, hx * wx * cx);
        }
        if (i < model.num_encoder_layers - 1)
            tot_encode_outputs_sz += hx * wx * cx;
        if (model.encoder_layers[i].post_op == LayerPostOp::MAX_POOL) {
            hx /= 2;
            wx /= 2;
        } else if (model.encoder_layers[i].post_op == LayerPostOp::NN_UPSAMPLE) {
            hx *= 2;
            wx *= 2;
        }
        max_sz = max(max_sz, hx * wx * cx);
    }

    int skip_idx = 3;
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
    
    const float normScale = Transfer::compute_norm_scale();
    const float rcpNormScale = Transfer::compute_rcp_norm_scale();


    float* d_input_img;
    CUDA_ERR(cudaMalloc(&d_input_img, h0 * w0 * c0 * sizeof(float)));
    CUDA_ERR(cudaMemcpy(d_input_img, input_img.tensor.data(), h0 * w0 * c0 * sizeof(float), cudaMemcpyHostToDevice));
    
    half* d_input;
    CUDA_ERR(cudaMalloc(&d_input, h * w * c * sizeof(half)));
    
    dim3 block(8, 8, 4);
    dim3 grid((h + block.x - 1) / block.x, (w + block.y - 1) / block.y, (c + block.z - 1) / block.z);
    cuda_apply_hdr_transfer_function<<<grid, block>>>(d_input_img, h0, w0, c0, d_input, h, w, normScale);
    CUDA_ERR(cudaGetLastError());
    CUDA_ERR(cudaDeviceSynchronize());
    CUDA_ERR(cudaFree(d_input_img));
    
    half* d_original_input;
    CUDA_ERR(cudaMalloc(&d_original_input, h * w * c * sizeof(half)));
    CUDA_ERR(cudaMemcpy(d_original_input, d_input, h * w * c * sizeof(half), cudaMemcpyDeviceToDevice));
    
    half** d_encode_outputs = new half*[model.num_encoder_layers];
    
    for (size_t layer_idx = 0; layer_idx < model.num_encoder_layers; layer_idx++) {
        const Layer& layer = model.encoder_layers[layer_idx];
        
        apply_convolutions(layer, model, d_input, h, w, c, block, grid);
        apply_post_op(layer.post_op, d_input, h, w, c, block, grid);
        
        CUDA_ERR(cudaMalloc(&d_encode_outputs[layer_idx], h * w * c * sizeof(half)));
        CUDA_ERR(cudaMemcpy(d_encode_outputs[layer_idx], d_input, h * w * c * sizeof(half), cudaMemcpyDeviceToDevice));
    }
    
    int skip_idx = 3;
    for (size_t layer_idx = 0; layer_idx < model.num_decoder_layers; layer_idx++) {
        const Layer& layer = model.decoder_layers[layer_idx];
        
        if (skip_idx >= 0) {
            const half* d_skip = (skip_idx == 0) ? d_original_input : d_encode_outputs[skip_idx];
            size_t skip_c = (skip_idx == 0) ? c0 : model.encoder_layers[skip_idx].out_channels[0];
            
            half* d_concat;
            CUDA_ERR(cudaMalloc(&d_concat, h * w * (c + skip_c) * sizeof(half)));
            
            dim3 concat_block(256);
            dim3 concat_grid((h * w + concat_block.x - 1) / concat_block.x);
            cuda_concat_skip<<<concat_grid, concat_block>>>(d_input, c, d_skip, skip_c, d_concat, h, w);
            CUDA_ERR(cudaGetLastError());
            CUDA_ERR(cudaDeviceSynchronize());
            
            CUDA_ERR(cudaFree(d_input));
            d_input = d_concat;
            c = c + skip_c;
            skip_idx--;
        }
        
        apply_convolutions(layer, model, d_input, h, w, c, block, grid);
        apply_post_op(layer.post_op, d_input, h, w, c, block, grid);
    }
    
    float* d_output;
    CUDA_ERR(cudaMalloc(&d_output, h0 * w0 * c * sizeof(float)));
    
    grid = dim3((h0 + block.x - 1) / block.x, (w0 + block.y - 1) / block.y, (c + block.z - 1) / block.z);
    cuda_apply_inverse_hdr_transfer_function<<<grid, block>>>(d_input, h0, w0, c, w, d_output, rcpNormScale);
    CUDA_ERR(cudaGetLastError());
    CUDA_ERR(cudaDeviceSynchronize());
    
    output_img = new float[h0 * w0 * c];
    CUDA_ERR(cudaMemcpy(output_img, d_output, h0 * w0 * c * sizeof(float), cudaMemcpyDeviceToHost));
    
    CUDA_ERR(cudaFree(d_input));
    CUDA_ERR(cudaFree(d_output));
    CUDA_ERR(cudaFree(d_original_input));
    for (size_t i = 0; i < model.num_encoder_layers; i++) {
        CUDA_ERR(cudaFree(d_encode_outputs[i]));
    }
    delete[] d_encode_outputs;
    
    freeUNetModel(model, true);
}
