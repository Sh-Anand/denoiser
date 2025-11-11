#include "unet.h"
#include "transfer.h"
#include "conv.h"
#include "layer.h"
#include "tza.h"
#include <cuda_fp16.h>
#include <cstring>

__global__ static void cuda_apply_hdr_transfer_function(const float* input_img, size_t h0, size_t w0, 
                                                         size_t c0, float* output, size_t h_padded, 
                                                         size_t w_padded, float norm_scale) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int ch = blockIdx.z * blockDim.z + threadIdx.z;

    if (y >= h_padded || x >= w_padded || ch >= c0)
        return;

    if (y < h0 && x < w0) {
        size_t src_idx = (y * w0 + x) * c0 + ch;
        size_t dst_idx = (y * w_padded + x) * c0 + ch;
        output[dst_idx] = Transfer::PU::forward(input_img[src_idx]) * norm_scale;
    } else {
        size_t dst_idx = (y * w_padded + x) * c0 + ch;
        output[dst_idx] = 0.0f;
    }
}

__global__ static void cuda_apply_inverse_hdr_transfer_function(const float* input, size_t h0, size_t w0,
                                                                  size_t c, size_t w_padded, float* output, 
                                                                  float rcp_norm_scale) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int ch = blockIdx.z * blockDim.z + threadIdx.z;

    if (y >= h0 || x >= w0 || ch >= c)
        return;

    size_t src_idx = (y * w_padded + x) * c + ch;
    size_t dst_idx = (y * w0 + x) * c + ch;
    output[dst_idx] = Transfer::PU::inverse(input[src_idx] * rcp_norm_scale);
}

__global__ static void cuda_concat_skip(const float* input, size_t c_input, 
                                        const float* skip, size_t c_skip,
                                        float* output, size_t h, size_t w) {
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

static void apply_convolutions(const Layer& layer, float*& d_input, size_t& h, size_t& w, size_t& c, 
                                const dim3& block, dim3& grid) {
    for (size_t i = 0; i < layer.num_convs; i++) {
        size_t out_c = layer.weights[i].out_channels;
        float* d_output;
        CUDA_ERR(cudaMalloc(&d_output, h * w * out_c * sizeof(float)));
        
        grid = dim3((h + block.x - 1) / block.x, (w + block.y - 1) / block.y, (out_c + block.z - 1) / block.z);
        conv_relu_nhwc_oihw_cuda<<<grid, block>>>(
            d_input, d_output, h, w, 3, 3, c, out_c,
            layer.weights[i].data.half_data,
            layer.biases[i].data.half_data);
        CUDA_ERR(cudaGetLastError());
        CUDA_ERR(cudaDeviceSynchronize());
        
        CUDA_ERR(cudaFree(d_input));
        d_input = d_output;
        c = out_c;
    }
}

static void apply_post_op(LayerPostOp post_op, float*& d_input, size_t& h, size_t& w, size_t c,
                          const dim3& block, dim3& grid) {
    if (post_op == LayerPostOp::MAX_POOL) {
        size_t out_h = h / 2;
        size_t out_w = w / 2;
        float* d_output;
        CUDA_ERR(cudaMalloc(&d_output, out_h * out_w * c * sizeof(float)));
        
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
        float* d_output;
        CUDA_ERR(cudaMalloc(&d_output, out_h * out_w * c * sizeof(float)));
        
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
    
    const float normScale = Transfer::compute_norm_scale();
    const float rcpNormScale = Transfer::compute_rcp_norm_scale();
    
    Layer* h_encoder_layers = new Layer[model.num_encoder_layers];
    Layer* h_decoder_layers = new Layer[model.num_decoder_layers];
    memcpy(h_encoder_layers, model.encoder_layers, model.num_encoder_layers * sizeof(Layer));
    memcpy(h_decoder_layers, model.decoder_layers, model.num_decoder_layers * sizeof(Layer));
    
    for (size_t i = 0; i < model.num_encoder_layers; i++) {
        size_t num_convs = h_encoder_layers[i].num_convs;
        TzaTensorStripped* h_weights = new TzaTensorStripped[num_convs];
        TzaTensorStripped* h_biases = new TzaTensorStripped[num_convs];
        CUDA_ERR(cudaMemcpy(h_weights, h_encoder_layers[i].weights, num_convs * sizeof(TzaTensorStripped), cudaMemcpyDeviceToHost));
        CUDA_ERR(cudaMemcpy(h_biases, h_encoder_layers[i].biases, num_convs * sizeof(TzaTensorStripped), cudaMemcpyDeviceToHost));
        h_encoder_layers[i].weights = h_weights;
        h_encoder_layers[i].biases = h_biases;
    }
    
    for (size_t i = 0; i < model.num_decoder_layers; i++) {
        size_t num_convs = h_decoder_layers[i].num_convs;
        TzaTensorStripped* h_weights = new TzaTensorStripped[num_convs];
        TzaTensorStripped* h_biases = new TzaTensorStripped[num_convs];
        CUDA_ERR(cudaMemcpy(h_weights, h_decoder_layers[i].weights, num_convs * sizeof(TzaTensorStripped), cudaMemcpyDeviceToHost));
        CUDA_ERR(cudaMemcpy(h_biases, h_decoder_layers[i].biases, num_convs * sizeof(TzaTensorStripped), cudaMemcpyDeviceToHost));
        h_decoder_layers[i].weights = h_weights;
        h_decoder_layers[i].biases = h_biases;
    }

    float* d_input_img;
    CUDA_ERR(cudaMalloc(&d_input_img, h0 * w0 * c0 * sizeof(float)));
    CUDA_ERR(cudaMemcpy(d_input_img, input_img.tensor.data(), h0 * w0 * c0 * sizeof(float), cudaMemcpyHostToDevice));
    
    float* d_input;
    CUDA_ERR(cudaMalloc(&d_input, h * w * c * sizeof(float)));
    
    dim3 block(8, 8, 4);
    dim3 grid((h + block.x - 1) / block.x, (w + block.y - 1) / block.y, (c + block.z - 1) / block.z);
    cuda_apply_hdr_transfer_function<<<grid, block>>>(d_input_img, h0, w0, c0, d_input, h, w, normScale);
    CUDA_ERR(cudaGetLastError());
    CUDA_ERR(cudaDeviceSynchronize());
    CUDA_ERR(cudaFree(d_input_img));
    
    float* d_original_input;
    CUDA_ERR(cudaMalloc(&d_original_input, h * w * c * sizeof(float)));
    CUDA_ERR(cudaMemcpy(d_original_input, d_input, h * w * c * sizeof(float), cudaMemcpyDeviceToDevice));
    
    float** d_encode_outputs = new float*[model.num_encoder_layers];
    
    for (size_t layer_idx = 0; layer_idx < model.num_encoder_layers; layer_idx++) {
        const Layer& layer = h_encoder_layers[layer_idx];
        
        apply_convolutions(layer, d_input, h, w, c, block, grid);
        apply_post_op(layer.post_op, d_input, h, w, c, block, grid);
        
        CUDA_ERR(cudaMalloc(&d_encode_outputs[layer_idx], h * w * c * sizeof(float)));
        CUDA_ERR(cudaMemcpy(d_encode_outputs[layer_idx], d_input, h * w * c * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    
    int skip_idx = 3;
    for (size_t layer_idx = 0; layer_idx < model.num_decoder_layers; layer_idx++) {
        const Layer& layer = h_decoder_layers[layer_idx];
        
        if (skip_idx >= 0) {
            const float* d_skip = (skip_idx == 0) ? d_original_input : d_encode_outputs[skip_idx];
            size_t skip_c = (skip_idx == 0) ? c0 : h_encoder_layers[skip_idx].weights->out_channels;
            
            float* d_concat;
            CUDA_ERR(cudaMalloc(&d_concat, h * w * (c + skip_c) * sizeof(float)));
            
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
        
        apply_convolutions(layer, d_input, h, w, c, block, grid);
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
    
    for (size_t i = 0; i < model.num_encoder_layers; i++) {
        delete[] h_encoder_layers[i].weights;
        delete[] h_encoder_layers[i].biases;
    }
    for (size_t i = 0; i < model.num_decoder_layers; i++) {
        delete[] h_decoder_layers[i].weights;
        delete[] h_decoder_layers[i].biases;
    }
    delete[] h_encoder_layers;
    delete[] h_decoder_layers;
}
