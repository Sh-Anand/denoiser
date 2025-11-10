#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <stdio.h>
#include <utility>

#include "conv.h"
#include "exr.h"
#include "layer.h"
#include "model.h"
#include "unet.h"
#include "transfer.h"

void oidn_unet(EXR::Image& input_img,
               UNetModel& model,
               float*& output_img) {

    const size_t h0 = input_img.height;
    const size_t w0 = input_img.width;
    const size_t c0 = input_img.loaded_channels.size();
    
    constexpr size_t alignment = 16;
    const size_t h_padded = ((h0 + alignment - 1) / alignment) * alignment;
    const size_t w_padded = ((w0 + alignment - 1) / alignment) * alignment;

    std::vector<std::unique_ptr<float[]>> encode_outputs;

    size_t h = h_padded;
    size_t w = w_padded;
    size_t c = c0;
    
    // Apply PU transfer function to input (for HDR)
    const float normScale = Transfer::compute_norm_scale();
    const float rcpNormScale = Transfer::compute_rcp_norm_scale();
    
    auto input = std::make_unique<float[]>(h * w * c);
    std::fill_n(input.get(), h * w * c, 0.0f);
    #pragma omp parallel for
    for (size_t y = 0; y < h0; y++) {
        for (size_t x = 0; x < w0; x++) {
            for (size_t ch = 0; ch < c; ch++) {
                size_t src_idx = (y * w0 + x) * c + ch;
                size_t dst_idx = (y * w + x) * c + ch;
                float val = input_img.tensor[src_idx];
                input[dst_idx] = Transfer::PU::forward(val) * normScale;
            }
        }
    }
    
    auto original_input = std::make_unique<float[]>(h * w * c);
    std::copy(input.get(), input.get() + h * w * c, original_input.get());
    
    for (size_t layer_idx = 0; layer_idx < model.num_encoder_layers; layer_idx++) {
        const auto& [layer, post_op] = model.encoder_layers[layer_idx];
        std::unique_ptr<float[]> output;
        for (auto [weights, bias]: layer) {
            size_t out_c = weights->dims[0];
            output = std::make_unique<float[]>(h * w * out_c);
            conv_relu_nhwc_oihw(input.get(), output.get(), h, w, 3, 3, c, out_c, 
                                reinterpret_cast<const _Float16*>(weights->data.data()), 
                                reinterpret_cast<const _Float16*>(bias->data.data()));
            c = out_c;
            input = std::move(output);
        }
        
        if (post_op == LayerPostOp::MAX_POOL) {
            output = std::make_unique<float[]>(h * w * c);
            max_pool_nhwc(input.get(), output.get(), h, w, c);
            input = std::move(output);
            h = h / 2;
            w = w / 2;
        } else if (post_op == LayerPostOp::NN_UPSAMPLE) {
            output = std::make_unique<float[]>(h * w * c * 4);
            nn_upsample_nhwc(input.get(), output.get(), h, w, c);
            input = std::move(output);
            h = h * 2;
            w = w * 2;
        }
        
        auto saved = std::make_unique<float[]>(h * w * c);
        std::copy(input.get(), input.get() + h * w * c, saved.get());
        encode_outputs.push_back(std::move(saved));
    }
    
    int skip_idx = 3;
    for (size_t layer_idx = 0; layer_idx < model.num_decoder_layers; layer_idx++) {
        const auto& [layer, post_op] = model.decoder_layers[layer_idx];
        std::unique_ptr<float[]> output;
        if (skip_idx >= 0) {
            const auto& skip = (skip_idx == 3) ? encode_outputs[3] : 
                               (skip_idx == 2) ? encode_outputs[2] :
                               (skip_idx == 1) ? encode_outputs[1] : original_input;
            size_t skip_c = (skip_idx == 3) ? model.encoder_layers[3].first.back().first->dims[0] :
                            (skip_idx == 2) ? model.encoder_layers[2].first.back().first->dims[0] :
                            (skip_idx == 1) ? model.encoder_layers[1].first.back().first->dims[0] : c0;
            auto concat = std::make_unique<float[]>(h * w * (c + skip_c));
            #pragma omp parallel for
            for (size_t i = 0; i < h * w; i++) {
                std::copy_n(input.get() + i * c, c, concat.get() + i * (c + skip_c));
                std::copy_n(skip.get() + i * skip_c, skip_c, concat.get() + i * (c + skip_c) + c);
            }
            c = c + skip_c;
            input = std::move(concat);
            skip_idx--;
        }
        for (auto [weights, bias]: layer) {
            size_t out_c = weights->dims[0];
            output = std::make_unique<float[]>(h * w * out_c);
            conv_relu_nhwc_oihw(input.get(), output.get(), h, w, 3, 3, c, out_c, 
                                reinterpret_cast<const _Float16*>(weights->data.data()), 
                                reinterpret_cast<const _Float16*>(bias->data.data()));
            c = out_c;
            input = std::move(output);
        }
        if (post_op == LayerPostOp::NN_UPSAMPLE) {
            output = std::make_unique<float[]>(h * w * c * 4);
            nn_upsample_nhwc(input.get(), output.get(), h, w, c);
            input = std::move(output);
            h = h * 2;
            w = w * 2;
        }
    }
    
    auto output = std::make_unique<float[]>(h0 * w0 * c);
    
    // Apply inverse PU transfer function to output (for HDR)
    #pragma omp parallel for
    for (size_t y = 0; y < h0; y++) {
        for (size_t x = 0; x < w0; x++) {
            for (size_t ch = 0; ch < c; ch++) {
                size_t src_idx = (y * w + x) * c + ch;
                size_t dst_idx = (y * w0 + x) * c + ch;
                float val = input[src_idx];
                output[dst_idx] = Transfer::PU::inverse(val * rcpNormScale);
            }
        }
    }

    output_img = output.release();
}
