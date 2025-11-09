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
#include "tza.h"
#include "unet.h"

void oidn_unet(EXR::Image& input_img,
               TzaFile& weights,
               EXR::Image& output_img) {

    const size_t h0 = input_img.height;
    const size_t w0 = input_img.width;
    const size_t c0 = input_img.loaded_channels.size();
    
    constexpr size_t alignment = 16;
    const size_t h_padded = ((h0 + alignment - 1) / alignment) * alignment;
    const size_t w_padded = ((w0 + alignment - 1) / alignment) * alignment;
    
    std::vector<Layer> encoder_layers = {
        {{{weights.find("enc_conv0.weight"), weights.find("enc_conv0.bias")}}, LayerPostOp::NONE},
        {{{weights.find("enc_conv1.weight"), weights.find("enc_conv1.bias")}}, LayerPostOp::MAX_POOL},
        {{{weights.find("enc_conv2.weight"), weights.find("enc_conv2.bias")}}, LayerPostOp::MAX_POOL},
        {{{weights.find("enc_conv3.weight"), weights.find("enc_conv3.bias")}}, LayerPostOp::MAX_POOL},
        {{{weights.find("enc_conv4.weight"), weights.find("enc_conv4.bias")}}, LayerPostOp::MAX_POOL}, 
        {{{weights.find("enc_conv5a.weight"), weights.find("enc_conv5a.bias")}, {weights.find("enc_conv5b.weight"), weights.find("enc_conv5b.bias")}}, LayerPostOp::NN_UPSAMPLE}
    };

    std::vector<Layer> decoder_layers = {
        {{{weights.find("dec_conv4a.weight"), weights.find("dec_conv4a.bias")}, {weights.find("dec_conv4b.weight"), weights.find("dec_conv4b.bias")}}, LayerPostOp::NN_UPSAMPLE},
        {{{weights.find("dec_conv3a.weight"), weights.find("dec_conv3a.bias")}, {weights.find("dec_conv3b.weight"), weights.find("dec_conv3b.bias")}}, LayerPostOp::NN_UPSAMPLE},
        {{{weights.find("dec_conv2a.weight"), weights.find("dec_conv2a.bias")}, {weights.find("dec_conv2b.weight"), weights.find("dec_conv2b.bias")}}, LayerPostOp::NN_UPSAMPLE},
        {{{weights.find("dec_conv1a.weight"), weights.find("dec_conv1a.bias")}, {weights.find("dec_conv1b.weight"), weights.find("dec_conv1b.bias")}}, LayerPostOp::NONE},
        {{{weights.find("dec_conv0.weight"), weights.find("dec_conv0.bias")}}, LayerPostOp::NONE}
    };

    std::vector<std::unique_ptr<float[]>> encode_outputs;

    size_t h = h_padded;
    size_t w = w_padded;
    size_t c = c0;
    
    auto input = std::make_unique<float[]>(h * w * c);
    std::fill_n(input.get(), h * w * c, 0.0f);
    #pragma omp parallel for
    for (size_t y = 0; y < h0; y++) {
        std::copy_n(&input_img.tensor[y * w0 * c], w0 * c, input.get() + y * w * c);
    }
    
    auto original_input = std::make_unique<float[]>(h * w * c);
    std::copy(input.get(), input.get() + h * w * c, original_input.get());
    
    for (const auto& [layer, post_op]: encoder_layers) {
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
        auto saved = std::make_unique<float[]>(h * w * c);
        std::copy(input.get(), input.get() + h * w * c, saved.get());
        encode_outputs.push_back(std::move(saved));

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
    }
    
    int skip_idx = 3;
    for (const auto& [layer, post_op]: decoder_layers) {
        std::unique_ptr<float[]> output;
        if (skip_idx >= 0) {
            const auto& skip = (skip_idx == 3) ? encode_outputs[3] : 
                               (skip_idx == 2) ? encode_outputs[2] :
                               (skip_idx == 1) ? encode_outputs[1] : original_input;
            size_t skip_c = (skip_idx == 3) ? encoder_layers[3].first.back().first->dims[0] :
                            (skip_idx == 2) ? encoder_layers[2].first.back().first->dims[0] :
                            (skip_idx == 1) ? encoder_layers[1].first.back().first->dims[0] : c0;
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
    
    output_img.width = w0;
    output_img.height = h0;
    output_img.tensor.resize(h0 * w0 * c);
    #pragma omp parallel for
    for (size_t y = 0; y < h0; y++) {
        std::copy_n(input.get() + y * w * c, w0 * c, &output_img.tensor[y * w0 * c]);
    }
}
