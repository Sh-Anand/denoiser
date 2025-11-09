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

void oidn_unet(EXR& input_img,
               TzaFile& weights,
               EXR& output_img) {

    const size_t h0 = input_img.height;
    const size_t w0 = input_img.width;
    const size_t c0 = input_img.loaded_channels.size();
    
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
        {{{weights.find("dec_conv1a.weight"), weights.find("dec_conv1a.bias")}, {weights.find("dec_conv1b.weight"), weights.find("dec_conv1b.bias")}}, LayerPostOp::NN_UPSAMPLE},
        {{{weights.find("dec_conv0.weight"), weights.find("dec_conv0.bias")}}, LayerPostOp::NONE}
    };

    // encode stage
    std::vector<std::unique_ptr<float[]>> encode_outputs;

    size_t h = h0;
    size_t w = w0;
    size_t c = c0;
    auto input = std::make_unique<float[]>(h * w * c);
    std::copy(input_img.tensor.begin(), input_img.tensor.end(), input.get());
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
            h = h * 0.5;
            w = w * 0.5;
        } else if (post_op == LayerPostOp::NN_UPSAMPLE) {
            output = std::make_unique<float[]>(h * w * c);
            nn_upsample_nhwc(input.get(), output.get(), h, w, c);
            input = std::move(output);
            h = h * 2;
            w = w * 2;
        }
    }
}
