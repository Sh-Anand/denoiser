#include <algorithm>
#include <memory>

#include "conv.h"
#include "exr.h"
#include "layer.h"
#include "model.h"
#include "unet.h"
#include "transfer.h"

static void apply_convolutions(const Layer& layer, const UNetModel& model, std::unique_ptr<float[]>& input, 
                                size_t h, size_t w, size_t& c) {
    for (size_t i = 0; i < layer.num_convs; i++) {
        size_t out_c = layer.out_channels[i];
        auto output = std::make_unique<float[]>(h * w * out_c);
        conv_relu_nhwc_oihw(input.get(), output.get(), h, w, c, out_c, 
                            model.weights->float16_data + layer.weight_idxs[i],
                            model.weights->float16_data + layer.bias_idxs[i]);
        c = out_c;
        input = std::move(output);
    }
}

static void apply_post_op(LayerPostOp post_op, std::unique_ptr<float[]>& input,
                          size_t& h, size_t& w, size_t c) {
    if (post_op == LayerPostOp::MAX_POOL) {
        size_t out_h = h / 2;
        size_t out_w = w / 2;
        auto output = std::make_unique<float[]>(out_h * out_w * c);
        max_pool_nhwc(input.get(), output.get(), h, w, c);
        input = std::move(output);
        h = out_h;
        w = out_w;
    } else if (post_op == LayerPostOp::NN_UPSAMPLE) {
        size_t out_h = h * 2;
        size_t out_w = w * 2;
        auto output = std::make_unique<float[]>(out_h * out_w * c);
        nn_upsample_nhwc(input.get(), output.get(), h, w, c);
        input = std::move(output);
        h = out_h;
        w = out_w;
    }
}

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
    
    for (size_t layer_idx = 0; layer_idx < model.encoder_layers.size(); layer_idx++) {
        const auto& layer = model.encoder_layers[layer_idx];
        
        apply_convolutions(layer, model, input, h, w, c);
        apply_post_op(layer.post_op, input, h, w, c);
        
        auto saved = std::make_unique<float[]>(h * w * c);
        std::copy(input.get(), input.get() + h * w * c, saved.get());
        encode_outputs.push_back(std::move(saved));
    }
    
    int skip_idx = 3;
    for (size_t layer_idx = 0; layer_idx < model.decoder_layers.size(); layer_idx++) {
        const auto& layer = model.decoder_layers[layer_idx];
        
        if (skip_idx >= 0) {
            const auto& skip = (skip_idx == 0) ?  original_input : encode_outputs[skip_idx];
            size_t skip_c = (skip_idx == 0) ? c0 : model.encoder_layers[skip_idx].out_channels[0]; // TODO: hack
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
        
        apply_convolutions(layer, model, input, h, w, c);
        apply_post_op(layer.post_op, input, h, w, c);
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
