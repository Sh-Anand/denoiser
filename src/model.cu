#include "cuda_defs.h"
#include "model.h"

// NOTE: massive assumption that the weights will remain in scope throughout the model's lifetime
// If cuda, assume weights are gpu virtual addresses
UNetModel createUNetModel(const std::string& model_name, TzaFile& weights, bool cuda) {
    UNetModel model;
    std::vector<std::vector<const TzaTensor*>> layers;
    std::vector<LayerPostOp> post_ops;
    std::vector<std::vector<dim3>> block_dims;
    std::vector<std::vector<int>> conv_im2col_tile_ks;
    int decoder_layer_offset = 0;
    auto reorder_oihw_to_krsc = [](const auto* src, auto* dst, uint32_t K, uint32_t C, uint32_t R, uint32_t S) {
        for (uint32_t k = 0; k < K; ++k) {
            for (uint32_t r = 0; r < R; ++r) {
                for (uint32_t s = 0; s < S; ++s) {
                    for (uint32_t c = 0; c < C; ++c) {
                        const uint32_t src_idx = (((k * C + c) * R + r) * S) + s;
                        const uint32_t dst_idx = (((k * R + r) * S + s) * C) + c;
                        dst[dst_idx] = src[src_idx];
                    }
                }
            }
        }
    };
    if (model_name == "rt_hdr") {
        layers = {
            {weights.find("enc_conv0.weight"), weights.find("enc_conv0.bias")},
            {weights.find("enc_conv1.weight"), weights.find("enc_conv1.bias")},
            {weights.find("enc_conv2.weight"), weights.find("enc_conv2.bias")},
            {weights.find("enc_conv3.weight"), weights.find("enc_conv3.bias")},
            {weights.find("enc_conv4.weight"), weights.find("enc_conv4.bias")},
            {weights.find("enc_conv5a.weight"), weights.find("enc_conv5a.bias"), weights.find("enc_conv5b.weight"), weights.find("enc_conv5b.bias")},
            {weights.find("dec_conv4a.weight"), weights.find("dec_conv4a.bias"), weights.find("dec_conv4b.weight"), weights.find("dec_conv4b.bias")},
            {weights.find("dec_conv3a.weight"), weights.find("dec_conv3a.bias"), weights.find("dec_conv3b.weight"), weights.find("dec_conv3b.bias")},
            {weights.find("dec_conv2a.weight"), weights.find("dec_conv2a.bias"), weights.find("dec_conv2b.weight"), weights.find("dec_conv2b.bias")},
            {weights.find("dec_conv1a.weight"), weights.find("dec_conv1a.bias"), weights.find("dec_conv1b.weight"), weights.find("dec_conv1b.bias")},
            {weights.find("dec_conv0.weight"), weights.find("dec_conv0.bias")}
        };

        post_ops = {
            LayerPostOp::NONE,
            LayerPostOp::MAX_POOL,
            LayerPostOp::MAX_POOL,
            LayerPostOp::MAX_POOL,
            LayerPostOp::MAX_POOL,
            LayerPostOp::NN_UPSAMPLE,
            LayerPostOp::NN_UPSAMPLE,
            LayerPostOp::NN_UPSAMPLE,
            LayerPostOp::NN_UPSAMPLE,
            LayerPostOp::NONE,
            LayerPostOp::NONE
        };

        block_dims = {
            {dim3(4, 2, 32)},
            {dim3(2, 8, 16)},
            {dim3(2, 8, 16)},
            {dim3(4, 4, 16)},
            {dim3(8, 2, 16)},
            {dim3(2, 8, 16), dim3(2, 8, 16)},
            {dim3(2, 8, 16), dim3(2, 8, 16)},
            {dim3(2, 8, 16), dim3(2, 8, 16)},
            {dim3(2, 12, 32), dim3(4, 4, 32)},
            {dim3(2, 12, 32), dim3(2, 8, 32)},
            {dim3(16, 2, 4)}
        };

        conv_im2col_tile_ks = {
            {16},
            {8},
            {8},
            {8},
            {8},
            {8, 8},
            {8, 8},
            {8, 8},
            {8, 8},
            {8, 8},
            {8}
        };

        decoder_layer_offset = 6;
    }

    size_t total_elements = 0;
    for (size_t i = 0; i < layers.size(); i++) {
        for (size_t j = 0; j < layers[i].size(); j++) {
            total_elements += layers[i][j]->elementCount();
        }
    }

    model.weights = new TensorData;
    if (cuda) {
        half* d_weights;
        CUDA_ERR(cudaMalloc((void**)&d_weights, total_elements * sizeof(half)));
        model.weights->half_data = d_weights;
    } else {
        model.weights->float16_data = new _Float16[total_elements];
    }

    size_t offset = 0;
    model.encoder_layers.resize(decoder_layer_offset);
    model.decoder_layers.resize(layers.size() - decoder_layer_offset);

    for (int i = 0; i < layers.size(); i++) {
        int decoder_layer_idx = i - decoder_layer_offset;

        if (decoder_layer_idx >= 0) {
            model.decoder_layers[decoder_layer_idx].weight_idxs = new uint32_t[layers[i].size()];
            model.decoder_layers[decoder_layer_idx].bias_idxs = new uint32_t[layers[i].size()];
            model.decoder_layers[decoder_layer_idx].out_channels = new uint32_t[layers[i].size()];
            model.decoder_layers[decoder_layer_idx].num_convs = layers[i].size()/2;
            model.decoder_layers[decoder_layer_idx].post_op = post_ops[i];
        } else {
            model.encoder_layers[i].weight_idxs = new uint32_t[layers[i].size()];
            model.encoder_layers[i].bias_idxs = new uint32_t[layers[i].size()];
            model.encoder_layers[i].out_channels = new uint32_t[layers[i].size()];
            model.encoder_layers[i].num_convs = layers[i].size()/2;
            model.encoder_layers[i].post_op = post_ops[i];
        }
        for (int j = 0; j < layers[i].size()/2; j++) {
            int weight_idx = j*2, bias_idx = j*2+1;
            if (decoder_layer_idx >= 0)
                model.decoder_layers[decoder_layer_idx].weight_idxs[j] = offset;
            else
                model.encoder_layers[i].weight_idxs[j] = offset;
            const TzaTensor* weight_tensor = layers[i][weight_idx];
            const bool is_conv_weight = weight_tensor->dims.size() == 4;
            const uint32_t K = is_conv_weight ? weight_tensor->dims[0] : 0;
            const uint32_t C = is_conv_weight ? weight_tensor->dims[1] : 0;
            const uint32_t R = is_conv_weight ? weight_tensor->dims[2] : 0;
            const uint32_t S = is_conv_weight ? weight_tensor->dims[3] : 0;
            if (cuda) {
                if (is_conv_weight) {
                    std::vector<half> reordered(weight_tensor->elementCount());
                    const half* src = reinterpret_cast<const half*>(weight_tensor->data.data());
                    reorder_oihw_to_krsc(src, reordered.data(), K, C, R, S);
                    CUDA_ERR(cudaMemcpy((void *)(model.weights->half_data + offset), reordered.data(), reordered.size() * sizeof(half), cudaMemcpyHostToDevice));
                } else {
                    CUDA_ERR(cudaMemcpy((void *)(model.weights->half_data + offset), weight_tensor->data.data(), weight_tensor->elementCount() * sizeof(half), cudaMemcpyHostToDevice));
                }
            } else {
                if (is_conv_weight) {
                    std::vector<_Float16> reordered(weight_tensor->elementCount());
                    const _Float16* src = reinterpret_cast<const _Float16*>(weight_tensor->data.data());
                    reorder_oihw_to_krsc(src, reordered.data(), K, C, R, S);
                    memcpy((void *)(model.weights->float16_data + offset), reordered.data(), reordered.size() * sizeof(_Float16));
                } else {
                    memcpy((void *)(model.weights->float16_data + offset), weight_tensor->data.data(), weight_tensor->elementCount() * sizeof(_Float16));
                }
            }

            offset += layers[i][weight_idx]->elementCount();
            if (decoder_layer_idx >= 0)
                model.decoder_layers[decoder_layer_idx].bias_idxs[j] = offset;
            else
                model.encoder_layers[i].bias_idxs[j] = offset;
            if (cuda) {
                CUDA_ERR(cudaMemcpy((void *)(model.weights->half_data + offset), layers[i][bias_idx]->data.data(), layers[i][bias_idx]->elementCount() * sizeof(half), cudaMemcpyHostToDevice));
            } else {
                memcpy((void *)(model.weights->float16_data + offset), layers[i][bias_idx]->data.data(), layers[i][bias_idx]->elementCount() * sizeof(_Float16));
            }
            offset += layers[i][bias_idx]->elementCount();
            if (decoder_layer_idx >= 0)
                model.decoder_layers[decoder_layer_idx].out_channels[j] = layers[i][weight_idx]->dims[0];
            else
                model.encoder_layers[i].out_channels[j] = layers[i][weight_idx]->dims[0];

            if (cuda) {
                if (decoder_layer_idx >= 0) {
                    model.decoder_layers[decoder_layer_idx].block_dims.push_back(block_dims[i][j]);
                    model.decoder_layers[decoder_layer_idx].conv_im2col_tile_ks.push_back(conv_im2col_tile_ks[i][j]);
                } else {
                    model.encoder_layers[i].block_dims.push_back(block_dims[i][j]);
                    model.encoder_layers[i].conv_im2col_tile_ks.push_back(conv_im2col_tile_ks[i][j]);
                }
            }
        }
    }
    return model;
}

void freeUNetModel(UNetModel& model, bool cuda) {
    // Free layer-specific arrays
    for (size_t i = 0; i < model.encoder_layers.size(); i++) {
        delete[] model.encoder_layers[i].weight_idxs;
        delete[] model.encoder_layers[i].bias_idxs;
        delete[] model.encoder_layers[i].out_channels;
    }
    for (size_t i = 0; i < model.decoder_layers.size(); i++) {
        delete[] model.decoder_layers[i].weight_idxs;
        delete[] model.decoder_layers[i].bias_idxs;
        delete[] model.decoder_layers[i].out_channels;
    }
    
    // Free weight buffer
    if (cuda) {
        CUDA_ERR(cudaFree((void*)model.weights->half_data));
    } else {
        delete[] model.weights->float16_data;
    }
    delete model.weights;
}
