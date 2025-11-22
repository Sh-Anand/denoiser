#ifndef MODEL_SPEC_H
#define MODEL_SPEC_H

#include "model/model.h"

#ifndef __CUDACC__
struct dim3 {
    int x;
    int y;
    int z;
    dim3(int x, int y, int z) : x(x), y(y), z(z) {}
};
#endif

static void get_model_spec(const std::string& model_name, TzaFile& weights,
                           std::vector<std::vector<const TzaTensor*>>& layers,
                           std::vector<LayerPostOp>& post_ops,
                           std::vector<std::vector<dim3>>& block_dims,
                           std::vector<std::vector<int>>& conv_im2col_tile_ks,
                           int& decoder_layer_offset) {

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
    } else if (model_name == "rt_hdr_alb_norm") {

    }
}

#endif // MODEL_SPEC_H