#include "model.h"

UNetModel::UNetModel(const std::string& model_name, TzaFile& weights) : model_name(model_name) {
    if (model_name == "rt_hdr") {
        encoder_layers = std::vector<Layer> {
            {{{weights.find("enc_conv0.weight"), weights.find("enc_conv0.bias")}}, LayerPostOp::NONE},
            {{{weights.find("enc_conv1.weight"), weights.find("enc_conv1.bias")}}, LayerPostOp::MAX_POOL},
            {{{weights.find("enc_conv2.weight"), weights.find("enc_conv2.bias")}}, LayerPostOp::MAX_POOL},
            {{{weights.find("enc_conv3.weight"), weights.find("enc_conv3.bias")}}, LayerPostOp::MAX_POOL},
            {{{weights.find("enc_conv4.weight"), weights.find("enc_conv4.bias")}}, LayerPostOp::MAX_POOL}, 
            {{{weights.find("enc_conv5a.weight"), weights.find("enc_conv5a.bias")}, {weights.find("enc_conv5b.weight"), weights.find("enc_conv5b.bias")}}, LayerPostOp::NN_UPSAMPLE}
        };
        decoder_layers = std::vector<Layer> {
            {{{weights.find("dec_conv4a.weight"), weights.find("dec_conv4a.bias")}, {weights.find("dec_conv4b.weight"), weights.find("dec_conv4b.bias")}}, LayerPostOp::NN_UPSAMPLE},
            {{{weights.find("dec_conv3a.weight"), weights.find("dec_conv3a.bias")}, {weights.find("dec_conv3b.weight"), weights.find("dec_conv3b.bias")}}, LayerPostOp::NN_UPSAMPLE},
            {{{weights.find("dec_conv2a.weight"), weights.find("dec_conv2a.bias")}, {weights.find("dec_conv2b.weight"), weights.find("dec_conv2b.bias")}}, LayerPostOp::NN_UPSAMPLE},
            {{{weights.find("dec_conv1a.weight"), weights.find("dec_conv1a.bias")}, {weights.find("dec_conv1b.weight"), weights.find("dec_conv1b.bias")}}, LayerPostOp::NONE},
            {{{weights.find("dec_conv0.weight"), weights.find("dec_conv0.bias")}}, LayerPostOp::NONE}
        };
    }
}