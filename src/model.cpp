#include "model.h"

UNetModel::UNetModel(const std::string& model_name, TzaFile& weights) : model_name(model_name) {
    if (model_name == "rt_hdr") {
        encoder_layers = std::vector<Layer> {
            {{{weights.find("enc_conv0.weight")->strip(), weights.find("enc_conv0.bias")->strip()}}, LayerPostOp::NONE},
            {{{weights.find("enc_conv1.weight")->strip(), weights.find("enc_conv1.bias")->strip()}}, LayerPostOp::MAX_POOL},
            {{{weights.find("enc_conv2.weight")->strip(), weights.find("enc_conv2.bias")->strip()}}, LayerPostOp::MAX_POOL},
            {{{weights.find("enc_conv3.weight")->strip(), weights.find("enc_conv3.bias")->strip()}}, LayerPostOp::MAX_POOL},
            {{{weights.find("enc_conv4.weight")->strip(), weights.find("enc_conv4.bias")->strip()}}, LayerPostOp::MAX_POOL}, 
            {{{weights.find("enc_conv5a.weight")->strip(), weights.find("enc_conv5a.bias")->strip()}, {weights.find("enc_conv5b.weight")->strip(), weights.find("enc_conv5b.bias")->strip()}}, LayerPostOp::NN_UPSAMPLE}
        };
        decoder_layers = std::vector<Layer> {
            {{{weights.find("dec_conv4a.weight")->strip(), weights.find("dec_conv4a.bias")->strip()}, {weights.find("dec_conv4b.weight")->strip(), weights.find("dec_conv4b.bias")->strip()}}, LayerPostOp::NN_UPSAMPLE},
            {{{weights.find("dec_conv3a.weight")->strip(), weights.find("dec_conv3a.bias")->strip()}, {weights.find("dec_conv3b.weight")->strip(), weights.find("dec_conv3b.bias")->strip()}}, LayerPostOp::NN_UPSAMPLE},
            {{{weights.find("dec_conv2a.weight")->strip(), weights.find("dec_conv2a.bias")->strip()}, {weights.find("dec_conv2b.weight")->strip(), weights.find("dec_conv2b.bias")->strip()}}, LayerPostOp::NN_UPSAMPLE},
            {{{weights.find("dec_conv1a.weight")->strip(), weights.find("dec_conv1a.bias")->strip()}, {weights.find("dec_conv1b.weight")->strip(), weights.find("dec_conv1b.bias")->strip()}}, LayerPostOp::NONE},
            {{{weights.find("dec_conv0.weight")->strip(), weights.find("dec_conv0.bias")->strip()}}, LayerPostOp::NONE}
        };
    }
}