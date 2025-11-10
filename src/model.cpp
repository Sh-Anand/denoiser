#include "model.h"

UNetModel::UNetModel(const std::string& model_name, TzaFile& weights) {

    std::vector<std::vector<TzaTensorStripped>> encoder_weights;
    std::vector<std::vector<TzaTensorStripped>> encoder_biases;
    std::vector<LayerPostOp> encoder_post_ops;
    std::vector<std::vector<TzaTensorStripped>> decoder_weights;
    std::vector<std::vector<TzaTensorStripped>> decoder_biases;
    std::vector<LayerPostOp> decoder_post_ops;

    if (model_name == "rt_hdr") {
        encoder_weights = {
            {weights.find("enc_conv0.weight")->strip()},
            {weights.find("enc_conv1.weight")->strip()},
            {weights.find("enc_conv2.weight")->strip()},
            {weights.find("enc_conv3.weight")->strip()},
            {weights.find("enc_conv4.weight")->strip()},
            {weights.find("enc_conv5a.weight")->strip(), weights.find("enc_conv5b.weight")->strip()}
        };

        encoder_biases = {
            {weights.find("enc_conv0.bias")->strip()},
            {weights.find("enc_conv1.bias")->strip()},
            {weights.find("enc_conv2.bias")->strip()},
            {weights.find("enc_conv3.bias")->strip()},
            {weights.find("enc_conv4.bias")->strip()},
            {weights.find("enc_conv5a.bias")->strip(), weights.find("enc_conv5b.bias")->strip()}
        };

        encoder_post_ops = {
            LayerPostOp::NONE,
            LayerPostOp::MAX_POOL,
            LayerPostOp::MAX_POOL,
            LayerPostOp::MAX_POOL,
            LayerPostOp::MAX_POOL,
            LayerPostOp::NN_UPSAMPLE
        };

        decoder_weights = {
            {weights.find("dec_conv4a.weight")->strip(), weights.find("dec_conv4b.weight")->strip()},
            {weights.find("dec_conv3a.weight")->strip(), weights.find("dec_conv3b.weight")->strip()},
            {weights.find("dec_conv2a.weight")->strip(), weights.find("dec_conv2b.weight")->strip()},
            {weights.find("dec_conv1a.weight")->strip(), weights.find("dec_conv1b.weight")->strip()},
            {weights.find("dec_conv0.weight")->strip()}
        };
        
        decoder_biases = {
            {weights.find("dec_conv4a.bias")->strip(), weights.find("dec_conv4b.bias")->strip()},
            {weights.find("dec_conv3a.bias")->strip(), weights.find("dec_conv3b.bias")->strip()},
            {weights.find("dec_conv2a.bias")->strip(), weights.find("dec_conv2b.bias")->strip()},
            {weights.find("dec_conv1a.bias")->strip(), weights.find("dec_conv1b.bias")->strip()},
            {weights.find("dec_conv0.bias")->strip()}
        };

        decoder_post_ops = {
            LayerPostOp::NN_UPSAMPLE,
            LayerPostOp::NN_UPSAMPLE,
            LayerPostOp::NN_UPSAMPLE,
            LayerPostOp::NONE,
            LayerPostOp::NONE
        };
    }

    encoder_layers = new Layer[encoder_weights.size()];
    for (size_t i = 0; i < encoder_weights.size(); i++) {
        encoder_layers[i] = {encoder_weights.at(i).data(), encoder_biases.at(i).data(), encoder_weights.at(i).size(), encoder_post_ops.at(i)};
    }
    decoder_layers = new Layer[decoder_weights.size()];
    for (size_t i = 0; i < decoder_weights.size(); i++) {
        decoder_layers[i] = {decoder_weights.at(i).data(), decoder_biases.at(i).data(), decoder_weights.at(i).size(), decoder_post_ops.at(i)};
    }
    num_encoder_layers = encoder_weights.size();
    num_decoder_layers = decoder_weights.size();
}