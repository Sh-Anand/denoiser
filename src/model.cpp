#include "model.h"

// NOTE: massive assumption that the weights will remain in scope throughout the model's lifetime
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

    // Flatten and allocate storage
    size_t total_encoder_weights = 0, total_encoder_biases = 0;
    size_t total_decoder_weights = 0, total_decoder_biases = 0;
    for (const auto& w : encoder_weights) total_encoder_weights += w.size();
    for (const auto& b : encoder_biases) total_encoder_biases += b.size();
    for (const auto& w : decoder_weights) total_decoder_weights += w.size();
    for (const auto& b : decoder_biases) total_decoder_biases += b.size();
    
    encoder_weights_storage = new TzaTensorStripped[total_encoder_weights];
    encoder_biases_storage = new TzaTensorStripped[total_encoder_biases];
    decoder_weights_storage = new TzaTensorStripped[total_decoder_weights];
    decoder_biases_storage = new TzaTensorStripped[total_decoder_biases];
    
    size_t offset = 0;
    encoder_layers = new Layer[encoder_weights.size()];
    for (size_t i = 0; i < encoder_weights.size(); i++) {
        for (size_t j = 0; j < encoder_weights[i].size(); j++) {
            encoder_weights_storage[offset + j] = encoder_weights[i][j];
            encoder_biases_storage[offset + j] = encoder_biases[i][j];
        }
        encoder_layers[i] = {&encoder_weights_storage[offset], &encoder_biases_storage[offset], encoder_weights[i].size(), encoder_post_ops[i]};
        offset += encoder_weights[i].size();
    }
    
    offset = 0;
    decoder_layers = new Layer[decoder_weights.size()];
    for (size_t i = 0; i < decoder_weights.size(); i++) {
        for (size_t j = 0; j < decoder_weights[i].size(); j++) {
            decoder_weights_storage[offset + j] = decoder_weights[i][j];
            decoder_biases_storage[offset + j] = decoder_biases[i][j];
        }
        decoder_layers[i] = {&decoder_weights_storage[offset], &decoder_biases_storage[offset], decoder_weights[i].size(), decoder_post_ops[i]};
        offset += decoder_weights[i].size();
    }
    
    num_encoder_layers = encoder_weights.size();
    num_decoder_layers = decoder_weights.size();
}

UNetModel::~UNetModel() {
    delete[] encoder_layers;
    delete[] decoder_layers;
    delete[] encoder_weights_storage;
    delete[] encoder_biases_storage;
    delete[] decoder_weights_storage;
    delete[] decoder_biases_storage;
}