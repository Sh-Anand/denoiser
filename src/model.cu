#include "model.h"

// NOTE: massive assumption that the weights will remain in scope throughout the model's lifetime
// If cuda, assume weights are gpu virtual addresses
UNetModel createUNetModel(const std::string& model_name, TzaFile& weights, bool cuda) {
    UNetModel model;
    std::vector<std::vector<TzaTensorStripped>> encoder_weights;
    std::vector<std::vector<TzaTensorStripped>> encoder_biases;
    std::vector<LayerPostOp> encoder_post_ops;
    std::vector<std::vector<TzaTensorStripped>> decoder_weights;
    std::vector<std::vector<TzaTensorStripped>> decoder_biases;
    std::vector<LayerPostOp> decoder_post_ops;
    
    if (model_name == "rt_hdr") {
        encoder_weights = {
            {weights.find("enc_conv0.weight")->strip(cuda)},
            {weights.find("enc_conv1.weight")->strip(cuda)},
            {weights.find("enc_conv2.weight")->strip(cuda)},
            {weights.find("enc_conv3.weight")->strip(cuda)},
            {weights.find("enc_conv4.weight")->strip(cuda)},
            {weights.find("enc_conv5a.weight")->strip(cuda), weights.find("enc_conv5b.weight")->strip(cuda)}
        };

        encoder_biases = {
            {weights.find("enc_conv0.bias")->strip(cuda)},
            {weights.find("enc_conv1.bias")->strip(cuda)},
            {weights.find("enc_conv2.bias")->strip(cuda)},
            {weights.find("enc_conv3.bias")->strip(cuda)},
            {weights.find("enc_conv4.bias")->strip(cuda)},
            {weights.find("enc_conv5a.bias")->strip(cuda), weights.find("enc_conv5b.bias")->strip(cuda)}
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
            {weights.find("dec_conv4a.weight")->strip(cuda), weights.find("dec_conv4b.weight")->strip(cuda)},
            {weights.find("dec_conv3a.weight")->strip(cuda), weights.find("dec_conv3b.weight")->strip(cuda)},
            {weights.find("dec_conv2a.weight")->strip(cuda), weights.find("dec_conv2b.weight")->strip(cuda)},
            {weights.find("dec_conv1a.weight")->strip(cuda), weights.find("dec_conv1b.weight")->strip(cuda)},
            {weights.find("dec_conv0.weight")->strip(cuda)}
        };
        
        decoder_biases = {
            {weights.find("dec_conv4a.bias")->strip(cuda), weights.find("dec_conv4b.bias")->strip(cuda)},
            {weights.find("dec_conv3a.bias")->strip(cuda), weights.find("dec_conv3b.bias")->strip(cuda)},
            {weights.find("dec_conv2a.bias")->strip(cuda), weights.find("dec_conv2b.bias")->strip(cuda)},
            {weights.find("dec_conv1a.bias")->strip(cuda), weights.find("dec_conv1b.bias")->strip(cuda)},
            {weights.find("dec_conv0.bias")->strip(cuda)}
        };

        decoder_post_ops = {
            LayerPostOp::NN_UPSAMPLE,
            LayerPostOp::NN_UPSAMPLE,
            LayerPostOp::NN_UPSAMPLE,
            LayerPostOp::NONE,
            LayerPostOp::NONE
        };
    }

    size_t total_encoder_weights = 0, total_decoder_weights = 0;
    for (const auto& w : encoder_weights) total_encoder_weights += w.size();
    for (const auto& w : decoder_weights) total_decoder_weights += w.size();
    
    model.encoder_weights_storage = new TzaTensorStripped[total_encoder_weights];
    model.encoder_biases_storage = new TzaTensorStripped[total_encoder_weights];
    model.decoder_weights_storage = new TzaTensorStripped[total_decoder_weights];
    model.decoder_biases_storage = new TzaTensorStripped[total_decoder_weights];
    
    size_t offset = 0;
    model.encoder_layers = new Layer[encoder_weights.size()];
    for (size_t i = 0; i < encoder_weights.size(); i++) {
        for (size_t j = 0; j < encoder_weights[i].size(); j++) {
            model.encoder_weights_storage[offset + j] = encoder_weights[i][j];
            model.encoder_biases_storage[offset + j] = encoder_biases[i][j];
        }
        model.encoder_layers[i] = {&model.encoder_weights_storage[offset], &model.encoder_biases_storage[offset], encoder_weights[i].size(), encoder_post_ops[i]};
        offset += encoder_weights[i].size();
    }
    
    offset = 0;
    model.decoder_layers = new Layer[decoder_weights.size()];
    for (size_t i = 0; i < decoder_weights.size(); i++) {
        for (size_t j = 0; j < decoder_weights[i].size(); j++) {
            model.decoder_weights_storage[offset + j] = decoder_weights[i][j];
            model.decoder_biases_storage[offset + j] = decoder_biases[i][j];
        }
        model.decoder_layers[i] = {&model.decoder_weights_storage[offset], &model.decoder_biases_storage[offset], decoder_weights[i].size(), decoder_post_ops[i]};
        offset += decoder_weights[i].size();
    }
    
    model.num_encoder_layers = encoder_weights.size();
    model.num_decoder_layers = decoder_weights.size();

    if (cuda) {
        TzaTensorStripped *d_enc_weights, *d_enc_biases, *d_dec_weights, *d_dec_biases;
        cudaMalloc(&d_enc_weights, total_encoder_weights * sizeof(TzaTensorStripped));
        cudaMalloc(&d_enc_biases, total_encoder_weights * sizeof(TzaTensorStripped));
        cudaMalloc(&d_dec_weights, total_decoder_weights * sizeof(TzaTensorStripped));
        cudaMalloc(&d_dec_biases, total_decoder_weights * sizeof(TzaTensorStripped));
        
        cudaMemcpy(d_enc_weights, model.encoder_weights_storage, total_encoder_weights * sizeof(TzaTensorStripped), cudaMemcpyHostToDevice);
        cudaMemcpy(d_enc_biases, model.encoder_biases_storage, total_encoder_weights * sizeof(TzaTensorStripped), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dec_weights, model.decoder_weights_storage, total_decoder_weights * sizeof(TzaTensorStripped), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dec_biases, model.decoder_biases_storage, total_decoder_weights * sizeof(TzaTensorStripped), cudaMemcpyHostToDevice);
        
        delete[] model.encoder_weights_storage;
        delete[] model.encoder_biases_storage;
        delete[] model.decoder_weights_storage;
        delete[] model.decoder_biases_storage;
        
        model.encoder_weights_storage = d_enc_weights;
        model.encoder_biases_storage = d_enc_biases;
        model.decoder_weights_storage = d_dec_weights;
        model.decoder_biases_storage = d_dec_biases;
        
        offset = 0;
        for (size_t i = 0; i < encoder_weights.size(); i++) {
            model.encoder_layers[i].weights = &d_enc_weights[offset];
            model.encoder_layers[i].biases = &d_enc_biases[offset];
            offset += encoder_weights[i].size();
        }
        
        offset = 0;
        for (size_t i = 0; i < decoder_weights.size(); i++) {
            model.decoder_layers[i].weights = &d_dec_weights[offset];
            model.decoder_layers[i].biases = &d_dec_biases[offset];
            offset += decoder_weights[i].size();
        }
        
        Layer *d_enc_layers, *d_dec_layers;
        cudaMalloc(&d_enc_layers, encoder_weights.size() * sizeof(Layer));
        cudaMalloc(&d_dec_layers, decoder_weights.size() * sizeof(Layer));
        
        cudaMemcpy(d_enc_layers, model.encoder_layers, encoder_weights.size() * sizeof(Layer), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dec_layers, model.decoder_layers, decoder_weights.size() * sizeof(Layer), cudaMemcpyHostToDevice);
        
        delete[] model.encoder_layers;
        delete[] model.decoder_layers;
        
        model.encoder_layers = d_enc_layers;
        model.decoder_layers = d_dec_layers;
    }

    return model;
}

void freeUNetModel(UNetModel& model, bool cuda) {
    if (cuda) {
        // TODO CRITICAL: free all weights and biases
        // too lazy to do this now
        cudaFree(model.encoder_layers);
        cudaFree(model.decoder_layers);
        cudaFree(model.encoder_weights_storage);
        cudaFree(model.encoder_biases_storage);
        cudaFree(model.decoder_weights_storage);
        cudaFree(model.decoder_biases_storage);
    } else {
        delete[] model.encoder_layers;
        delete[] model.decoder_layers;
        delete[] model.encoder_weights_storage;
        delete[] model.encoder_biases_storage;
        delete[] model.decoder_weights_storage;
        delete[] model.decoder_biases_storage;
    }
}