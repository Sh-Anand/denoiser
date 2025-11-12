#include "cuda_defs.h"
#include "model.h"

// NOTE: massive assumption that the weights will remain in scope throughout the model's lifetime
// If cuda, assume weights are gpu virtual addresses
UNetModel createUNetModel(const std::string& model_name, TzaFile& weights, bool cuda) {
    UNetModel model;
    std::vector<std::vector<const TzaTensor*>> encoder_layers;
    std::vector<LayerPostOp> encoder_post_ops;
    std::vector<std::vector<const TzaTensor*>> decoder_layers;
    std::vector<LayerPostOp> decoder_post_ops;
    
    if (model_name == "rt_hdr") {
        encoder_layers = {
            {weights.find("enc_conv0.weight"), weights.find("enc_conv0.bias")},
            {weights.find("enc_conv1.weight"), weights.find("enc_conv1.bias")},
            {weights.find("enc_conv2.weight"), weights.find("enc_conv2.bias")},
            {weights.find("enc_conv3.weight"), weights.find("enc_conv3.bias")},
            {weights.find("enc_conv4.weight"), weights.find("enc_conv4.bias")},
            {weights.find("enc_conv5a.weight"), weights.find("enc_conv5a.bias"), weights.find("enc_conv5b.weight"), weights.find("enc_conv5b.bias")}
        };

        encoder_post_ops = {
            LayerPostOp::NONE,
            LayerPostOp::MAX_POOL,
            LayerPostOp::MAX_POOL,
            LayerPostOp::MAX_POOL,
            LayerPostOp::MAX_POOL,
            LayerPostOp::NN_UPSAMPLE
        };

        decoder_layers = {
            {weights.find("dec_conv4a.weight"), weights.find("dec_conv4a.bias"), weights.find("dec_conv4b.weight"), weights.find("dec_conv4b.bias")},
            {weights.find("dec_conv3a.weight"), weights.find("dec_conv3a.bias"), weights.find("dec_conv3b.weight"), weights.find("dec_conv3b.bias")},
            {weights.find("dec_conv2a.weight"), weights.find("dec_conv2a.bias"), weights.find("dec_conv2b.weight"), weights.find("dec_conv2b.bias")},
            {weights.find("dec_conv1a.weight"), weights.find("dec_conv1a.bias"), weights.find("dec_conv1b.weight"), weights.find("dec_conv1b.bias")},
            {weights.find("dec_conv0.weight"), weights.find("dec_conv0.bias")}
        };
        
        decoder_post_ops = {
            LayerPostOp::NN_UPSAMPLE,
            LayerPostOp::NN_UPSAMPLE,
            LayerPostOp::NN_UPSAMPLE,
            LayerPostOp::NONE,
            LayerPostOp::NONE
        };
    }

    size_t total_elements = 0;
    for (size_t i = 0; i < encoder_layers.size(); i++) {
        for (size_t j = 0; j < encoder_layers[i].size(); j++) {
            total_elements += encoder_layers[i][j]->elementCount();
        }
    }

    for (size_t i = 0; i < decoder_layers.size(); i++) {
        for (size_t j = 0; j < decoder_layers[i].size(); j++) {
            total_elements += decoder_layers[i][j]->elementCount();
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
    model.encoder_layers = new Layer[encoder_layers.size()];
    model.decoder_layers = new Layer[decoder_layers.size()];
    model.num_encoder_layers = encoder_layers.size();
    model.num_decoder_layers = decoder_layers.size();

    for (int i = 0; i < encoder_layers.size(); i++) {
        model.encoder_layers[i].weight_idxs = new uint32_t[encoder_layers[i].size()];
        model.encoder_layers[i].bias_idxs = new uint32_t[encoder_layers[i].size()];
        model.encoder_layers[i].out_channels = new uint32_t[encoder_layers[i].size()];
        model.encoder_layers[i].num_convs = encoder_layers[i].size()/2;
        model.encoder_layers[i].post_op = encoder_post_ops[i];
        for (int j = 0; j < encoder_layers[i].size()/2; j++) {
            int weight_idx = j*2, bias_idx = j*2+1;
            model.encoder_layers[i].weight_idxs[j] = offset;
            if (cuda) {
                CUDA_ERR(cudaMemcpy((void *)(model.weights->half_data + offset), encoder_layers[i][weight_idx]->data.data(), encoder_layers[i][weight_idx]->elementCount() * sizeof(half), cudaMemcpyHostToDevice));
            } else {
                memcpy((void *)(model.weights->float16_data + offset), encoder_layers[i][weight_idx]->data.data(), encoder_layers[i][weight_idx]->elementCount() * sizeof(_Float16));
            }
            offset += encoder_layers[i][weight_idx]->elementCount();
            model.encoder_layers[i].bias_idxs[j] = offset;
            if (cuda) {
                CUDA_ERR(cudaMemcpy((void *)(model.weights->half_data + offset), encoder_layers[i][bias_idx]->data.data(), encoder_layers[i][bias_idx]->elementCount() * sizeof(half), cudaMemcpyHostToDevice));
            } else {
                memcpy((void *)(model.weights->float16_data + offset), encoder_layers[i][bias_idx]->data.data(), encoder_layers[i][bias_idx]->elementCount() * sizeof(_Float16));
            }
            offset += encoder_layers[i][bias_idx]->elementCount();
            model.encoder_layers[i].out_channels[j] = encoder_layers[i][weight_idx]->dims[0];
        }
    }

    for (int i = 0; i < decoder_layers.size(); i++) {
        model.decoder_layers[i].weight_idxs = new uint32_t[decoder_layers[i].size()];
        model.decoder_layers[i].bias_idxs = new uint32_t[decoder_layers[i].size()];
        model.decoder_layers[i].out_channels = new uint32_t[decoder_layers[i].size()];
        model.decoder_layers[i].num_convs = decoder_layers[i].size()/2;
        model.decoder_layers[i].post_op = decoder_post_ops[i];
        for (int j = 0; j < decoder_layers[i].size()/2; j++) {
            int weight_idx = j*2, bias_idx = j*2+1;
            model.decoder_layers[i].weight_idxs[j] = offset;
            if (cuda) {
                CUDA_ERR(cudaMemcpy((void *)(model.weights->half_data + offset), decoder_layers[i][weight_idx]->data.data(), decoder_layers[i][weight_idx]->elementCount() * sizeof(half), cudaMemcpyHostToDevice));
            } else {
                memcpy((void *)(model.weights->float16_data + offset), decoder_layers[i][weight_idx]->data.data(), decoder_layers[i][weight_idx]->elementCount() * sizeof(_Float16));
            }
            offset += decoder_layers[i][weight_idx]->elementCount();
            model.decoder_layers[i].bias_idxs[j] = offset;
            if (cuda) {
                CUDA_ERR(cudaMemcpy((void *)(model.weights->half_data + offset), decoder_layers[i][bias_idx]->data.data(), decoder_layers[i][bias_idx]->elementCount() * sizeof(half), cudaMemcpyHostToDevice));
            } else {
                memcpy((void *)(model.weights->float16_data + offset), decoder_layers[i][bias_idx]->data.data(), decoder_layers[i][bias_idx]->elementCount() * sizeof(_Float16));
            }
            offset += decoder_layers[i][bias_idx]->elementCount();
            model.decoder_layers[i].out_channels[j] = decoder_layers[i][weight_idx]->dims[0];
        }
    }
    return model;
}

void freeUNetModel(UNetModel& model, bool cuda) {
    // Free layer-specific arrays
    for (size_t i = 0; i < model.num_encoder_layers; i++) {
        delete[] model.encoder_layers[i].weight_idxs;
        delete[] model.encoder_layers[i].bias_idxs;
        delete[] model.encoder_layers[i].out_channels;
    }
    for (size_t i = 0; i < model.num_decoder_layers; i++) {
        delete[] model.decoder_layers[i].weight_idxs;
        delete[] model.decoder_layers[i].bias_idxs;
        delete[] model.decoder_layers[i].out_channels;
    }
    
    // Free layer arrays
    delete[] model.encoder_layers;
    delete[] model.decoder_layers;
    
    // Free weight buffer
    if (cuda) {
        CUDA_ERR(cudaFree((void*)model.weights->half_data));
    } else {
        delete[] model.weights->float16_data;
    }
    delete model.weights;
}