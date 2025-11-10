#ifndef MODEL_H
#define MODEL_H

#include "layer.h"

class UNetModel {
public:
    UNetModel(const std::string& model_name, TzaFile& weights);
    ~UNetModel();
    Layer* encoder_layers;
    Layer* decoder_layers;
    size_t num_encoder_layers;
    size_t num_decoder_layers;

private:
    TzaTensorStripped* encoder_weights_storage;
    TzaTensorStripped* encoder_biases_storage;
    TzaTensorStripped* decoder_weights_storage;
    TzaTensorStripped* decoder_biases_storage;
};

#endif // MODEL_H