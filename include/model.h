#ifndef MODEL_H
#define MODEL_H

#include "layer.h"

class UNetModel {
public:
    UNetModel(const std::string& model_name, TzaFile& weights);
    Layer* encoder_layers;
    Layer* decoder_layers;
    size_t num_encoder_layers;
    size_t num_decoder_layers;
};

#endif // MODEL_H