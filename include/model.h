#ifndef MODEL_H
#define MODEL_H

#include "layer.h"

struct UNetModel {
    // if cuda, all pointers in CUDA mem
    Layer* encoder_layers;
    Layer* decoder_layers;
    size_t num_encoder_layers;
    size_t num_decoder_layers;

    TzaTensorStripped* encoder_weights_storage;
    TzaTensorStripped* encoder_biases_storage;
    TzaTensorStripped* decoder_weights_storage;
    TzaTensorStripped* decoder_biases_storage;
};

UNetModel createUNetModel(const std::string& model_name, TzaFile& weights, bool cuda = false);
void freeUNetModel(UNetModel& model, bool cuda = false);

#endif // MODEL_H