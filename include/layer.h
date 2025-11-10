#ifndef LAYER_H
#define LAYER_H

#include "tza.h"

enum LayerPostOp {
    MAX_POOL,
    AVG_POOL,
    NN_UPSAMPLE,
    NONE
};

struct Layer {
    const TzaTensorStripped* weights;
    const TzaTensorStripped* biases;
    size_t num_convs;
    LayerPostOp post_op;
};

#endif // LAYER_H