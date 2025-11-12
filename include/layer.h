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
    uint32_t *weight_idxs;
    uint32_t *bias_idxs;
    uint32_t *out_channels;
    uint32_t num_convs;
    LayerPostOp post_op;
};

#endif // LAYER_H