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

    #ifdef __CUDACC__
    std::vector<dim3> block_dims;
    std::vector<int> conv_im2col_tile_ks;
    #endif
};

#endif // LAYER_H