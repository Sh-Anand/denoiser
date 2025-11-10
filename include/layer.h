#ifndef LAYER_H
#define LAYER_H

#include "tza.h"

enum LayerPostOp {
    MAX_POOL,
    AVG_POOL,
    NN_UPSAMPLE,
    NONE
};

typedef std::pair<std::vector<std::pair<const TzaTensorStripped*, const TzaTensorStripped*>>, LayerPostOp> Layer;

#endif // LAYER_H