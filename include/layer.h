#include "tza.h"

enum LayerPostOp {
    MAX_POOL,
    AVG_POOL,
    NN_UPSAMPLE,
    NONE
};

typedef std::pair<std::vector<std::pair<const TzaTensor*, const TzaTensor*>>, LayerPostOp> Layer;