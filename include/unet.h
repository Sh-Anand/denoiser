#ifndef UNET_H
#define UNET_H

#include "exr.h"
#include "model/model.h"

void oidn_unet(EXR::Image& input_img, UNetModel& model, float*& output_img);

void oidn_unet_cuda(EXR::Image& input_img, UNetModel& model, float*& output_img);

#endif // UNET_H