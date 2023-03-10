/* Copyright 2015 The math21 Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "average_pooling_cuda.h"

__global__ void math21_ml_function_average_pooling_forward_cuda_kernel(int n, int w, int h, int c, const float *input, float *output)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int k = id % c;
    id /= c;
    int b = id;

    int i;
    int out_index = (k + c*b);
    output[out_index] = 0;
    for(i = 0; i < w*h; ++i){
        int in_index = i + h*w*(k + b*c);
        output[out_index] += input[in_index];
    }
    output[out_index] /= w*h;
}

__global__ void math21_ml_function_average_pooling_backward_cuda_kernel(int n, int w, int h, int c, float *in_delta,
                                                                        const float *out_delta)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int k = id % c;
    id /= c;
    int b = id;

    int i;
    int out_index = (k + c*b);
    for(i = 0; i < w*h; ++i){
        int in_index = i + h*w*(k + b*c);
        in_delta[in_index] += out_delta[out_index] / (w*h);
    }
}

void math21_ml_function_average_pooling_forward_cuda(FnAveragePooling *f, const mlfunction_node*finput)
{
    size_t n = f->c*f->batch;

    math21_ml_function_average_pooling_forward_cuda_kernel<<<math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >>>(n, f->w, f->h, f->c, finput->y, f->output);
    math21_cuda_check_error(cudaPeekAtLastError());
}

void math21_ml_function_average_pooling_backward_cuda(FnAveragePooling *f, mlfunction_node *finput)
{
    size_t n = f->c*f->batch;

    math21_ml_function_average_pooling_backward_cuda_kernel<<<math21_cuda_gridsize(n), MATH21_CUDA_BLOCK_SIZE >>>(n, f->w, f->h, f->c, finput->dy, f->delta);
    math21_cuda_check_error(cudaPeekAtLastError());
}
