// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"

#include <cmath>

using namespace std; 

namespace onnxruntime {
namespace contrib {
template <typename T>
class Mish : public OpKernel {
 public:
  explicit Mish(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override {
    auto X = context->Input<Tensor>(0);
    auto& dims = X->Shape();
    auto Y = context->Output(0, dims);
    
    auto X_Data = (X->template Data<T>());
    auto Y_Data = (Y->template MutableData<T>());

    for (int64_t i = 0, sz = dims.Size(); i < sz; i++,Y_Data++,X_Data++) {
      *Y_Data = *X_Data * tanh(log((exp(*X_Data) + 1)));
    }

    return Status::OK();
  }
};
}  // namespace contrib
}  // namespace onnxruntime
