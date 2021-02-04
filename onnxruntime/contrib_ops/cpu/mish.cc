#include "contrib_ops/cpu/mish.h"
namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    Mish,
    kOnnxDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float,double>()),
    Mish
);

}
}
