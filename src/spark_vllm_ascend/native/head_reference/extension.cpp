#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/core/DeviceGuard.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
extern "C" void launch_head_bf16(void*,uint8_t*,uint8_t*,uint8_t*,int,int);
void run_scale(const at::Tensor&x,const at::Tensor&w,const at::Tensor&y,int64_t rows,bool decode){
 TORCH_CHECK(x.device()==w.device()&&x.device()==y.device()&&x.is_contiguous()&&w.is_contiguous()&&y.is_contiguous(),"device/layout");
 TORCH_CHECK(x.scalar_type()==at::kBFloat16&&x.numel()==2560&&w.scalar_type()==at::kByte&&y.scalar_type()==at::kBFloat16,"dtype/input");
 TORCH_CHECK(rows>0&&rows<=131072&&rows%8==0&&w.numel()==rows*5120&&y.numel()==rows*(decode?2560:1),"size");
 const c10::OptionalDeviceGuard guard(x.device());auto s=c10_npu::getCurrentNPUStream().stream(false);auto fn=[=]()->int{launch_head_bf16(s,(uint8_t*)x.data_ptr(),(uint8_t*)w.data_ptr(),(uint8_t*)y.data_ptr(),rows,decode);return 0;};at_npu::native::OpCommand::RunOpApi("BF16OrderedHeadTest",fn);
}
TORCH_LIBRARY(spark_head_bf16,m){m.def("run(Tensor x, Tensor w,Tensor(a!) y,int rows,bool decode) -> ()");}
TORCH_LIBRARY_IMPL(spark_head_bf16,PrivateUse1,m){m.impl("run",run_scale);}
