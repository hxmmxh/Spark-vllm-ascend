#pragma once
#include "kernel_operator.h"
namespace HeadQ6 {
namespace R=AscendC::Reg;
constexpr R::CastTrait up{R::RegLayout::ZERO,R::SatMode::UNKNOWN,R::MaskMergeMode::ZEROING,AscendC::RoundMode::UNKNOWN};
constexpr R::CastTrait upper{R::RegLayout::ONE,R::SatMode::UNKNOWN,R::MaskMergeMode::ZEROING,AscendC::RoundMode::UNKNOWN};
constexpr R::CastTrait down{R::RegLayout::ZERO,R::SatMode::NO_SAT,R::MaskMergeMode::ZEROING,AscendC::RoundMode::CAST_RINT};
template<bool GEMV> __simd_vf__ void compute(__ubuf__ float*x,__ubuf__ uint8_t*w,__ubuf__ float*tmp,__ubuf__ bfloat16_t*y){
 constexpr int K=2560,M=8;
 auto p=R::CreateMask<float,R::MaskPattern::ALL>();auto p16=R::CreateMask<uint16_t,R::MaskPattern::ALL>();
 R::RegTensor<bfloat16_t>rounded,b;R::RegTensor<float>a,lo;
 R::RegTensor<float>s0;if constexpr(GEMV)R::Duplicate(s0,0.0f);
 R::RegTensor<float>s1;if constexpr(GEMV)R::Duplicate(s1,0.0f);
 R::RegTensor<float>s2;if constexpr(GEMV)R::Duplicate(s2,0.0f);
 R::RegTensor<float>s3;if constexpr(GEMV)R::Duplicate(s3,0.0f);
 R::RegTensor<float>s4;if constexpr(GEMV)R::Duplicate(s4,0.0f);
 R::RegTensor<float>s5;if constexpr(GEMV)R::Duplicate(s5,0.0f);
 R::RegTensor<float>s6;if constexpr(GEMV)R::Duplicate(s6,0.0f);
 R::RegTensor<float>s7;if constexpr(GEMV)R::Duplicate(s7,0.0f);
 for(uint16_t j=0;j<K/64;j++){
 if constexpr(GEMV)R::LoadAlign(a,x+j*64);
 R::LoadAlign<bfloat16_t,R::LoadDist::DIST_UNPACK_B16>(b,(__ubuf__ bfloat16_t*)w+(j*M+0)*64);
 R::Cast<float,bfloat16_t,up>(lo,b,p);
 if constexpr(GEMV)R::MulAddDst(s0,a,lo,p);else R::StoreAlign<bfloat16_t,R::StoreDist::DIST_PACK_B32>(y+0*K+j*64,b,p);
 R::LoadAlign<bfloat16_t,R::LoadDist::DIST_UNPACK_B16>(b,(__ubuf__ bfloat16_t*)w+(j*M+1)*64);
 R::Cast<float,bfloat16_t,up>(lo,b,p);
 if constexpr(GEMV)R::MulAddDst(s1,a,lo,p);else R::StoreAlign<bfloat16_t,R::StoreDist::DIST_PACK_B32>(y+1*K+j*64,b,p);
 R::LoadAlign<bfloat16_t,R::LoadDist::DIST_UNPACK_B16>(b,(__ubuf__ bfloat16_t*)w+(j*M+2)*64);
 R::Cast<float,bfloat16_t,up>(lo,b,p);
 if constexpr(GEMV)R::MulAddDst(s2,a,lo,p);else R::StoreAlign<bfloat16_t,R::StoreDist::DIST_PACK_B32>(y+2*K+j*64,b,p);
 R::LoadAlign<bfloat16_t,R::LoadDist::DIST_UNPACK_B16>(b,(__ubuf__ bfloat16_t*)w+(j*M+3)*64);
 R::Cast<float,bfloat16_t,up>(lo,b,p);
 if constexpr(GEMV)R::MulAddDst(s3,a,lo,p);else R::StoreAlign<bfloat16_t,R::StoreDist::DIST_PACK_B32>(y+3*K+j*64,b,p);
 R::LoadAlign<bfloat16_t,R::LoadDist::DIST_UNPACK_B16>(b,(__ubuf__ bfloat16_t*)w+(j*M+4)*64);
 R::Cast<float,bfloat16_t,up>(lo,b,p);
 if constexpr(GEMV)R::MulAddDst(s4,a,lo,p);else R::StoreAlign<bfloat16_t,R::StoreDist::DIST_PACK_B32>(y+4*K+j*64,b,p);
 R::LoadAlign<bfloat16_t,R::LoadDist::DIST_UNPACK_B16>(b,(__ubuf__ bfloat16_t*)w+(j*M+5)*64);
 R::Cast<float,bfloat16_t,up>(lo,b,p);
 if constexpr(GEMV)R::MulAddDst(s5,a,lo,p);else R::StoreAlign<bfloat16_t,R::StoreDist::DIST_PACK_B32>(y+5*K+j*64,b,p);
 R::LoadAlign<bfloat16_t,R::LoadDist::DIST_UNPACK_B16>(b,(__ubuf__ bfloat16_t*)w+(j*M+6)*64);
 R::Cast<float,bfloat16_t,up>(lo,b,p);
 if constexpr(GEMV)R::MulAddDst(s6,a,lo,p);else R::StoreAlign<bfloat16_t,R::StoreDist::DIST_PACK_B32>(y+6*K+j*64,b,p);
 R::LoadAlign<bfloat16_t,R::LoadDist::DIST_UNPACK_B16>(b,(__ubuf__ bfloat16_t*)w+(j*M+7)*64);
 R::Cast<float,bfloat16_t,up>(lo,b,p);
 if constexpr(GEMV)R::MulAddDst(s7,a,lo,p);else R::StoreAlign<bfloat16_t,R::StoreDist::DIST_PACK_B32>(y+7*K+j*64,b,p);
 }
 if constexpr(GEMV){uint32_t one=1;auto p1=R::UpdateMask<float>(one);
 R::Reduce<R::ReduceType::SUM>(s0,s0,p);R::StoreAlign(tmp+0*64,s0,p1);
 R::Reduce<R::ReduceType::SUM>(s1,s1,p);R::StoreAlign(tmp+1*64,s1,p1);
 R::Reduce<R::ReduceType::SUM>(s2,s2,p);R::StoreAlign(tmp+2*64,s2,p1);
 R::Reduce<R::ReduceType::SUM>(s3,s3,p);R::StoreAlign(tmp+3*64,s3,p1);
 R::Reduce<R::ReduceType::SUM>(s4,s4,p);R::StoreAlign(tmp+4*64,s4,p1);
 R::Reduce<R::ReduceType::SUM>(s5,s5,p);R::StoreAlign(tmp+5*64,s5,p1);
 R::Reduce<R::ReduceType::SUM>(s6,s6,p);R::StoreAlign(tmp+6*64,s6,p1);
 R::Reduce<R::ReduceType::SUM>(s7,s7,p);R::StoreAlign(tmp+7*64,s7,p1);
 R::LocalMemBar<R::MemType::VEC_STORE,R::MemType::VEC_LOAD>();uint32_t count=8;auto pm=R::UpdateMask<float>(count);R::RegTensor<int32_t>ix;R::RegTensor<float>out;R::Arange(ix,0);R::Muls(ix,ix,64,pm);R::Gather(out,tmp,(R::RegTensor<uint32_t>&)ix,pm);R::Cast<bfloat16_t,float,down>(rounded,out,pm);R::StoreAlign<bfloat16_t,R::StoreDist::DIST_PACK_B32>(y,rounded,pm);
 }
}
__simd_vf__ void expand(__ubuf__ bfloat16_t*x,__ubuf__ float*y){auto p=R::CreateMask<float,R::MaskPattern::ALL>();R::RegTensor<bfloat16_t>b;R::RegTensor<float>f;for(uint16_t j=0;j<40;j++){R::LoadAlign<bfloat16_t,R::LoadDist::DIST_UNPACK_B16>(b,x+j*64);R::Cast<float,bfloat16_t,up>(f,b,p);R::StoreAlign(y+j*64,f,p);}R::LocalMemBar<R::MemType::VEC_STORE,R::MemType::VEC_LOAD>();}
}
