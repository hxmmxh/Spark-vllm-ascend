#pragma once
#include "kernel_operator.h"
namespace HeadQ6 {
namespace R=AscendC::Reg;
constexpr R::CastTrait up{R::RegLayout::ZERO,R::SatMode::UNKNOWN,R::MaskMergeMode::ZEROING,AscendC::RoundMode::UNKNOWN};
constexpr R::CastTrait upper{R::RegLayout::ONE,R::SatMode::UNKNOWN,R::MaskMergeMode::ZEROING,AscendC::RoundMode::UNKNOWN};
constexpr R::CastTrait down{R::RegLayout::ZERO,R::SatMode::NO_SAT,R::MaskMergeMode::ZEROING,AscendC::RoundMode::CAST_RINT};
template<bool GEMV> __simd_vf__ void compute(__ubuf__ float*x,__ubuf__ uint8_t*w,__ubuf__ float*tmp,__ubuf__ bfloat16_t*y){
 constexpr int K=2560,M=8;
 auto p=R::CreateMask<float,R::MaskPattern::ALL>();auto p16=R::CreateMask<uint16_t,R::MaskPattern::ALL>();auto p8=R::CreateMask<int8_t,R::MaskPattern::ALL>();
 R::RegTensor<uint32_t>index,indexHigh;R::Arange((R::RegTensor<int32_t>&)index,0);R::ShiftRights(index,index,(int16_t)4,p);R::Adds(indexHigh,index,4u,p);
 // Exact FP32 d * signed-subscale reconstruction, once per packed tile.
 auto expanded=(__ubuf__ float*)183296;
 R::RegTensor<int32_t>dix;R::Arange(dix,0);R::ShiftRights(dix,dix,(int16_t)2,p);
 R::RegTensor<int8_t>sub;R::RegTensor<half>subh,dh;R::RegTensor<float>sf,df;
 for(uint16_t si=0;si<20;si++){
  R::LoadAlign<uint8_t,R::LoadDist::DIST_UNPACK4_B8>((R::RegTensor<uint8_t>&)sub,w+K*M+si*64);
  R::Cast<half,int8_t,up>(subh,sub,p8);R::Cast<float,half,up>(sf,subh,p16);
  R::Gather(dh,(__ubuf__ half*)(w+K*M+K*M/16)+si*16,(R::RegTensor<uint16_t>&)dix,p);
  R::Cast<float,half,up>(df,dh,p);R::Mul(sf,sf,df,p);R::StoreAlign(expanded+si*64,sf,p);
 }
 R::LocalMemBar<R::MemType::VEC_STORE,R::MemType::VEC_LOAD>();
 R::RegTensor<int8_t>codes;R::RegTensor<half>weights;R::RegTensor<bfloat16_t>rounded;R::RegTensor<float>a,lo,hi,sl,sh;
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
 R::LoadAlign<uint8_t,R::LoadDist::DIST_UNPACK_B8>((R::RegTensor<uint8_t>&)codes,w+(j*4+0)*128);
 R::Cast<half,int8_t,up>(weights,codes,p8);
 R::Cast<float,half,up>(lo,weights,p16);R::Cast<float,half,upper>(hi,weights,p16);
 R::Gather(sl,expanded+j*32+0,index,p);R::Gather(sh,expanded+j*32+0,indexHigh,p);
 R::Mul(lo,lo,sl,p);R::Mul(hi,hi,sh,p);
 R::Cast<bfloat16_t,float,down>(rounded,lo,p);
 if constexpr(GEMV){R::Cast<float,bfloat16_t,up>(lo,rounded,p);R::MulAddDst(s0,a,lo,p);}else R::StoreAlign<bfloat16_t,R::StoreDist::DIST_PACK_B32>(y+0*K+j*64,rounded,p);
 R::Cast<bfloat16_t,float,down>(rounded,hi,p);
 if constexpr(GEMV){R::Cast<float,bfloat16_t,up>(hi,rounded,p);R::MulAddDst(s1,a,hi,p);}else R::StoreAlign<bfloat16_t,R::StoreDist::DIST_PACK_B32>(y+1*K+j*64,rounded,p);
 R::LoadAlign<uint8_t,R::LoadDist::DIST_UNPACK_B8>((R::RegTensor<uint8_t>&)codes,w+(j*4+1)*128);
 R::Cast<half,int8_t,up>(weights,codes,p8);
 R::Cast<float,half,up>(lo,weights,p16);R::Cast<float,half,upper>(hi,weights,p16);
 R::Gather(sl,expanded+j*32+8,index,p);R::Gather(sh,expanded+j*32+8,indexHigh,p);
 R::Mul(lo,lo,sl,p);R::Mul(hi,hi,sh,p);
 R::Cast<bfloat16_t,float,down>(rounded,lo,p);
 if constexpr(GEMV){R::Cast<float,bfloat16_t,up>(lo,rounded,p);R::MulAddDst(s2,a,lo,p);}else R::StoreAlign<bfloat16_t,R::StoreDist::DIST_PACK_B32>(y+2*K+j*64,rounded,p);
 R::Cast<bfloat16_t,float,down>(rounded,hi,p);
 if constexpr(GEMV){R::Cast<float,bfloat16_t,up>(hi,rounded,p);R::MulAddDst(s3,a,hi,p);}else R::StoreAlign<bfloat16_t,R::StoreDist::DIST_PACK_B32>(y+3*K+j*64,rounded,p);
 R::LoadAlign<uint8_t,R::LoadDist::DIST_UNPACK_B8>((R::RegTensor<uint8_t>&)codes,w+(j*4+2)*128);
 R::Cast<half,int8_t,up>(weights,codes,p8);
 R::Cast<float,half,up>(lo,weights,p16);R::Cast<float,half,upper>(hi,weights,p16);
 R::Gather(sl,expanded+j*32+16,index,p);R::Gather(sh,expanded+j*32+16,indexHigh,p);
 R::Mul(lo,lo,sl,p);R::Mul(hi,hi,sh,p);
 R::Cast<bfloat16_t,float,down>(rounded,lo,p);
 if constexpr(GEMV){R::Cast<float,bfloat16_t,up>(lo,rounded,p);R::MulAddDst(s4,a,lo,p);}else R::StoreAlign<bfloat16_t,R::StoreDist::DIST_PACK_B32>(y+4*K+j*64,rounded,p);
 R::Cast<bfloat16_t,float,down>(rounded,hi,p);
 if constexpr(GEMV){R::Cast<float,bfloat16_t,up>(hi,rounded,p);R::MulAddDst(s5,a,hi,p);}else R::StoreAlign<bfloat16_t,R::StoreDist::DIST_PACK_B32>(y+5*K+j*64,rounded,p);
 R::LoadAlign<uint8_t,R::LoadDist::DIST_UNPACK_B8>((R::RegTensor<uint8_t>&)codes,w+(j*4+3)*128);
 R::Cast<half,int8_t,up>(weights,codes,p8);
 R::Cast<float,half,up>(lo,weights,p16);R::Cast<float,half,upper>(hi,weights,p16);
 R::Gather(sl,expanded+j*32+24,index,p);R::Gather(sh,expanded+j*32+24,indexHigh,p);
 R::Mul(lo,lo,sl,p);R::Mul(hi,hi,sh,p);
 R::Cast<bfloat16_t,float,down>(rounded,lo,p);
 if constexpr(GEMV){R::Cast<float,bfloat16_t,up>(lo,rounded,p);R::MulAddDst(s6,a,lo,p);}else R::StoreAlign<bfloat16_t,R::StoreDist::DIST_PACK_B32>(y+6*K+j*64,rounded,p);
 R::Cast<bfloat16_t,float,down>(rounded,hi,p);
 if constexpr(GEMV){R::Cast<float,bfloat16_t,up>(hi,rounded,p);R::MulAddDst(s7,a,hi,p);}else R::StoreAlign<bfloat16_t,R::StoreDist::DIST_PACK_B32>(y+7*K+j*64,rounded,p);
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
