"""Spark gate lookup and FP32 split-KV merge."""
from cannbotdsl import dtypes
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ops.sync import vec_sync_all
from cannbotdsl.ops.reg import *

@jit
def gate_prepare(rawgm,lutgm,outgm,arena):
    gch=arena.channel((16,),dtypes.bfloat16,depth=1)
    lch=arena.channel((4352,),dtypes.bfloat16,depth=1)
    ych=arena.channel((16,),dtypes.bfloat16,depth=1)
    mem_copy(gch.produce().view(1,16),rawgm.view(1,16))
    compact=lch.produce()
    mem_copy(compact[0:2048,].view(1,2048),lutgm[0x3b00:0x4300,].view(1,2048))
    mem_copy(compact[2048:4096,].view(1,2048),lutgm[0xbb00:0xc300,].view(1,2048))
    mem_copy(compact[4096:4224,].view(1,128),lutgm[0x7f80:0x8000,].view(1,128))
    mem_copy(compact[4224:4352,].view(1,128),lutgm[0xff80:0x10000,].view(1,128))
    g,l,y=gch.consume(),lch.consume(),ych.produce()
    lf=arena.buffer((4352,),dtypes.float32)
    with vf(mode='simd'):
        p=create_mask(pattern='all',elem_bits=32)
        p1,_=update_mask(16,elem_bits=32)
        gb=vload_unpack(g,0,unpack_mode=UnpackMode.UNPACK2)
        bits=vreinterpret_lanes(gb,dtypes.uint32)
        for j in range(68):
            vstore(lf,j*64,vcast(vload_unpack(l,j*64,unpack_mode=UnpackMode.UNPACK2),dtypes.float32,mask=p),p)
        vmem_bar('vst_vld')
        ab=vbitwise_and(bits,vdups(32767,dtypes.uint32,mask=p1),mask=p1)
        index=vbitwise_and(vadds(ab,4294952192,mask=p1),vdups(2047,dtypes.uint32,mask=p1),mask=p1)
        sign=vbitwise_and(vshr(bits,4,mask=p1),vdups(2048,dtypes.uint32,mask=p1),mask=p1)
        index=vbitwise_or(index,sign,mask=p1)
        ext=vadds(vbitwise_and(bits,vdups(127,dtypes.uint32,mask=p1),mask=p1),4096,mask=p1)
        ext=vadd(ext,vbitwise_and(vshr(bits,8,mask=p1),vdups(128,dtypes.uint32,mask=p1),mask=p1),mask=p1)
        index=vselect(ext,index,cond_mask=vges(ab,0x7f80,mask=p1))
        value=vgather(lf,index,mask=p1)
        large=vselect(vdups(1.0,dtypes.float32,mask=p1),vdups(0.0,dtypes.float32,mask=p1),cond_mask=vlts(bits,32768,mask=p1))
        adjusted=vselect(vdups(0.5,dtypes.float32,mask=p1),value,cond_mask=vlts(ab,0x3b00,mask=p1))
        adjusted=vselect(large,adjusted,cond_mask=vges(ab,0x4300,mask=p1))
        value=vselect(value,adjusted,cond_mask=vges(ab,0x7f80,mask=p1))
        vstore_pack(y,0,vcast(value,dtypes.bfloat16,mask=p1,rounding=RoundingMode.RN),p1,pack_mode=PackMode.B32_TO_B16)
    mem_copy(outgm.view(1,16),ych.consume().view(1,16))

@jit
def merge_attention(partials,lses,gategm,outgm,head,parts,pool):
    weights=pool.merge_weights
    x=pool.scratch[0:16384,].view(64,256)
    l=pool.scratch[16384:16896,].view(64,8)
    for part in range(parts):
        mem_copy(x[part,None].view(256),partials[part,head,None].view(256))
        mem_copy(l[part,None].view(8),lses[part,head,None].view(8))
    mem_copy(pool.merge_g,gategm)
    vec_sync_all()
    g=pool.merge_g;y=pool.merge_y
    count=8
    if parts>8:count=64
    with vf(mode='simd'):
        p=create_mask(pattern='all',elem_bits=32)
        pm,_=update_mask(count,elem_bits=32)
        ix=varange(0,dtypes.int32)
        valid=vlts(ix,parts,mask=pm)
        offsets=vreinterpret(vmuls(ix,8,mask=pm),dtypes.uint32)
        z=vgather(l,offsets,mask=valid)
        z=vselect(z,vdups(-1.e30,dtypes.float32,mask=pm),cond_mask=valid)
        mx=vdup(vreduce_max(z,mask=pm),mask=pm)
        z=vexp(vsub(z,mx,mask=pm),mask=pm)
        den=vdup(vreduce_sum(z,mask=pm),mask=pm)
        z=vdiv(z,den,mask=pm)
        vstore(weights,0,z,pm)
        vmem_bar('vst_vld')
        gf=vcast(vload_broadcast(g,0),dtypes.float32,mask=p)
        for j in range(4):
            acc=vdups(0.0,dtypes.float32,mask=p)
            for i in range(parts):
                value=vload(x,i*256+j*64)
                w=vload_broadcast(weights,i)
                acc=vadd(acc,vmul(value,w,mask=p),mask=p)
            ab=vcast(acc,dtypes.bfloat16,mask=p,rounding=RoundingMode.RN)
            acc=vcast(ab,dtypes.float32,mask=p)
            out=vcast(vmul(acc,gf,mask=p),dtypes.bfloat16,mask=p,rounding=RoundingMode.RN)
            vstore_pack(y,j*64,out,p,pack_mode=PackMode.B32_TO_B16)
    mem_copy(outgm.view(1,256),pool.merge_y.view(1,256))
