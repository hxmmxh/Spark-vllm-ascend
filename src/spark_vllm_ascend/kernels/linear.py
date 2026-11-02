"""Q4 GEMV, RMSNorm/UP, and shared UB pipeline for Spark inference."""
from cannbotdsl import dtypes
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ops.reg import PackMode, UnpackMode, RegLayout, RoundingMode, create_mask, update_mask, vcast, vload_unpack, vload_broadcast, vreinterpret, vreinterpret_lanes, vmul, vselect, vstore_pack
from cannbotdsl.tensor import reinterpret
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.ops.reg import vshr, vadds, vload, vstore, vdups, vmula, vreduce_sum, vgather, varange, vmuls, vmem_bar, vinterleave, vdeinterleave
from cannbotdsl.tensor import tile_slice
from cannbotdsl.ops.reg import *
from cannbotdsl import make_buffer, make_channel
from cannbotdsl.ops.sync import PIPE, vec_sync_notify, vec_sync_wait

class GemvStage:
    @jit
    def calculate(self, w, y):
        scales = reinterpret(w, dtype=dtypes.uint32, shape=(self.k*self.m*9//64,))
        with vf(mode="simd"):
            p = create_mask(pattern="all", elem_bits=32)
            p16 = create_mask(pattern="all", elem_bits=16)
            pq = create_mask(pattern="all", elem_bits=8)
            # E2B broadcasts repeat each uint32 eight times. Selecting even
            # lanes of two loads packs exactly the original fourfold expansion.
            for si in range(self.k*self.base_m//1024):
                for group in tuple(range(self.m//self.base_m)):
                    base=group*(self.k*self.base_m*9//64)+self.k*self.base_m//8+si*16
                    p0=vload_broadcast(scales,base,mode="elem2datablock")
                    p1=vload_broadcast(scales,base+8,mode="elem2datablock")
                    packed,unused=vdeinterleave(p0,p1)
                    vstore(self.expanded,group*(self.k*self.base_m//16)+si*64,packed,p)
            vmem_bar("vst_vld")
            s0 = vdups(0.0, dtypes.float32, mask=p)
            s1 = vdups(0.0, dtypes.float32, mask=p)
            s2 = vdups(0.0, dtypes.float32, mask=p)
            s3 = vdups(0.0, dtypes.float32, mask=p)
            s4 = vdups(0.0, dtypes.float32, mask=p)
            s5 = vdups(0.0, dtypes.float32, mask=p)
            s6 = vdups(0.0, dtypes.float32, mask=p)
            s7 = vdups(0.0, dtypes.float32, mask=p)
            s8 = vdups(0.0, dtypes.float32, mask=p)
            s9 = vdups(0.0, dtypes.float32, mask=p)
            s10 = vdups(0.0, dtypes.float32, mask=p)
            s11 = vdups(0.0, dtypes.float32, mask=p)
            s12 = vdups(0.0, dtypes.float32, mask=p)
            s13 = vdups(0.0, dtypes.float32, mask=p)
            s14 = vdups(0.0, dtypes.float32, mask=p)
            s15 = vdups(0.0, dtypes.float32, mask=p)
            for j in range(self.k//64):
                a = vload(self.xfp32, j*64)
                codes0 = vreinterpret_lanes(vload_unpack(w, (0//(self.base_m//2))*(self.k*self.base_m*9//16)+(j*(self.base_m//2)+0%(self.base_m//2))*64, unpack_mode=UnpackMode.UNPACK4), dtypes.int4x2)
                scale0 = vreinterpret_lanes(vload_broadcast(self.expanded, (0//(self.base_m//2))*(self.k*self.base_m//16)+4*(j*self.base_m+2*(0%(self.base_m//2))), mode="elem2datablock"), dtypes.bfloat16)
                codes1 = vreinterpret_lanes(vload_unpack(w, (1//(self.base_m//2))*(self.k*self.base_m*9//16)+(j*(self.base_m//2)+1%(self.base_m//2))*64, unpack_mode=UnpackMode.UNPACK4), dtypes.int4x2)
                scale1 = vreinterpret_lanes(vload_broadcast(self.expanded, (1//(self.base_m//2))*(self.k*self.base_m//16)+4*(j*self.base_m+2*(1%(self.base_m//2))), mode="elem2datablock"), dtypes.bfloat16)
                if const_expr(self.m > 4):
                    codes2 = vreinterpret_lanes(vload_unpack(w, (2//(self.base_m//2))*(self.k*self.base_m*9//16)+(j*(self.base_m//2)+2%(self.base_m//2))*64, unpack_mode=UnpackMode.UNPACK4), dtypes.int4x2)
                    scale2 = vreinterpret_lanes(vload_broadcast(self.expanded, (2//(self.base_m//2))*(self.k*self.base_m//16)+4*(j*self.base_m+2*(2%(self.base_m//2))), mode="elem2datablock"), dtypes.bfloat16)
                if const_expr(self.m > 6):
                    codes3 = vreinterpret_lanes(vload_unpack(w, (3//(self.base_m//2))*(self.k*self.base_m*9//16)+(j*(self.base_m//2)+3%(self.base_m//2))*64, unpack_mode=UnpackMode.UNPACK4), dtypes.int4x2)
                    scale3 = vreinterpret_lanes(vload_broadcast(self.expanded, (3//(self.base_m//2))*(self.k*self.base_m//16)+4*(j*self.base_m+2*(3%(self.base_m//2))), mode="elem2datablock"), dtypes.bfloat16)
                if const_expr(self.m > 8):
                    codes4 = vreinterpret_lanes(vload_unpack(w, (4//(self.base_m//2))*(self.k*self.base_m*9//16)+(j*(self.base_m//2)+4%(self.base_m//2))*64, unpack_mode=UnpackMode.UNPACK4), dtypes.int4x2)
                    scale4 = vreinterpret_lanes(vload_broadcast(self.expanded, (4//(self.base_m//2))*(self.k*self.base_m//16)+4*(j*self.base_m+2*(4%(self.base_m//2))), mode="elem2datablock"), dtypes.bfloat16)
                if const_expr(self.m > 10):
                    codes5 = vreinterpret_lanes(vload_unpack(w, (5//(self.base_m//2))*(self.k*self.base_m*9//16)+(j*(self.base_m//2)+5%(self.base_m//2))*64, unpack_mode=UnpackMode.UNPACK4), dtypes.int4x2)
                    scale5 = vreinterpret_lanes(vload_broadcast(self.expanded, (5//(self.base_m//2))*(self.k*self.base_m//16)+4*(j*self.base_m+2*(5%(self.base_m//2))), mode="elem2datablock"), dtypes.bfloat16)
                if const_expr(self.m > 12):
                    codes6 = vreinterpret_lanes(vload_unpack(w, (6//(self.base_m//2))*(self.k*self.base_m*9//16)+(j*(self.base_m//2)+6%(self.base_m//2))*64, unpack_mode=UnpackMode.UNPACK4), dtypes.int4x2)
                    scale6 = vreinterpret_lanes(vload_broadcast(self.expanded, (6//(self.base_m//2))*(self.k*self.base_m//16)+4*(j*self.base_m+2*(6%(self.base_m//2))), mode="elem2datablock"), dtypes.bfloat16)
                if const_expr(self.m > 14):
                    codes7 = vreinterpret_lanes(vload_unpack(w, (7//(self.base_m//2))*(self.k*self.base_m*9//16)+(j*(self.base_m//2)+7%(self.base_m//2))*64, unpack_mode=UnpackMode.UNPACK4), dtypes.int4x2)
                    scale7 = vreinterpret_lanes(vload_broadcast(self.expanded, (7//(self.base_m//2))*(self.k*self.base_m//16)+4*(j*self.base_m+2*(7%(self.base_m//2))), mode="elem2datablock"), dtypes.bfloat16)
                weights0 = vcast(codes0, dtypes.bfloat16, mask=pq, reg_layout=RegLayout.ZERO)
                weights1 = vcast(codes1, dtypes.bfloat16, mask=pq, reg_layout=RegLayout.ZERO)
                if const_expr(self.m > 4):
                    weights2 = vcast(codes2, dtypes.bfloat16, mask=pq, reg_layout=RegLayout.ZERO)
                if const_expr(self.m > 6):
                    weights3 = vcast(codes3, dtypes.bfloat16, mask=pq, reg_layout=RegLayout.ZERO)
                if const_expr(self.m > 8):
                    weights4 = vcast(codes4, dtypes.bfloat16, mask=pq, reg_layout=RegLayout.ZERO)
                if const_expr(self.m > 10):
                    weights5 = vcast(codes5, dtypes.bfloat16, mask=pq, reg_layout=RegLayout.ZERO)
                if const_expr(self.m > 12):
                    weights6 = vcast(codes6, dtypes.bfloat16, mask=pq, reg_layout=RegLayout.ZERO)
                if const_expr(self.m > 14):
                    weights7 = vcast(codes7, dtypes.bfloat16, mask=pq, reg_layout=RegLayout.ZERO)
                weights0 = vmul(weights0, scale0, mask=p16)
                weights1 = vmul(weights1, scale1, mask=p16)
                if const_expr(self.m > 4):
                    weights2 = vmul(weights2, scale2, mask=p16)
                if const_expr(self.m > 6):
                    weights3 = vmul(weights3, scale3, mask=p16)
                if const_expr(self.m > 8):
                    weights4 = vmul(weights4, scale4, mask=p16)
                if const_expr(self.m > 10):
                    weights5 = vmul(weights5, scale5, mask=p16)
                if const_expr(self.m > 12):
                    weights6 = vmul(weights6, scale6, mask=p16)
                if const_expr(self.m > 14):
                    weights7 = vmul(weights7, scale7, mask=p16)
                lo0 = vcast(weights0, dtypes.float32, mask=p16, reg_layout=RegLayout.ZERO)
                hi0 = vcast(weights0, dtypes.float32, mask=p16, reg_layout=RegLayout.ONE)
                s0 = vmula(s0, a, lo0, mask=p)
                s1 = vmula(s1, a, hi0, mask=p)
                lo1 = vcast(weights1, dtypes.float32, mask=p16, reg_layout=RegLayout.ZERO)
                hi1 = vcast(weights1, dtypes.float32, mask=p16, reg_layout=RegLayout.ONE)
                s2 = vmula(s2, a, lo1, mask=p)
                s3 = vmula(s3, a, hi1, mask=p)
                if const_expr(self.m > 4):
                    lo2 = vcast(weights2, dtypes.float32, mask=p16, reg_layout=RegLayout.ZERO)
                    hi2 = vcast(weights2, dtypes.float32, mask=p16, reg_layout=RegLayout.ONE)
                    s4 = vmula(s4, a, lo2, mask=p)
                    s5 = vmula(s5, a, hi2, mask=p)
                if const_expr(self.m > 6):
                    lo3 = vcast(weights3, dtypes.float32, mask=p16, reg_layout=RegLayout.ZERO)
                    hi3 = vcast(weights3, dtypes.float32, mask=p16, reg_layout=RegLayout.ONE)
                    s6 = vmula(s6, a, lo3, mask=p)
                    s7 = vmula(s7, a, hi3, mask=p)
                if const_expr(self.m > 8):
                    lo4 = vcast(weights4, dtypes.float32, mask=p16, reg_layout=RegLayout.ZERO)
                    hi4 = vcast(weights4, dtypes.float32, mask=p16, reg_layout=RegLayout.ONE)
                    s8 = vmula(s8, a, lo4, mask=p)
                    s9 = vmula(s9, a, hi4, mask=p)
                if const_expr(self.m > 10):
                    lo5 = vcast(weights5, dtypes.float32, mask=p16, reg_layout=RegLayout.ZERO)
                    hi5 = vcast(weights5, dtypes.float32, mask=p16, reg_layout=RegLayout.ONE)
                    s10 = vmula(s10, a, lo5, mask=p)
                    s11 = vmula(s11, a, hi5, mask=p)
                if const_expr(self.m > 12):
                    lo6 = vcast(weights6, dtypes.float32, mask=p16, reg_layout=RegLayout.ZERO)
                    hi6 = vcast(weights6, dtypes.float32, mask=p16, reg_layout=RegLayout.ONE)
                    s12 = vmula(s12, a, lo6, mask=p)
                    s13 = vmula(s13, a, hi6, mask=p)
                if const_expr(self.m > 14):
                    lo7 = vcast(weights7, dtypes.float32, mask=p16, reg_layout=RegLayout.ZERO)
                    hi7 = vcast(weights7, dtypes.float32, mask=p16, reg_layout=RegLayout.ONE)
                    s14 = vmula(s14, a, lo7, mask=p)
                    s15 = vmula(s15, a, hi7, mask=p)
            if const_expr(self.m == 16):
                r0 = vreduce_sum(s0,mask=p)
                r1 = vreduce_sum(s1,mask=p)
                r2 = vreduce_sum(s2,mask=p)
                r3 = vreduce_sum(s3,mask=p)
                r4 = vreduce_sum(s4,mask=p)
                r5 = vreduce_sum(s5,mask=p)
                r6 = vreduce_sum(s6,mask=p)
                r7 = vreduce_sum(s7,mask=p)
                r8 = vreduce_sum(s8,mask=p)
                r9 = vreduce_sum(s9,mask=p)
                r10 = vreduce_sum(s10,mask=p)
                r11 = vreduce_sum(s11,mask=p)
                r12 = vreduce_sum(s12,mask=p)
                r13 = vreduce_sum(s13,mask=p)
                r14 = vreduce_sum(s14,mask=p)
                r15 = vreduce_sum(s15,mask=p)
                t16_0_0, discarded = vinterleave(r0, r8)
                t16_0_1, discarded = vinterleave(r1, r9)
                t16_0_2, discarded = vinterleave(r2, r10)
                t16_0_3, discarded = vinterleave(r3, r11)
                t16_0_4, discarded = vinterleave(r4, r12)
                t16_0_5, discarded = vinterleave(r5, r13)
                t16_0_6, discarded = vinterleave(r6, r14)
                t16_0_7, discarded = vinterleave(r7, r15)
                t16_1_0, discarded = vinterleave(t16_0_0, t16_0_4)
                t16_1_1, discarded = vinterleave(t16_0_1, t16_0_5)
                t16_1_2, discarded = vinterleave(t16_0_2, t16_0_6)
                t16_1_3, discarded = vinterleave(t16_0_3, t16_0_7)
                t16_2_0, discarded = vinterleave(t16_1_0, t16_1_2)
                t16_2_1, discarded = vinterleave(t16_1_1, t16_1_3)
                t16_3_0, discarded = vinterleave(t16_2_0, t16_2_1)
                values = t16_3_0
            elif const_expr(self.m == 8):
                r0 = vreduce_sum(s0,mask=p)
                r1 = vreduce_sum(s1,mask=p)
                r2 = vreduce_sum(s2,mask=p)
                r3 = vreduce_sum(s3,mask=p)
                r4 = vreduce_sum(s4,mask=p)
                r5 = vreduce_sum(s5,mask=p)
                r6 = vreduce_sum(s6,mask=p)
                r7 = vreduce_sum(s7,mask=p)
                t8_0_0, discarded = vinterleave(r0, r4)
                t8_0_1, discarded = vinterleave(r1, r5)
                t8_0_2, discarded = vinterleave(r2, r6)
                t8_0_3, discarded = vinterleave(r3, r7)
                t8_1_0, discarded = vinterleave(t8_0_0, t8_0_2)
                t8_1_1, discarded = vinterleave(t8_0_1, t8_0_3)
                t8_2_0, discarded = vinterleave(t8_1_0, t8_1_1)
                values = t8_2_0
            elif const_expr(self.m == 4):
                r0 = vreduce_sum(s0,mask=p)
                r1 = vreduce_sum(s1,mask=p)
                r2 = vreduce_sum(s2,mask=p)
                r3 = vreduce_sum(s3,mask=p)
                t4_0_0, discarded = vinterleave(r0, r2)
                t4_0_1, discarded = vinterleave(r1, r3)
                t4_1_0, discarded = vinterleave(t4_0_0, t4_0_1)
                values = t4_1_0
            pm, _ = update_mask(self.m, elem_bits=32)
            out = vcast(values, dtypes.bfloat16, mask=pm, rounding=RoundingMode.RN)
            vstore_pack(y, 0, out, pm, pack_mode=PackMode.B32_TO_B16)

@jit
def norm_ub(x,r,g,y,o,eps):
    with vf(mode='simd'):
        p = create_mask(pattern='all', elem_bits=32)
        ss = vdups(0.0, dtypes.float32, mask=p)
        for j in range(40):
            a = vcast(vload_unpack(x, j*64, unpack_mode=UnpackMode.UNPACK2), dtypes.float32, mask=p)
            b = vcast(vload_unpack(r, j*64, unpack_mode=UnpackMode.UNPACK2), dtypes.float32, mask=p)
            s = vadd(a, b, mask=p)
            ss = vadd(ss, vmul(s, s, mask=p), mask=p)
        ss = vdup(vreduce_sum(ss, mask=p), mask=p)
        root = vsqrt(vadds(vmuls(ss, 1.0/2560.0, mask=p), eps, mask=p), mask=p)
        inverse = vdiv(vdups(1.0, dtypes.float32, mask=p), root, mask=p)
        for j in range(40):
            a = vcast(vload_unpack(x, j*64, unpack_mode=UnpackMode.UNPACK2), dtypes.float32, mask=p)
            b = vcast(vload_unpack(r, j*64, unpack_mode=UnpackMode.UNPACK2), dtypes.float32, mask=p)
            gamma = vcast(vload_unpack(g, j*64, unpack_mode=UnpackMode.UNPACK2), dtypes.float32, mask=p)
            s = vadd(a, b, mask=p)
            rb = vcast(s, dtypes.bfloat16, mask=p, rounding=RoundingMode.RN)
            vstore_pack(o, j*64, rb, p, pack_mode=PackMode.B32_TO_B16)
            z = vmul(vmul(s, inverse, mask=p), gamma, mask=p)
            yb = vcast(z, dtypes.bfloat16, mask=p, rounding=RoundingMode.RN)
            vstore_pack(y, j*64, yb, p, pack_mode=PackMode.B32_TO_B16)

class FusedUpStage(GemvStage):
    @jit
    def activate(self,x,y,count,lut):
        with vf(mode='simd'):
            allp=create_mask(pattern='all',elem_bits=32)
            p16=create_mask(pattern='all',elem_bits=16)
            vmem_bar('vst_vld')
            remaining=count
            for j in range((count+63)//64):
                p,remaining=update_mask(remaining,elem_bits=32)
                ix=varange(j*64,dtypes.int32)
                low=vbitwise_and(ix,vdups(3,dtypes.int32,mask=p),mask=p)
                gi=vreinterpret(vadd(vshl(vshr(ix,2,mask=p),3,mask=p),low,mask=p),dtypes.uint32)
                ui=vadds(gi,4,mask=p)
                gb=vgather(x,gi,mask=p16)
                uf=vcast(vgather(x,ui,mask=p16),dtypes.float32,mask=p)
                bits=vreinterpret_lanes(gb,dtypes.uint32)
                ab=vbitwise_and(bits,vdups(32767,dtypes.uint32,mask=p),mask=p)
                index=vbitwise_and(vadds(ab,4294952192,mask=p),vdups(2047,dtypes.uint32,mask=p),mask=p)
                sign=vbitwise_and(vshr(bits,4,mask=p),vdups(2048,dtypes.uint32,mask=p),mask=p)
                index=vbitwise_or(index,sign,mask=p)
                gf=vcast(vgather(lut,index,mask=p16),dtypes.float32,mask=p)
                xf=vcast(gb,dtypes.float32,mask=p)
                small=vmuls(xf,.5,mask=p)
                large=vselect(xf,vdups(-0.0,dtypes.float32,mask=p),cond_mask=vlts(bits,32768,mask=p))
                gf=vselect(small,gf,cond_mask=vlts(ab,0x3b00,mask=p))
                gf=vselect(large,gf,cond_mask=vges(ab,0x4300,mask=p))
                rounded=vcast(gf,dtypes.bfloat16,mask=p,rounding=RoundingMode.RN)
                rb=vreinterpret_lanes(rounded,dtypes.uint32)
                rb=vselect(vdups(0x8000,dtypes.uint32,mask=p),rb,cond_mask=vges(bits,0xc300,mask=p))
                rb=vselect(vdups(0x7fff,dtypes.uint32,mask=p),rb,cond_mask=vgts(ab,0x7f80,mask=p))
                rb=vselect(vdups(0x7fff,dtypes.uint32,mask=p),rb,cond_mask=veqs(bits,0xff80,mask=p))
                gf=vcast(vreinterpret_lanes(rb,dtypes.bfloat16),dtypes.float32,mask=p)
                out=vcast(vmul(gf,uf,mask=p),dtypes.bfloat16,mask=p,rounding=RoundingMode.RN)
                vstore_pack(y,j*64,out,p,pack_mode=PackMode.B32_TO_B16)

@jit
def prior_vector_reads_complete():
    vec_sync_notify(PIPE.V,PIPE.MTE2,7)
    vec_sync_wait(PIPE.V,PIPE.MTE2,7)

class NativePool:
    def __init__(self,root):
        self.w0=make_buffer(root.reinterpret(dtype=dtypes.uint8,shape=(65536,),offset=0))
        self.w1=make_buffer(root.reinterpret(dtype=dtypes.uint8,shape=(65536,),offset=65536))
        self.upch=make_channel([
            root.reinterpret(dtype=dtypes.uint8,shape=(23040,),offset=0),
            root.reinterpret(dtype=dtypes.uint8,shape=(23040,),offset=65536),
            root.reinterpret(dtype=dtypes.uint8,shape=(23040,),offset=23040),
            root.reinterpret(dtype=dtypes.uint8,shape=(23040,),offset=88576)])
        self.first=make_buffer(root.reinterpret(dtype=dtypes.uint8,shape=(36864,),offset=212992))
        self.x=make_buffer(root.reinterpret(dtype=dtypes.bfloat16,shape=(10240,),offset=131264))
        self.xfp=make_buffer(root.reinterpret(dtype=dtypes.float32,shape=(10240,),offset=151744))
        self.res=make_buffer(root.reinterpret(dtype=dtypes.bfloat16,shape=(2560,),offset=168128))
        self.gamma=make_buffer(root.reinterpret(dtype=dtypes.bfloat16,shape=(2560,),offset=173248))
        self.lut=make_buffer(root.reinterpret(dtype=dtypes.bfloat16,shape=(4096,),offset=178368))
        self.expanded=make_buffer(root.reinterpret(dtype=dtypes.uint32,shape=(4096,),offset=192704))
        self.down_expanded=make_buffer(root.reinterpret(dtype=dtypes.uint32,shape=(5120,),offset=212992))
        self.batch=make_buffer(root.reinterpret(dtype=dtypes.bfloat16,shape=(24,16),offset=209088))
        self.upbatch=make_buffer(root.reinterpret(dtype=dtypes.bfloat16,shape=(24,32),offset=209088))
        # Merge finishes before any O/UP/DOWN writes into the low weight arena.
        self.scratch=make_buffer(root.reinterpret(dtype=dtypes.float32,shape=(16896,),offset=0))
        self.merge_g=make_buffer(root.reinterpret(dtype=dtypes.bfloat16,shape=(1,),offset=67584))
        self.merge_y=make_buffer(root.reinterpret(dtype=dtypes.bfloat16,shape=(256,),offset=67616))
        self.merge_weights=make_buffer(root.reinterpret(dtype=dtypes.float32,shape=(64,),offset=68128))

    @jit
    def prefetch_metadata(self,res,gamma,lut):
        mem_copy(self.res,res)
        mem_copy(self.gamma,gamma)
        mem_copy(self.lut,lut)

    @jit
    def final_norm(self,xgm,resgm,gammagm,y,ro,eps):
        mem_copy(self.x[0:2560,],xgm)
        mem_copy(self.res,resgm)
        mem_copy(self.gamma,gammagm)
        norm_ub(self.x[0:2560,],self.res,self.gamma,self.x[0:2560,],self.res,eps)
        mem_copy(y,self.x[0:2560,])
        mem_copy(ro,self.res)

class NativeLinear(GemvStage):
    def __init__(self,k,m,pool):
        self.k,self.m=k,m
        self.base_m=4 if k==10240 else 8
        self.pool=pool
        self.xfp32=pool.xfp
        self.expanded=pool.down_expanded if k==10240 else pool.expanded

    @jit
    def expand_input(self,xgm):
        mem_copy(self.pool.x[0:self.k,],xgm)
        with vf(mode='simd'):
            p=create_mask(pattern='all',elem_bits=32)
            for j in range(self.k//64):
                a=vcast(vload_unpack(self.pool.x,j*64,unpack_mode=UnpackMode.UNPACK2),dtypes.float32,mask=p)
                vstore(self.xfp32,j*64,a,p)
            vmem_bar('vst_vld')

    @jit
    def load_slot(self,w,row,slot):
        size=self.k*self.m*9//16
        if slot==0:
            mem_copy(self.pool.w0[0:size,].view(1,size),tile_slice(w,(size,),(row,)).view(1,size),l2_cache_ctl=4)
        else:
            mem_copy(self.pool.w1[0:size,].view(1,size),tile_slice(w,(size,),(row,)).view(1,size),l2_cache_ctl=4)

    @jit
    def prefetch_o(self,w,lane):
        begin=160*lane//56
        mem_copy(self.pool.first.view(1,36864),tile_slice(w,(36864,),(begin,)).view(1,36864),l2_cache_ctl=4)

    @jit
    def execute_o(self,x,w,y,lane,up,next_w):
        self.expand_input(x)
        begin=160*lane//56
        end=160*(lane+1)//56
        tiles=end-begin
        # Rotate the two-slot ring so the final O tile reads the upper half;
        # the lower half can then carry the first UP tile without an extra copy.
        base=tiles%2
        for i in range(tiles):
            if i+1<tiles:self.load_slot(w,begin+i+1,(base+i+1)%2)
            if i+1==tiles:
                prior_vector_reads_complete()
                up.prefetch_first_weight(next_w,lane)
            if i==0:
                self.calculate(self.pool.first,self.pool.batch[i,None].view(16))
            elif (base+i)%2==0:
                self.calculate(self.pool.w0,self.pool.batch[i,None].view(16))
            else:
                self.calculate(self.pool.w1,self.pool.batch[i,None].view(16))
        mem_copy(y[begin*16:end*16,].view(end-begin,16),self.pool.batch[0:end-begin,0:16])

    @jit
    def execute_down(self,x,w,y,lane,first_slot):
        self.expand_input(x)
        begin=320*lane//56
        end=320*(lane+1)//56
        for i in range(end-begin):
            slot=(first_slot+i)%2
            if begin+i+1<end:self.load_slot(w,begin+i+1,1-slot)
            if slot==0:self.calculate(self.pool.w0,self.pool.batch[i,None].view(16))
            else:self.calculate(self.pool.w1,self.pool.batch[i,None].view(16))
        mem_copy(y[begin*8:end*8,].view(end-begin,8),self.pool.batch[0:end-begin,0:8])

class NativeUp(FusedUpStage):
    def __init__(self,pool):
        self.k,self.m,self.base_m=2560,16,8
        self.pool=pool
        self.xfp32=pool.xfp
        self.expanded=pool.expanded

    @jit
    def calculate(self, w, y):
        scales = reinterpret(w, dtype=dtypes.uint32, shape=(self.k*self.m*9//64,))
        with vf(mode="simd"):
            p = create_mask(pattern="all", elem_bits=32)
            p16 = create_mask(pattern="all", elem_bits=16)
            pq = create_mask(pattern="all", elem_bits=8)
            # E2B broadcasts repeat each uint32 eight times. Selecting even
            # lanes of two loads packs exactly the original fourfold expansion.
            for si in range(self.k*self.base_m//1024):
                for group in tuple(range(self.m//self.base_m)):
                    base=group*(self.k*self.base_m*9//64)+self.k*self.base_m//8+si*16
                    p0=vload_broadcast(scales,base,mode="elem2datablock")
                    p1=vload_broadcast(scales,base+8,mode="elem2datablock")
                    packed,unused=vdeinterleave(p0,p1)
                    vstore(self.expanded,group*(self.k*self.base_m//16)+si*64,packed,p)
            vmem_bar("vst_vld")
            s0 = vdups(0.0, dtypes.float32, mask=p)
            s1 = vdups(0.0, dtypes.float32, mask=p)
            s2 = vdups(0.0, dtypes.float32, mask=p)
            s3 = vdups(0.0, dtypes.float32, mask=p)
            s4 = vdups(0.0, dtypes.float32, mask=p)
            s5 = vdups(0.0, dtypes.float32, mask=p)
            s6 = vdups(0.0, dtypes.float32, mask=p)
            s7 = vdups(0.0, dtypes.float32, mask=p)
            s8 = vdups(0.0, dtypes.float32, mask=p)
            s9 = vdups(0.0, dtypes.float32, mask=p)
            s10 = vdups(0.0, dtypes.float32, mask=p)
            s11 = vdups(0.0, dtypes.float32, mask=p)
            s12 = vdups(0.0, dtypes.float32, mask=p)
            s13 = vdups(0.0, dtypes.float32, mask=p)
            s14 = vdups(0.0, dtypes.float32, mask=p)
            s15 = vdups(0.0, dtypes.float32, mask=p)
            for j in range(self.k//64):
                a = vload(self.xfp32, j*64)
                codes0 = vreinterpret_lanes(vload_unpack(w, (0//(self.base_m//2))*(self.k*self.base_m*9//16)+(j*(self.base_m//2)+0%(self.base_m//2))*64, unpack_mode=UnpackMode.UNPACK4), dtypes.int4x2)
                scale0 = vreinterpret_lanes(vload_broadcast(self.expanded, (0//(self.base_m//2))*(self.k*self.base_m//16)+4*(j*self.base_m+2*(0%(self.base_m//2))), mode="elem2datablock"), dtypes.bfloat16)
                codes1 = vreinterpret_lanes(vload_unpack(w, (1//(self.base_m//2))*(self.k*self.base_m*9//16)+(j*(self.base_m//2)+1%(self.base_m//2))*64, unpack_mode=UnpackMode.UNPACK4), dtypes.int4x2)
                scale1 = vreinterpret_lanes(vload_broadcast(self.expanded, (1//(self.base_m//2))*(self.k*self.base_m//16)+4*(j*self.base_m+2*(1%(self.base_m//2))), mode="elem2datablock"), dtypes.bfloat16)
                if const_expr(self.m > 4):
                    codes2 = vreinterpret_lanes(vload_unpack(w, (2//(self.base_m//2))*(self.k*self.base_m*9//16)+(j*(self.base_m//2)+2%(self.base_m//2))*64, unpack_mode=UnpackMode.UNPACK4), dtypes.int4x2)
                    scale2 = vreinterpret_lanes(vload_broadcast(self.expanded, (2//(self.base_m//2))*(self.k*self.base_m//16)+4*(j*self.base_m+2*(2%(self.base_m//2))), mode="elem2datablock"), dtypes.bfloat16)
                if const_expr(self.m > 6):
                    codes3 = vreinterpret_lanes(vload_unpack(w, (3//(self.base_m//2))*(self.k*self.base_m*9//16)+(j*(self.base_m//2)+3%(self.base_m//2))*64, unpack_mode=UnpackMode.UNPACK4), dtypes.int4x2)
                    scale3 = vreinterpret_lanes(vload_broadcast(self.expanded, (3//(self.base_m//2))*(self.k*self.base_m//16)+4*(j*self.base_m+2*(3%(self.base_m//2))), mode="elem2datablock"), dtypes.bfloat16)
                if const_expr(self.m > 8):
                    codes4 = vreinterpret_lanes(vload_unpack(w, (4//(self.base_m//2))*(self.k*self.base_m*9//16)+(j*(self.base_m//2)+4%(self.base_m//2))*64, unpack_mode=UnpackMode.UNPACK4), dtypes.int4x2)
                    scale4 = vreinterpret_lanes(vload_broadcast(self.expanded, (4//(self.base_m//2))*(self.k*self.base_m//16)+4*(j*self.base_m+2*(4%(self.base_m//2))), mode="elem2datablock"), dtypes.bfloat16)
                if const_expr(self.m > 10):
                    codes5 = vreinterpret_lanes(vload_unpack(w, (5//(self.base_m//2))*(self.k*self.base_m*9//16)+(j*(self.base_m//2)+5%(self.base_m//2))*64, unpack_mode=UnpackMode.UNPACK4), dtypes.int4x2)
                    scale5 = vreinterpret_lanes(vload_broadcast(self.expanded, (5//(self.base_m//2))*(self.k*self.base_m//16)+4*(j*self.base_m+2*(5%(self.base_m//2))), mode="elem2datablock"), dtypes.bfloat16)
                if const_expr(self.m > 12):
                    codes6 = vreinterpret_lanes(vload_unpack(w, (6//(self.base_m//2))*(self.k*self.base_m*9//16)+(j*(self.base_m//2)+6%(self.base_m//2))*64, unpack_mode=UnpackMode.UNPACK4), dtypes.int4x2)
                    scale6 = vreinterpret_lanes(vload_broadcast(self.expanded, (6//(self.base_m//2))*(self.k*self.base_m//16)+4*(j*self.base_m+2*(6%(self.base_m//2))), mode="elem2datablock"), dtypes.bfloat16)
                if const_expr(self.m > 14):
                    codes7 = vreinterpret_lanes(vload_unpack(w, (7//(self.base_m//2))*(self.k*self.base_m*9//16)+(j*(self.base_m//2)+7%(self.base_m//2))*64, unpack_mode=UnpackMode.UNPACK4), dtypes.int4x2)
                    scale7 = vreinterpret_lanes(vload_broadcast(self.expanded, (7//(self.base_m//2))*(self.k*self.base_m//16)+4*(j*self.base_m+2*(7%(self.base_m//2))), mode="elem2datablock"), dtypes.bfloat16)
                weights0 = vcast(codes0, dtypes.bfloat16, mask=pq, reg_layout=RegLayout.ZERO)
                weights1 = vcast(codes1, dtypes.bfloat16, mask=pq, reg_layout=RegLayout.ZERO)
                if const_expr(self.m > 4):
                    weights2 = vcast(codes2, dtypes.bfloat16, mask=pq, reg_layout=RegLayout.ZERO)
                if const_expr(self.m > 6):
                    weights3 = vcast(codes3, dtypes.bfloat16, mask=pq, reg_layout=RegLayout.ZERO)
                if const_expr(self.m > 8):
                    weights4 = vcast(codes4, dtypes.bfloat16, mask=pq, reg_layout=RegLayout.ZERO)
                if const_expr(self.m > 10):
                    weights5 = vcast(codes5, dtypes.bfloat16, mask=pq, reg_layout=RegLayout.ZERO)
                if const_expr(self.m > 12):
                    weights6 = vcast(codes6, dtypes.bfloat16, mask=pq, reg_layout=RegLayout.ZERO)
                if const_expr(self.m > 14):
                    weights7 = vcast(codes7, dtypes.bfloat16, mask=pq, reg_layout=RegLayout.ZERO)
                weights0 = vmul(weights0, scale0, mask=p16)
                weights1 = vmul(weights1, scale1, mask=p16)
                if const_expr(self.m > 4):
                    weights2 = vmul(weights2, scale2, mask=p16)
                if const_expr(self.m > 6):
                    weights3 = vmul(weights3, scale3, mask=p16)
                if const_expr(self.m > 8):
                    weights4 = vmul(weights4, scale4, mask=p16)
                if const_expr(self.m > 10):
                    weights5 = vmul(weights5, scale5, mask=p16)
                if const_expr(self.m > 12):
                    weights6 = vmul(weights6, scale6, mask=p16)
                if const_expr(self.m > 14):
                    weights7 = vmul(weights7, scale7, mask=p16)
                lo0 = vcast(weights0, dtypes.float32, mask=p16, reg_layout=RegLayout.ZERO)
                hi0 = vcast(weights0, dtypes.float32, mask=p16, reg_layout=RegLayout.ONE)
                s0 = vmula(s0, a, lo0, mask=p)
                s1 = vmula(s1, a, hi0, mask=p)
                lo1 = vcast(weights1, dtypes.float32, mask=p16, reg_layout=RegLayout.ZERO)
                hi1 = vcast(weights1, dtypes.float32, mask=p16, reg_layout=RegLayout.ONE)
                s2 = vmula(s2, a, lo1, mask=p)
                s3 = vmula(s3, a, hi1, mask=p)
                if const_expr(self.m > 4):
                    lo2 = vcast(weights2, dtypes.float32, mask=p16, reg_layout=RegLayout.ZERO)
                    hi2 = vcast(weights2, dtypes.float32, mask=p16, reg_layout=RegLayout.ONE)
                    s4 = vmula(s4, a, lo2, mask=p)
                    s5 = vmula(s5, a, hi2, mask=p)
                if const_expr(self.m > 6):
                    lo3 = vcast(weights3, dtypes.float32, mask=p16, reg_layout=RegLayout.ZERO)
                    hi3 = vcast(weights3, dtypes.float32, mask=p16, reg_layout=RegLayout.ONE)
                    s6 = vmula(s6, a, lo3, mask=p)
                    s7 = vmula(s7, a, hi3, mask=p)
                if const_expr(self.m > 8):
                    lo4 = vcast(weights4, dtypes.float32, mask=p16, reg_layout=RegLayout.ZERO)
                    hi4 = vcast(weights4, dtypes.float32, mask=p16, reg_layout=RegLayout.ONE)
                    s8 = vmula(s8, a, lo4, mask=p)
                    s9 = vmula(s9, a, hi4, mask=p)
                if const_expr(self.m > 10):
                    lo5 = vcast(weights5, dtypes.float32, mask=p16, reg_layout=RegLayout.ZERO)
                    hi5 = vcast(weights5, dtypes.float32, mask=p16, reg_layout=RegLayout.ONE)
                    s10 = vmula(s10, a, lo5, mask=p)
                    s11 = vmula(s11, a, hi5, mask=p)
                if const_expr(self.m > 12):
                    lo6 = vcast(weights6, dtypes.float32, mask=p16, reg_layout=RegLayout.ZERO)
                    hi6 = vcast(weights6, dtypes.float32, mask=p16, reg_layout=RegLayout.ONE)
                    s12 = vmula(s12, a, lo6, mask=p)
                    s13 = vmula(s13, a, hi6, mask=p)
                if const_expr(self.m > 14):
                    lo7 = vcast(weights7, dtypes.float32, mask=p16, reg_layout=RegLayout.ZERO)
                    hi7 = vcast(weights7, dtypes.float32, mask=p16, reg_layout=RegLayout.ONE)
                    s14 = vmula(s14, a, lo7, mask=p)
                    s15 = vmula(s15, a, hi7, mask=p)
            r0=vreduce_sum(s0,mask=p)
            r1=vreduce_sum(s1,mask=p)
            r2=vreduce_sum(s2,mask=p)
            r3=vreduce_sum(s3,mask=p)
            r4=vreduce_sum(s4,mask=p)
            r5=vreduce_sum(s5,mask=p)
            r6=vreduce_sum(s6,mask=p)
            r7=vreduce_sum(s7,mask=p)
            r8=vreduce_sum(s8,mask=p)
            r9=vreduce_sum(s9,mask=p)
            r10=vreduce_sum(s10,mask=p)
            r11=vreduce_sum(s11,mask=p)
            r12=vreduce_sum(s12,mask=p)
            r13=vreduce_sum(s13,mask=p)
            r14=vreduce_sum(s14,mask=p)
            r15=vreduce_sum(s15,mask=p)
            p0_0_0,unused=vinterleave(r0,r4)
            p0_0_1,unused=vinterleave(r1,r5)
            p0_0_2,unused=vinterleave(r2,r6)
            p0_0_3,unused=vinterleave(r3,r7)
            p0_1_0,unused=vinterleave(p0_0_0,p0_0_2)
            p0_1_1,unused=vinterleave(p0_0_1,p0_0_3)
            p0_2_0,unused=vinterleave(p0_1_0,p0_1_1)
            pm,_=update_mask(8,elem_bits=32)
            out=vcast(p0_2_0,dtypes.bfloat16,mask=pm,rounding=RoundingMode.RN)
            vstore_pack(y,0,out,pm,pack_mode=PackMode.B32_TO_B16)
            p1_0_0,unused=vinterleave(r8,r12)
            p1_0_1,unused=vinterleave(r9,r13)
            p1_0_2,unused=vinterleave(r10,r14)
            p1_0_3,unused=vinterleave(r11,r15)
            p1_1_0,unused=vinterleave(p1_0_0,p1_0_2)
            p1_1_1,unused=vinterleave(p1_0_1,p1_0_3)
            p1_2_0,unused=vinterleave(p1_1_0,p1_1_1)
            pm,_=update_mask(8,elem_bits=32)
            out=vcast(p1_2_0,dtypes.bfloat16,mask=pm,rounding=RoundingMode.RN)
            vstore_pack(y,16,out,pm,pack_mode=PackMode.B32_TO_B16)

    @jit
    def activate_native(self,x,count,lut):
        with vf(mode='simd'):
            allp=create_mask(pattern='all',elem_bits=32)
            p16=create_mask(pattern='all',elem_bits=16)
            vmem_bar('vst_vld')
            halves=varange(0,dtypes.uint16)
            odd=vbitwise_and(halves,vdups(1,dtypes.uint16,mask=p16),mask=p16)
            even=veqs(odd,0,mask=p16)
            remaining=count
            for j in range((count+63)//64):
                p,remaining=update_mask(remaining,elem_bits=32)
                ix=varange(j*64,dtypes.int32)
                low=vbitwise_and(ix,vdups(3,dtypes.int32,mask=p),mask=p)
                gi=vreinterpret(vadd(vshl(vshr(ix,2,mask=p),4,mask=p),low,mask=p),dtypes.uint32)
                ui=vadds(gi,4,mask=p)
                gb=vgather(x,gi,mask=p16)
                uf=vcast(vgather(x,ui,mask=p16),dtypes.float32,mask=p)
                bits=vreinterpret_lanes(gb,dtypes.uint32)
                ab=vbitwise_and(bits,vdups(32767,dtypes.uint32,mask=p),mask=p)
                index=vbitwise_and(vadds(ab,4294952192,mask=p),vdups(2047,dtypes.uint32,mask=p),mask=p)
                sign=vbitwise_and(vshr(bits,4,mask=p),vdups(2048,dtypes.uint32,mask=p),mask=p)
                index=vbitwise_or(index,sign,mask=p)
                gf=vcast(vgather(lut,index,mask=p16),dtypes.float32,mask=p)
                xf=vcast(gb,dtypes.float32,mask=p)
                small=vmuls(xf,.5,mask=p)
                large=vselect(xf,vdups(-0.0,dtypes.float32,mask=p),cond_mask=vlts(bits,32768,mask=p))
                gf=vselect(small,gf,cond_mask=vlts(ab,0x3b00,mask=p))
                gf=vselect(large,gf,cond_mask=vges(ab,0x4300,mask=p))
                rounded=vcast(gf,dtypes.bfloat16,mask=p,rounding=RoundingMode.RN)
                rb=vreinterpret_lanes(rounded,dtypes.uint32)
                rb=vselect(vdups(0x8000,dtypes.uint32,mask=p),rb,cond_mask=vges(bits,0xc300,mask=p))
                rb=vselect(vdups(0x7fff,dtypes.uint32,mask=p),rb,cond_mask=vgts(ab,0x7f80,mask=p))
                rb=vselect(vdups(0x7fff,dtypes.uint32,mask=p),rb,cond_mask=veqs(bits,0xff80,mask=p))
                gf=vcast(vreinterpret_lanes(rb,dtypes.bfloat16),dtypes.float32,mask=p)
                out=vcast(vmul(gf,uf,mask=p),dtypes.bfloat16,mask=p,rounding=RoundingMode.RN)
                live16,_=update_mask((count-j*64)*2,elem_bits=16)
                storemask=mask_and(live16,even,exec_mask=p16)
                vscatter(x,out,vreinterpret_lanes(gi,dtypes.uint16),mask=storemask)

    @jit
    def prefetch_first_weight(self,w,lane):
        begin=1280*lane//56
        mem_copy(self.pool.upch.produce().view(1,23040),tile_slice(w,(23040,),(begin,)).view(1,23040),l2_cache_ctl=4)

    @jit
    def execute(self,xgm,w,resout,mid,lane,eps,dn,next_w):
        mem_copy(self.pool.x[0:2560,],xgm)
        norm_ub(self.pool.x[0:2560,],self.pool.res,self.pool.gamma,self.pool.x[0:2560,],self.pool.res,eps)
        if lane==0:mem_copy(resout,self.pool.res)
        with vf(mode='simd'):
            p=create_mask(pattern='all',elem_bits=32)
            vmem_bar('vst_vld')
            for j in range(40):
                a=vcast(vload_unpack(self.pool.x,j*64,unpack_mode=UnpackMode.UNPACK2),dtypes.float32,mask=p)
                vstore(self.xfp32,j*64,a,p)
            vmem_bar('vst_vld')
        begin=1280*lane//56
        end=1280*(lane+1)//56
        for i in range(1,4):
            mem_copy(self.pool.upch.produce().view(1,23040),tile_slice(w,(23040,),(begin+i,)).view(1,23040),l2_cache_ctl=4)
        down_slot=1-((end-begin-1)%2)
        for i in range(end-begin):
            wt=self.pool.upch.consume()
            if begin+i+1==end:
                prior_vector_reads_complete()
                dn.load_slot(next_w,320*lane//56,down_slot)
            self.calculate(wt,self.pool.upbatch[i,None].view(32))
            if begin+i+4<end:
                mem_copy(self.pool.upch.produce().view(1,23040),tile_slice(w,(23040,),(begin+i+4,)).view(1,23040),l2_cache_ctl=4)
        self.activate_native(self.pool.upbatch.view(768),(end-begin)*8,self.pool.lut)
        mem_copy(mid[begin*8:end*8,].view((end-begin)*2,4),self.pool.upbatch.view(48,16)[0:(end-begin)*2,0:4])
        return down_slot
