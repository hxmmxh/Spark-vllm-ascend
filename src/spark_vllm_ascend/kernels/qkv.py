"""M16 QKV with local RoPE and direct BF16 cache stores; packed weights unchanged."""
from spark_vllm_ascend.kernels.linear import GemvStage
from cannbotdsl import dtypes
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ops.reg import PackMode, UnpackMode, RegLayout, RoundingMode, create_mask, update_mask, vcast, vload_unpack, vload_broadcast, vreinterpret, vreinterpret_lanes, vmul, vstore_pack
from cannbotdsl.tensor import reinterpret
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.ops.reg import *
from cannbotdsl.tensor import tile_slice

class FusedQkvStage(GemvStage):
    def __init__(self,rd,arena):
        self.k,self.m,self.base_m=2560,16,8
        self.rd=rd
        self.wch=arena.channel((23040,),dtypes.uint8,depth=2)
        self.xch=arena.channel((2560,),dtypes.bfloat16,depth=1)
        self.ych=arena.channel((32,),dtypes.bfloat16,depth=4)
        self.xfp32=arena.buffer((2560,),dtypes.float32)
        self.expanded=arena.buffer((2560,),dtypes.uint32)
        self.cch=arena.channel((rd,),dtypes.bfloat16,depth=1)

    @jit
    def calculate_rotated(self, w, y, cs, off):
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
            ra0_0,discard=vinterleave(r0,r8)
            ra0_1,discard=vinterleave(r1,r9)
            ra0_2,discard=vinterleave(r2,r10)
            ra0_3,discard=vinterleave(r3,r11)
            ra1_0,discard=vinterleave(ra0_0,ra0_2)
            ra1_1,discard=vinterleave(ra0_1,ra0_3)
            ra2_0,discard=vinterleave(ra1_0,ra1_1)
            rb0_0,discard=vinterleave(r4,r12)
            rb0_1,discard=vinterleave(r5,r13)
            rb0_2,discard=vinterleave(r6,r14)
            rb0_3,discard=vinterleave(r7,r15)
            rb1_0,discard=vinterleave(rb0_0,rb0_2)
            rb1_1,discard=vinterleave(rb0_1,rb0_3)
            rb2_0,discard=vinterleave(rb1_0,rb1_1)
            pm,_=update_mask(8,elem_bits=32)
            pm16,_=update_mask(16,elem_bits=16)
            ab=vcast(ra2_0,dtypes.bfloat16,mask=pm,rounding=RoundingMode.RN)
            bb=vcast(rb2_0,dtypes.bfloat16,mask=pm,rounding=RoundingMode.RN)
            a=vcast(ab,dtypes.float32,mask=pm)
            b=vcast(bb,dtypes.float32,mask=pm)
            ix=vreinterpret(vadds(varange(0,dtypes.int32),off//2,mask=pm),dtypes.uint32)
            cv=vcast(vgather(cs,ix,mask=pm16),dtypes.float32,mask=pm)
            sv=vcast(vgather(cs,vadds(ix,self.rd//2,mask=pm),mask=pm16),dtypes.float32,mask=pm)
            u=vsub(vmul(a,cv,mask=pm),vmul(b,sv,mask=pm),mask=pm)
            v=vadd(vmul(b,cv,mask=pm),vmul(a,sv,mask=pm),mask=pm)
            vstore_pack(y,0,vcast(u,dtypes.bfloat16,mask=pm,rounding=RoundingMode.RN),pm,pack_mode=PackMode.B32_TO_B16)
            vstore_pack(y,16,vcast(v,dtypes.bfloat16,mask=pm,rounding=RoundingMode.RN),pm,pack_mode=PackMode.B32_TO_B16)

    @jit
    def store8(self,y,qkv,kc,nk,head,off,slot):
        mem_copy(tile_slice(qkv.view(-1),(8,),(head*32+off//8,)).view(1,8),y.view(1,8))
        if head>=16 and slot>=0:
            mem_copy(tile_slice(kc.view(-1),(8,),((slot*1024+(head-16)*256+off)//8,)).view(1,8),y.view(1,8))
            if const_expr(self.rd==64):
                noff=(slot//128)*128*1024+(head-16)*128*256+(off//16)*128*16+(slot%128)*16+off%16
                mem_copy(tile_slice(nk.view(-1),(8,),(noff//8,)).view(1,8),y.view(1,8))

    @jit
    def execute_fused(self,x,w,rawgm,csgm,qkv,kc,vc,nk,nv,slot,first,stride):
        mem_copy(self.xch.produce(),x)
        mem_copy(self.cch.produce(),csgm)
        xx=self.xch.consume();cs=self.cch.consume()
        with vf(mode='simd'):
            p=create_mask(pattern='all',elem_bits=32)
            for j in range(40):
                vstore(self.xfp32,j*64,vcast(vload_unpack(xx,j*64,unpack_mode=UnpackMode.UNPACK2),dtypes.float32,mask=p),p)
            vmem_bar('vst_vld')
        if first<384:
            mem_copy(self.wch.produce().view(1,23040),tile_slice(w,(23040,),(first,)).view(1,23040),l2_cache_ctl=4)
        for row in range(first,384,stride):
            if row+stride<384:
                mem_copy(self.wch.produce().view(1,23040),tile_slice(w,(23040,),(row+stride,)).view(1,23040),l2_cache_ctl=4)
            wt=self.wch.consume();yt=self.ych.produce()
            head=row//16;off=(row%16)*16
            if head<20 and off<self.rd:
                self.calculate_rotated(wt,yt,cs,off)
                rr=self.ych.consume()
                self.store8(rr[0:8,],qkv,kc,nk,head,off//2,slot)
                self.store8(rr[16:24,],qkv,kc,nk,head,off//2+self.rd//2,slot)
            else:
                self.calculate(wt,yt)
                raw=self.ych.consume()[0:16,]
                mem_copy(tile_slice(qkv.view(-1),(16,),(row,)).view(1,16),raw.view(1,16))
                if slot>=0:
                    if head>=16 and head<20:
                        mem_copy(tile_slice(kc.view(-1),(16,),((slot*1024+(head-16)*256+off)//16,)).view(1,16),raw.view(1,16))
                        if const_expr(self.rd==64):
                            noff=(slot//128)*128*1024+(head-16)*128*256+(off//16)*128*16+(slot%128)*16
                            mem_copy(tile_slice(nk.view(-1),(16,),(noff//16,)).view(1,16),raw.view(1,16))
                    elif head>=20:
                        mem_copy(tile_slice(vc.view(-1),(16,),((slot*1024+(head-20)*256+off)//16,)).view(1,16),raw.view(1,16))
                        if const_expr(self.rd==64):
                            noff=(slot//128)*128*1024+(head-20)*128*256+(off//16)*128*16+(slot%128)*16
                            mem_copy(tile_slice(nv.view(-1),(16,),(noff//16,)).view(1,16),raw.view(1,16))

    @jit
    def gate_from_expanded_input(self,wgm,outgm):
        wt=reinterpret(self.wch.produce(),dtype=dtypes.bfloat16,shape=(2560,))
        mem_copy(wt,wgm,l2_cache_ctl=4)
        w=reinterpret(self.wch.consume(),dtype=dtypes.bfloat16,shape=(2560,))
        y=self.ych.produce()
        with vf(mode='simd'):
            p=create_mask(pattern='all',elem_bits=32)
            p1,_=update_mask(1,elem_bits=32)
            acc=vdups(0.0,dtypes.float32,mask=p)
            for j in range(40):
                a=vload(self.xfp32,j*64)
                b=vcast(vload_unpack(w,j*64,unpack_mode=UnpackMode.UNPACK2),dtypes.float32,mask=p)
                acc=vmula(acc,a,b,mask=p)
            gb=vcast(vreduce_sum(acc,mask=p),dtypes.bfloat16,mask=p1,rounding=RoundingMode.RN)
            vstore_pack(y,0,gb,p1,pack_mode=PackMode.B32_TO_B16)
        mem_copy(outgm.view(1,1),self.ych.consume()[0:1,].view(1,1))
