"""Batch-one Full/SWA split-KV attention, with FP32 partials and LSE.

The initial DSL implementation retains the baseline partition/tile widths and
BF16 probability rounding. Channel synchronization replaces hand-written flags.
"""
from cannbotdsl import Channel, ChannelKind, MemLoc, dtypes
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.arch import get_block_idx, get_block_num, get_subblock_id
from cannbotdsl.ops.memcpy import mem_copy, make_copy_engine
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.scalar import load_bypass
from cannbotdsl.ops.reg import *
from cannbotdsl.tensor import tile_slice, reinterpret

class AttentionStage:
    def __init__(self, full, arena, debug=False):
        self.debug=debug
        self.full=full
        self.n=384 if full else 128
        self.width=5120 if full else 128
        self.q=Channel(MemLoc.L1,(4,256),dtypes.bfloat16,depth=1)
        if full:
            # Two phase-ordered slots shared by K and V, as in native FullStage.
            self.kv=Channel(MemLoc.L1,(self.n,256),dtypes.bfloat16,depth=2)
        else:
            self.k=Channel(MemLoc.L1,(self.n,256),dtypes.bfloat16,depth=1)
            self.v=Channel(MemLoc.L1,(self.n,256),dtypes.bfloat16,depth=1)
        self.a=Buffer(MemLoc.L0A,(4096,),dtypes.bfloat16,data_format="nd")
        self.b=Buffer(MemLoc.L0B,(32768,),dtypes.bfloat16,data_format="nd")
        self.c=Channel(MemLoc.L0C,(4,self.n),dtypes.float32,depth=1)
        self.cv=Channel(MemLoc.L0C,(4,256),dtypes.float32,depth=1)
        self.scores=arena.channel((2,self.n),dtypes.float32,depth=2 if full else 1,kind=ChannelKind.CrossCore)
        self.p=Channel(MemLoc.L1,(self.n//16,256),dtypes.bfloat16,depth=2 if full else 1,kind=ChannelKind.CrossCore,data_format="nd")
        self.pu=arena.channel((2,self.n),dtypes.bfloat16,depth=1,data_format="nd")
        self.pv=arena.channel((2,256),dtypes.float32,depth=1,kind=ChannelKind.CrossCore)
        self.state_rows=8 if full else 6
        self.state=arena.buffer((self.state_rows,64),dtypes.float32)
        self.acc=arena.buffer((2,256),dtypes.float32)
        self.out=arena.channel((2,256),dtypes.float32,depth=1)
        self.lse=arena.channel((2,8),dtypes.float32,depth=1)
        self.nd2nz=make_copy_engine(format_transform='nd2nz')
        self.split=make_copy_engine(split_axis=0)
        self.probcopy=make_copy_engine(format_transform="nd2nz",split_axis=0)

    @jit
    def initialize(self):
        with vf(mode='simd'):
            p=create_mask(pattern='all',elem_bits=32)
            z=vdups(0.0,dtypes.float32,mask=p)
            neg=vdups(-1.e30,dtypes.float32,mask=p)
            for i in range(2):vstore(self.state,i*64,neg,p)
            for i in range(2,self.state_rows):vstore(self.state,i*64,z,p)
            for i in range(8):vstore(self.acc,i*64,z,p)

    @jit
    def softmax(self, score, prob, start, valid, lower, slot):
        with vf(mode='simd'):
            p=create_mask(pattern='all',elem_bits=32)
            for row in range(2):
                neg=vdups(-1.e30,dtypes.float32,mask=p)
                mx=vdups(-1.e30,dtypes.float32,mask=p)
                for j in range(self.n//64):
                    x=vmuls(vload(score,row*self.n+j*64),.0625,mask=p)
                    ix=varange(j*64,dtypes.int32)
                    x=vselect(x,neg,cond_mask=vlts(ix,valid,mask=p))
                    x=vselect(x,neg,cond_mask=vges(ix,lower-start,mask=p))
                    mx=vmax(mx,x,mask=p)
                    vstore(score,row*self.n+j*64,x,p)
                mx=vdup(vreduce_max(mx,mask=p),mask=p)
                oldm=vload(self.state,row*64)
                olds=vload(self.state,(row+2)*64)
                mx=vmax(mx,oldm,mask=p)
                alpha=vexp(vsub(oldm,mx,mask=p),mask=p)
                total=vdups(0.0,dtypes.float32,mask=p)
                vmem_bar('vst_vld')
                for j in range(self.n//64):
                    x=vexp(vsub(vload(score,row*self.n+j*64),mx,mask=p),mask=p)
                    total=vadd(total,x,mask=p)
                    vstore(score,row*self.n+j*64,x,p)
                    pb=vcast(x,dtypes.bfloat16,mask=p,rounding=RoundingMode.RN)
                    vstore_pack(prob,row*self.n+j*64,pb,p,pack_mode=PackMode.B32_TO_B16)
                total=vdup(vreduce_sum(total,mask=p),mask=p)
                new_sum=vadd(vmul(olds,alpha,mask=p),total,mask=p)
                vstore(self.state,row*64,mx,p)
                vstore(self.state,(row+2)*64,new_sum,p)
                vstore(self.state,(row+4+slot*2)*64,alpha,p)

    @jit
    def accumulate(self,pv,slot):
        with vf(mode='simd'):
            p=create_mask(pattern='all',elem_bits=32)
            for row in range(2):
                alpha=vload(self.state,(row+4+slot*2)*64)
                for j in range(4):
                    a=vload(self.acc,row*256+j*64)
                    b=vload(pv,row*256+j*64)
                    vstore(self.acc,row*256+j*64,vadd(vmul(a,alpha,mask=p),b,mask=p),p)

    @jit
    def finish(self,partgm,lsegm):
        out=self.out.produce()
        lse=self.lse.produce()
        with vf(mode='simd'):
            p=create_mask(pattern='all',elem_bits=32)
            for row in range(2):
                total=vload(self.state,(row+2)*64)
                maximum=vload(self.state,row*64)
                lv=vadd(vlog(total,mask=p),maximum,mask=p)
                vstore_first(lse,row*8,lv)
                for j in range(4):
                    z=vdiv(vload(self.acc,row*256+j*64),total,mask=p)
                    vstore(out,row*256+j*64,z,p)
        mem_copy(partgm,self.out.consume())
        mem_copy(lsegm,self.lse.consume())

    @jit
    def execute_serial(self,qgm,kcache,vcache,nk,nv,table,pos,partials,lses):
        length=load_bypass(pos.ptr(0))+1
        lower=0
        begin=0
        if not self.full:
            lower=max(0,length-513)
            begin=(lower//128)*128
        parts=(length-begin+self.width-1)//self.width
        for task in range(get_block_idx(),parts*4,get_block_num()):
            part=task//4
            head=task%4
            start=begin+part*self.width
            valid=min(self.width,length-start)
            self.initialize()
            qslot=self.q.produce()
            mem_copy(qslot,tile_slice(qgm,(4,256),(head,0)),engine=self.nd2nz)
            for step in range((valid+self.n-1)//self.n):
                first=start+step*self.n
                count=min(self.n,valid-step*self.n)
                ks=self.k.produce()
                for page in range((count+127)//128):
                    physical=load_bypass(table.ptr(first//128+page))
                    if const_expr(self.full):
                        source=nk[physical,head,None,None,None].view(16,2048)
                        target=reinterpret(ks,shape=(16,self.n*16),data_format="nd")
                        mem_copy(tile_slice(target,(16,2048),(0,page)),source)
                    else:
                        mem_copy(tile_slice(ks,(128,256),(page,0)),tile_slice(kcache.view(-1,1024),(128,256),(physical,head)),engine=self.nd2nz)
                kl=self.k.consume()
                cc=self.c.produce()
                for sub in range((count+127)//128):
                    aa=reinterpret(self.a,shape=(4,256),data_format="nz")
                    bb=reinterpret(self.b,shape=(128,256),data_format="nz")
                    mem_copy(aa,qslot)
                    mem_copy(bb,tile_slice(kl,(128,256),(sub,0)))
                    matmul(tile_slice(cc,(4,128),(0,sub)),aa,bb,init=True)
                score_slot=self.scores.produce()
                mem_copy(score_slot,self.c.consume(),engine=self.split)
                # Issue V after all QK reads, overlapping its MTE2 work with softmax.
                vs=self.v.produce()
                for page in range((count+127)//128):
                    physical=load_bypass(table.ptr(first//128+page))
                    if const_expr(self.full):
                        source=nv[physical,head,None,None,None].view(16,2048)
                        target=reinterpret(vs,shape=(16,self.n*16),data_format="nd")
                        mem_copy(tile_slice(target,(16,2048),(0,page)),source)
                    else:
                        mem_copy(tile_slice(vs,(128,256),(page,0)),tile_slice(vcache.view(-1,1024),(128,256),(physical,head)),engine=self.nd2nz)
                vl=self.v.consume()
                probs=self.pu.produce()
                score=self.scores.consume()
                if const_expr(self.debug):
                    h=head*4+get_subblock_id()*2
                    mem_copy(tile_slice(self.debugqk,(1,2,self.n),(part,h//2,0)).view(2,self.n),score)
                self.softmax(score,probs,first,count,lower,0)
                if const_expr(self.debug):
                    mem_copy(tile_slice(self.debugexp,(1,2,self.n),(part,h//2,0)).view(2,self.n),score)

                prob_l1=self.p.produce()
                prob_ub=self.pu.consume()
                for row in range(2):
                    target=reinterpret(prob_l1,shape=(self.n//16,16),stride=(256,1),offset=(get_subblock_id()*2+row)*32,data_format="nd")
                    mem_copy(target,prob_ub[row,None].view(self.n//16,16))
                pl=reinterpret(self.p.consume(),shape=(4,self.n),data_format="nz")
                cc=self.cv.produce()
                for sub in range((count+127)//128):
                    aa=reinterpret(self.a,shape=(4,128),data_format="nz")
                    bb=reinterpret(self.b,shape=(128,256),data_format="nz")
                    mem_copy(aa,tile_slice(pl,(4,128),(0,sub)))
                    mem_copy(bb,tile_slice(vl,(128,256),(sub,0)),transpose=True)
                    matmul(cc,aa,bb,init=sub==0)
                mem_copy(self.pv.produce(),cc,engine=self.split)
                pv=self.pv.consume()
                if const_expr(self.debug):
                    mem_copy(tile_slice(self.debugpv,(1,2,256),(part,h//2,0)).view(2,256),pv)
                self.accumulate(pv,0)
            h=head*4+get_subblock_id()*2
            self.finish(tile_slice(partials,(1,2,256),(part,h//2,0)).view(2,256),tile_slice(lses,(1,2,8),(part,h//2,0)).view(2,8))

    @jit
    def prefetch_phase(self,nk,nv,table,phase,tiles,start,valid,head):
        is_q=(phase==0) or ((phase%2==1) and (phase<2*tiles-1))
        tile=0
        if is_q:
            tile=(phase+1)//2
        else:
            tile=phase//2-1
            if phase==2*tiles-1:
                tile=tiles-1
        first=start+tile*self.n
        count=min(self.n,valid-tile*self.n)
        slot=self.kv.produce()
        target=reinterpret(slot,shape=(16,self.n*16),data_format="nd")
        for page in range((count+127)//128):
            physical=load_bypass(table.ptr(first//128+page))
            if is_q:
                mem_copy(tile_slice(target,(16,2048),(0,page)),nk[physical,head,None,None,None].view(16,2048))
            else:
                mem_copy(tile_slice(target,(16,2048),(0,page)),nv[physical,head,None,None,None].view(16,2048))

    @jit
    def execute_pipeline(self,qgm,kcache,vcache,nk,nv,table,pos,partials,lses):
        length=load_bypass(pos.ptr(0))+1
        parts=(length+self.width-1)//self.width
        for task in range(get_block_idx(),parts*4,get_block_num()):
            part=task//4
            head=task%4
            start=part*self.width
            valid=min(self.width,length-start)
            tiles=(valid+self.n-1)//self.n
            self.initialize()
            qslot=self.q.produce()
            mem_copy(qslot,tile_slice(qgm,(4,256),(head,0)),engine=self.nd2nz)
            h=head*4+get_subblock_id()*2
            # Native phase order: Q0,Q1,V0,Q2,V1,...,Qlast,Vprev,Vlast.
            # Stage next phase's paged K/V before consuming the current L1 slot.
            self.prefetch_phase(nk,nv,table,0,tiles,start,valid,head)
            for phase in range(2*tiles):
                if phase+1<2*tiles:
                    self.prefetch_phase(nk,nv,table,phase+1,tiles,start,valid,head)
                is_q=(phase==0) or ((phase%2==1) and (phase<2*tiles-1))
                tile=0
                if is_q:
                    tile=(phase+1)//2
                else:
                    tile=phase//2-1
                    if phase==2*tiles-1:
                        tile=tiles-1
                first=start+tile*self.n
                count=min(self.n,valid-tile*self.n)
                kv=self.kv.consume()
                if is_q:
                    cc=self.c.produce()
                    for sub in range((count+127)//128):
                        aa=reinterpret(self.a,shape=(4,256),data_format="nz")
                        bb=reinterpret(self.b,shape=(128,256),data_format="nz")
                        mem_copy(aa,qslot)
                        mem_copy(bb,tile_slice(kv,(128,256),(sub,0)))
                        matmul(tile_slice(cc,(4,128),(0,sub)),aa,bb,init=True)
                    mem_copy(self.scores.produce(),self.c.consume(),engine=self.split)
                    probs=self.pu.produce()
                    score=self.scores.consume()
                    if const_expr(self.debug):
                        mem_copy(tile_slice(self.debugqk,(1,2,self.n),(part,h//2,0)).view(2,self.n),score)
                    self.softmax(score,probs,first,count,0,tile%2)
                    if const_expr(self.debug):
                        mem_copy(tile_slice(self.debugexp,(1,2,self.n),(part,h//2,0)).view(2,self.n),score)
                    prob_l1=self.p.produce()
                    prob_ub=self.pu.consume()
                    for row in range(2):
                        target=reinterpret(prob_l1,shape=(self.n//16,16),stride=(256,1),offset=(get_subblock_id()*2+row)*32,data_format="nd")
                        mem_copy(target,prob_ub[row,None].view(self.n//16,16))
                else:
                    pl=reinterpret(self.p.consume(),shape=(4,self.n),data_format="nz")
                    cc=self.cv.produce()
                    for sub in range((count+127)//128):
                        aa=reinterpret(self.a,shape=(4,128),data_format="nz")
                        bb=reinterpret(self.b,shape=(128,256),data_format="nz")
                        mem_copy(aa,tile_slice(pl,(4,128),(0,sub)))
                        mem_copy(bb,tile_slice(kv,(128,256),(sub,0)),transpose=True)
                        matmul(cc,aa,bb,init=sub==0)
                    mem_copy(self.pv.produce(),cc,engine=self.split)
                    pv=self.pv.consume()
                    if const_expr(self.debug):
                        mem_copy(tile_slice(self.debugpv,(1,2,256),(part,h//2,0)).view(2,256),pv)
                    self.accumulate(pv,tile%2)
            self.finish(tile_slice(partials,(1,2,256),(part,h//2,0)).view(2,256),tile_slice(lses,(1,2,8),(part,h//2,0)).view(2,8))

    @jit
    def execute(self,qgm,kcache,vcache,nk,nv,table,pos,partials,lses):
        if const_expr(self.full):
            self.execute_pipeline(qgm,kcache,vcache,nk,nv,table,pos,partials,lses)
        else:
            self.execute_serial(qgm,kcache,vcache,nk,nv,table,pos,partials,lses)

