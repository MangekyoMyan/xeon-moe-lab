    if constexpr (with_margin && has_bias) {
        float next=selection_wt[0];int eid=threadIdx.x;
#pragma unroll
        for(int i=1;i<experts_per_thread;i++){
            int e=threadIdx.x+i*WARP_SIZE;
            if(selection_wt[i]>next){next=selection_wt[i];eid=e;}
        }
#pragma unroll
        for(int mask=WARP_SIZE/2;mask>0;mask/=2){
            float v=__shfl_xor_sync(0xffffffff,next,mask,WARP_SIZE);
            int e=__shfl_xor_sync(0xffffffff,eid,mask,WARP_SIZE);
            if(v>next||(v==next&&e<eid)){next=v;eid=e;}
        }
        if(threadIdx.x==0){
            float gap=last_score-next;
            config.margin[row]={last_id,eid,gap,gap<=config.cut};
        }
    }
