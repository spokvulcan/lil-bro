// probe_rms.m — RMSNorm-bwd-on-ANE: correctness vs CPU + per-eval cost.
// Walls cleared: channel-axis reduce_sum/pow EVAL-OK; the only constraint was the
// input spatial must be a MULTIPLE OF 32 (SP sweep), so SP rounds 2*SEQ+1 -> 544.
// Now the real question: is a STANDALONE rms-bwd eval a win or a wash on this
// dispatch-bound ANE? LOCAL eval.
#include "mil_dynamic.h"
#include "cpu_ops.h"

static BOOL try_eval(Kern *k) {
    id mdl=(__bridge id)k->model; id req=(__bridge id)k->request; NSError *e=nil;
    return ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        mdl,@selector(evaluateWithQoS:options:request:error:),21,@{},req,&e);
}
static double time_kern(Kern *k,int it){ for(int i=0;i<20;i++) if(!try_eval(k)) return -1;
    uint64_t t0=mach_absolute_time(); for(int i=0;i<it;i++) try_eval(k); return tb_ms(mach_absolute_time()-t0)/it; }

int main(void){
    @autoreleasepool{
        ane_init(); mach_timebase_info(&g_tb);
        const int SP=((2*SEQ+1+31)/32)*32, N=400;

        srand48(7);
        float *dy=(float*)malloc((size_t)DIM*SEQ*4),*x=(float*)malloc((size_t)DIM*SEQ*4),*w=(float*)malloc((size_t)DIM*4);
        for(int i=0;i<DIM*SEQ;i++){dy[i]=(float)(drand48()*2-1);x[i]=(float)(drand48()*2-1);}
        for(int i=0;i<DIM;i++) w[i]=(float)(drand48()*0.5+0.75);
        float *dx_cpu=(float*)calloc((size_t)DIM*SEQ,4),*dw=(float*)calloc((size_t)DIM,4);
        rmsnorm_bwd(dx_cpu,dw,dy,x,w,DIM,SEQ);

        Kern *k=compile_kern_mil_w(gen_rmsnorm_bwd_dynamic(),@{},DIM*SP*2,DIM*SEQ*2);
        if(!k){printf("compile FAIL\n");return 1;}
        float *host=(float*)calloc((size_t)DIM*SP,4);
        for(int c=0;c<DIM;c++){ for(int s=0;s<SEQ;s++){host[c*SP+s]=dy[c*SEQ+s];host[c*SP+SEQ+s]=x[c*SEQ+s];} host[c*SP+2*SEQ]=w[c]; }
        io_write_fp16(k->ioIn,host,DIM,SP);
        if(!try_eval(k)){printf("EVAL REJECTED (0x1d) SP=%d\n",SP);return 2;}

        float *dx_ane=(float*)malloc((size_t)DIM*SEQ*4);
        io_read_fp16(k->ioOut,dx_ane,0,DIM,SEQ);
        double dot=0,na=0,nc=0,mx=0;
        for(int i=0;i<DIM*SEQ;i++){dot+=(double)dx_ane[i]*dx_cpu[i];na+=(double)dx_ane[i]*dx_ane[i];nc+=(double)dx_cpu[i]*dx_cpu[i];
            double d=fabs((double)dx_ane[i]-dx_cpu[i]); if(d>mx)mx=d;}
        printf("\n[rms-bwd ANE] SP=%d  correctness vs CPU: cos=%.5f  max_abs_err=%.4e  ||dx_cpu||=%.3f\n",
               SP, dot/(sqrt(na)*sqrt(nc)+1e-30), mx, sqrt(nc));
        double ms=time_kern(k,N); int nev=2*NLAYERS+1;
        printf("[rms-bwd ANE] per-eval=%.4f ms; %d evals/step => %.2f ms ANE (CPU rms_bwd bucket ~4.1 ms; dispatch floor ~0.12)\n",
               ms,nev,nev*ms);
        free(dy);free(x);free(w);free(dx_cpu);free(dw);free(host);free(dx_ane);free_kern(k);
    }
    return 0;
}
