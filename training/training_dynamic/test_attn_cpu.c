// FD check of the UNIFIED CPU attention core (issues #8 sink + #7 QK-norm), GQA-aware.
// These functions become attn_cpu_forward/backward in cpu_ops.h. QK-norm is RMSNorm
// over head_dim applied to Q and K (post-RoPE, just before scores) with learnable
// per-dim gains gq[HD], gk[HD]; sink is the per-head softmax-denominator logit.
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#define HEADS 4
#define KV_HEADS 2
#define HD 8
#define SEQ 5
#define Q_DIM (HEADS*HD)
#define KV_DIM (KV_HEADS*HD)
#define NORM_EPS 1e-5f

static void attn_cpu_forward(float *attn_out, const float *Q, const float *K, const float *V,
                             const float *sink_h, const float *gq, const float *gk, int seq) {
    float scale = 1.0f/sqrtf((float)HD);
    for (int h=0; h<HEADS; h++) {
        int kvh = h % KV_HEADS;
        int have_sink = (sink_h != NULL);
        float s = have_sink ? sink_h[h] : 0.0f;
        for (int q=0; q<seq; q++) {
            float qn[HD];
            if (gq) { float ms=0; for(int d=0;d<HD;d++){float v=Q[(h*HD+d)*seq+q]; ms+=v*v;}
                      float r=sqrtf(ms/HD+NORM_EPS); for(int d=0;d<HD;d++) qn[d]=Q[(h*HD+d)*seq+q]/r*gq[d]; }
            else    { for(int d=0;d<HD;d++) qn[d]=Q[(h*HD+d)*seq+q]; }
            float sc[SEQ]; float m = have_sink ? s : -1e30f;
            for (int j=0;j<=q;j++) {
                float dot=0;
                if (gk) { float ms=0; for(int d=0;d<HD;d++){float v=K[(kvh*HD+d)*seq+j]; ms+=v*v;}
                          float r=sqrtf(ms/HD+NORM_EPS); for(int d=0;d<HD;d++) dot+=qn[d]*(K[(kvh*HD+d)*seq+j]/r*gk[d]); }
                else    { for(int d=0;d<HD;d++) dot+=qn[d]*K[(kvh*HD+d)*seq+j]; }
                sc[j]=dot*scale; if(sc[j]>m)m=sc[j];
            }
            float Z=0; for(int j=0;j<=q;j++){ sc[j]=expf(sc[j]-m); Z+=sc[j]; }
            float inv=1.0f/(Z + (have_sink?expf(s-m):0.0f));
            for(int d=0;d<HD;d++){ float acc=0; for(int j=0;j<=q;j++) acc+=sc[j]*V[(kvh*HD+d)*seq+j];
                                   attn_out[(h*HD+d)*seq+q]=acc*inv; }
        }
    }
}

static void attn_cpu_backward(const float *da, const float *Q, const float *K, const float *V,
                              const float *sink_h, const float *gq, const float *gk,
                              float *dQ, float *dK, float *dV, float *dsink,
                              float *dgq, float *dgk, int seq) {
    float scale=1.0f/sqrtf((float)HD);
    memset(dQ,0,Q_DIM*seq*4); memset(dK,0,KV_DIM*seq*4); memset(dV,0,KV_DIM*seq*4);
    float *Qn=malloc(Q_DIM*seq*4), *Kn=malloc(KV_DIM*seq*4);
    float *rq=malloc(HEADS*seq*4), *rk=malloc(KV_HEADS*seq*4);
    float *dQn=calloc(Q_DIM*seq,4), *dKn=calloc(KV_DIM*seq,4);
    // Precompute normalized Q,K (or alias) + the per-vector rms.
    for (int h=0;h<HEADS;h++) for(int q=0;q<seq;q++){
        if (gq){ float ms=0; for(int d=0;d<HD;d++){float v=Q[(h*HD+d)*seq+q]; ms+=v*v;}
                 float r=sqrtf(ms/HD+NORM_EPS); rq[h*seq+q]=r;
                 for(int d=0;d<HD;d++) Qn[(h*HD+d)*seq+q]=Q[(h*HD+d)*seq+q]/r*gq[d]; }
        else { for(int d=0;d<HD;d++) Qn[(h*HD+d)*seq+q]=Q[(h*HD+d)*seq+q]; }
    }
    for (int kvh=0;kvh<KV_HEADS;kvh++) for(int j=0;j<seq;j++){
        if (gk){ float ms=0; for(int d=0;d<HD;d++){float v=K[(kvh*HD+d)*seq+j]; ms+=v*v;}
                 float r=sqrtf(ms/HD+NORM_EPS); rk[kvh*seq+j]=r;
                 for(int d=0;d<HD;d++) Kn[(kvh*HD+d)*seq+j]=K[(kvh*HD+d)*seq+j]/r*gk[d]; }
        else { for(int d=0;d<HD;d++) Kn[(kvh*HD+d)*seq+j]=K[(kvh*HD+d)*seq+j]; }
    }
    // Main attention backward, in normalized space.
    for (int h=0;h<HEADS;h++){
        int kvh=h%KV_HEADS; int have_sink=(sink_h!=NULL); float s=have_sink?sink_h[h]:0.0f; float dsh=0;
        for (int q=0;q<seq;q++){
            float sc[SEQ]; float m=have_sink?s:-1e30f;
            for(int j=0;j<=q;j++){ float dot=0; for(int d=0;d<HD;d++) dot+=Qn[(h*HD+d)*seq+q]*Kn[(kvh*HD+d)*seq+j];
                                   sc[j]=dot*scale; if(sc[j]>m)m=sc[j]; }
            float Z=0; for(int j=0;j<=q;j++){ sc[j]=expf(sc[j]-m); Z+=sc[j]; }
            float esink=have_sink?expf(s-m):0.0f, inv=1.0f/(Z+esink);
            float p[SEQ]; for(int j=0;j<=q;j++) p[j]=sc[j]*inv;
            float psink=esink*inv;
            float dp[SEQ];
            for(int j=0;j<=q;j++){ float acc=0; for(int d=0;d<HD;d++){ float dad=da[(h*HD+d)*seq+q];
                acc+=dad*V[(kvh*HD+d)*seq+j]; dV[(kvh*HD+d)*seq+j]+=p[j]*dad; } dp[j]=acc; }
            float g=0; for(int j=0;j<=q;j++) g+=p[j]*dp[j];
            if(have_sink) dsh += -psink*g;
            for(int j=0;j<=q;j++){ float dscore=p[j]*(dp[j]-g)*scale;
                for(int d=0;d<HD;d++){ dQn[(h*HD+d)*seq+q]+=dscore*Kn[(kvh*HD+d)*seq+j];
                                       dKn[(kvh*HD+d)*seq+j]+=dscore*Qn[(h*HD+d)*seq+q]; } }
        }
        if(have_sink) dsink[h]+=dsh;
    }
    // RMSNorm VJP: dQn -> dQ (+dgq), dKn -> dK (+dgk). dL/dx_i = g_i dy_i/r - x_i c/(HD r^3),
    // c = sum_d g_d dy_d x_d ; dL/dg_d += dy_d x_d / r.
    if (gq){ for(int h=0;h<HEADS;h++) for(int q=0;q<seq;q++){ float r=rq[h*seq+q]; float c=0;
        for(int d=0;d<HD;d++) c+=gq[d]*dQn[(h*HD+d)*seq+q]*Q[(h*HD+d)*seq+q];
        for(int d=0;d<HD;d++){ float xd=Q[(h*HD+d)*seq+q], dy=dQn[(h*HD+d)*seq+q];
            dQ[(h*HD+d)*seq+q]= gq[d]*dy/r - xd*c/(HD*r*r*r); dgq[d]+= dy*xd/r; } } }
    else memcpy(dQ,dQn,Q_DIM*seq*4);
    if (gk){ for(int kvh=0;kvh<KV_HEADS;kvh++) for(int j=0;j<seq;j++){ float r=rk[kvh*seq+j]; float c=0;
        for(int d=0;d<HD;d++) c+=gk[d]*dKn[(kvh*HD+d)*seq+j]*K[(kvh*HD+d)*seq+j];
        for(int d=0;d<HD;d++){ float xd=K[(kvh*HD+d)*seq+j], dy=dKn[(kvh*HD+d)*seq+j];
            dK[(kvh*HD+d)*seq+j]= gk[d]*dy/r - xd*c/(HD*r*r*r); dgk[d]+= dy*xd/r; } } }
    else memcpy(dK,dKn,KV_DIM*seq*4);
    free(Qn);free(Kn);free(rq);free(rk);free(dQn);free(dKn);
}

static float fr(void){ return 2.0f*((float)rand()/(float)RAND_MAX)-1.0f; }
static float loss_of(const float *Q,const float *K,const float *V,const float *sink,
                     const float *gq,const float *gk,const float *da){
    float ao[Q_DIM*SEQ]; attn_cpu_forward(ao,Q,K,V,sink,gq,gk,SEQ);
    float l=0; for(int i=0;i<Q_DIM*SEQ;i++) l+=da[i]*ao[i]; return l;
}

static void run(const char *name, int use_sink, int use_norm){
    srand(7);
    float Q[Q_DIM*SEQ],K[KV_DIM*SEQ],V[KV_DIM*SEQ],sinkb[HEADS],da[Q_DIM*SEQ],gqb[HD],gkb[HD];
    for(int i=0;i<Q_DIM*SEQ;i++){Q[i]=fr();da[i]=fr();}
    for(int i=0;i<KV_DIM*SEQ;i++){K[i]=fr();V[i]=fr();}
    for(int h=0;h<HEADS;h++) sinkb[h]=0.4f*fr();
    for(int d=0;d<HD;d++){ gqb[d]=1.0f+0.3f*fr(); gkb[d]=1.0f+0.3f*fr(); }
    const float *sink = use_sink?sinkb:NULL;
    const float *gq = use_norm?gqb:NULL;
    const float *gk = use_norm?gkb:NULL;
    float dQ[Q_DIM*SEQ],dK[KV_DIM*SEQ],dV[KV_DIM*SEQ],dsink[HEADS]={0},dgq[HD]={0},dgk[HD]={0};
    attn_cpu_backward(da,Q,K,V,sink,gq,gk,dQ,dK,dV,dsink,dgq,dgk,SEQ);
    float eps=1e-3f, worst=0;
    #define CHK(arr,n,grad) do{ for(int i=0;i<n;i++){ float sv=arr[i]; \
        arr[i]=sv+eps; float lp=loss_of(Q,K,V,sink,gq,gk,da); arr[i]=sv-eps; float lm=loss_of(Q,K,V,sink,gq,gk,da); arr[i]=sv; \
        float num=(lp-lm)/(2*eps); float e=fabsf(num-grad[i]); if(e>worst)worst=e; }}while(0)
    CHK(Q,Q_DIM*SEQ,dQ); CHK(K,KV_DIM*SEQ,dK); CHK(V,KV_DIM*SEQ,dV);
    if(use_sink) CHK(sinkb,HEADS,dsink);
    if(use_norm){ CHK(gqb,HD,dgq); CHK(gkb,HD,dgk); }
    printf("%-22s max|analytic-numerical| = %.2e  %s\n", name, worst, worst<5e-3?"OK":"FAIL");
}

int main(void){
    run("qk_norm only:", 0, 1);
    run("sink only:",    1, 0);
    run("sink + qk_norm:",1, 1);
    run("neither (identity):",0,0);
    return 0;
}
