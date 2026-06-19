// Finite-difference check of the attention-sink CPU fwd+bwd (issue #8), incl GQA.
// These are the exact functions that go into cpu_ops.h (attn_sink_forward/backward).
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

// Tiny GQA config for the test: 4 q-heads, 2 kv-heads (GQA_RATIO=2), hd=8, seq=5.
#define HEADS 4
#define KV_HEADS 2
#define HD 8
#define SEQ 5
#define Q_DIM (HEADS*HD)
#define KV_DIM (KV_HEADS*HD)

static void attn_sink_forward(float *attn_out, const float *Q, const float *K,
                              const float *V, const float *sink_h, int seq) {
    float scale = 1.0f/sqrtf((float)HD);
    for (int h=0; h<HEADS; h++) {
        int kvh = h % KV_HEADS;
        float s = sink_h[h];
        for (int q=0; q<seq; q++) {
            float sc[SEQ]; float m = s;
            for (int j=0; j<=q; j++) {
                float dot=0;
                for (int d=0; d<HD; d++) dot += Q[(h*HD+d)*seq+q]*K[(kvh*HD+d)*seq+j];
                sc[j]=dot*scale; if (sc[j]>m) m=sc[j];
            }
            float Z=0; for (int j=0;j<=q;j++){ sc[j]=expf(sc[j]-m); Z+=sc[j]; }
            float inv = 1.0f/(Z + expf(s-m));
            for (int d=0; d<HD; d++) {
                float acc=0;
                for (int j=0;j<=q;j++) acc += sc[j]*V[(kvh*HD+d)*seq+j];
                attn_out[(h*HD+d)*seq+q] = acc*inv;
            }
        }
    }
}

static void attn_sink_backward(const float *da, const float *Q, const float *K,
                               const float *V, const float *sink_h,
                               float *dQ, float *dK, float *dV, float *dsink, int seq) {
    float scale = 1.0f/sqrtf((float)HD);
    memset(dQ, 0, Q_DIM*seq*4); memset(dK, 0, KV_DIM*seq*4); memset(dV, 0, KV_DIM*seq*4);
    for (int h=0;h<HEADS;h++) {
        int kvh = h % KV_HEADS; float s = sink_h[h]; float dsh = 0;
        for (int q=0;q<seq;q++) {
            float sc[SEQ]; float m=s;
            for (int j=0;j<=q;j++){
                float dot=0;
                for (int d=0;d<HD;d++) dot += Q[(h*HD+d)*seq+q]*K[(kvh*HD+d)*seq+j];
                sc[j]=dot*scale; if(sc[j]>m)m=sc[j];
            }
            float Z=0; for(int j=0;j<=q;j++){ sc[j]=expf(sc[j]-m); Z+=sc[j]; }
            float esink=expf(s-m), inv=1.0f/(Z+esink);
            float p[SEQ]; for(int j=0;j<=q;j++) p[j]=sc[j]*inv;
            float psink = esink*inv;
            float dp[SEQ];
            for(int j=0;j<=q;j++){
                float acc=0;
                for(int d=0;d<HD;d++){
                    float dad = da[(h*HD+d)*seq+q];
                    acc += dad * V[(kvh*HD+d)*seq+j];
                    dV[(kvh*HD+d)*seq+j] += p[j]*dad;
                }
                dp[j]=acc;
            }
            float g=0; for(int j=0;j<=q;j++) g += p[j]*dp[j];
            dsh += -psink*g;
            for(int j=0;j<=q;j++){
                float dscore = p[j]*(dp[j]-g)*scale;
                for(int d=0;d<HD;d++){
                    dQ[(h*HD+d)*seq+q] += dscore*K[(kvh*HD+d)*seq+j];
                    dK[(kvh*HD+d)*seq+j] += dscore*Q[(h*HD+d)*seq+q];
                }
            }
        }
        dsink[h] += dsh;
    }
}

static float fr(void){ return 2.0f*((float)rand()/RAND_MAX)-1.0f; }
static float loss_of(const float *Q,const float *K,const float *V,const float *sink,const float *da){
    float ao[Q_DIM*SEQ]; attn_sink_forward(ao,Q,K,V,sink,SEQ);
    float l=0; for(int i=0;i<Q_DIM*SEQ;i++) l+=da[i]*ao[i]; return l;
}

int main(void){
    srand(3);
    float Q[Q_DIM*SEQ],K[KV_DIM*SEQ],V[KV_DIM*SEQ],sink[HEADS],da[Q_DIM*SEQ];
    for(int i=0;i<Q_DIM*SEQ;i++){Q[i]=fr();da[i]=fr();}
    for(int i=0;i<KV_DIM*SEQ;i++){K[i]=fr();V[i]=fr();}
    for(int h=0;h<HEADS;h++) sink[h]=0.4f*fr();
    float dQ[Q_DIM*SEQ],dK[KV_DIM*SEQ],dV[KV_DIM*SEQ],dsink[HEADS]={0};
    attn_sink_backward(da,Q,K,V,sink,dQ,dK,dV,dsink,SEQ);
    float eps=1e-3f, worst=0;
    #define CHECK(arr,n) do{ for(int i=0;i<n;i++){ float sv=arr[i]; \
        arr[i]=sv+eps; float lp=loss_of(Q,K,V,sink,da); arr[i]=sv-eps; float lm=loss_of(Q,K,V,sink,da); arr[i]=sv; \
        float num=(lp-lm)/(2*eps); float an=d##arr[i]; float e=fabsf(num-an); if(e>worst)worst=e; }}while(0)
    CHECK(Q,Q_DIM*SEQ); CHECK(K,KV_DIM*SEQ); CHECK(V,KV_DIM*SEQ); CHECK(sink,HEADS);
    printf("attn-sink fwd+bwd (GQA %d/%d, hd=%d, seq=%d): max |analytic-numerical| = %.2e  %s\n",
           HEADS,KV_HEADS,HD,SEQ,worst, worst<5e-3?"OK":"FAIL");
    return worst<5e-3?0:1;
}
