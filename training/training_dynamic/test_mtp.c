// End-to-end FD check of the CPU-first MTP path (issue #6): a full transformer
// block (identical to the main block) + the MTP glue (rms_h/rms_e, concat, proj,
// per-depth head/CE) + cross-depth gradient chaining. Verifies the combined MTP
// loss's gradient w.r.t. every MTP parameter AND the shared trunk/embed, against
// central differences. These functions become mtp_* in cpu_ops.h.
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <Accelerate/Accelerate.h>

#define DIM 8
#define HD 4
#define HEADS 2
#define KV_HEADS 1
#define HIDDEN 16
#define SEQ 6
#define VOCAB 10
#define MTP_DEPTH 2
#define Q_DIM (HEADS*HD)
#define KV_DIM (KV_HEADS*HD)
#define NORM_EPS 1e-5f
#define MTP_LAMBDA 0.3f
static const float RES_ALPHA = 0.5f;

// ---- primitives (channel-major [C,S], weights row-major [O,I]) ----
static void mm(float*o,const float*W,const float*x,int O,int IN,int S){ // o=W@x
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,O,S,IN,1.0f,W,IN,x,S,0.0f,o,S); }
static void mmWT(float*o,const float*W,const float*dy,int O,int IN,int S){ // o=W^T@dy
    cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,IN,S,O,1.0f,W,IN,dy,S,0.0f,o,S); }
static void dW_acc(float*dW,const float*dy,const float*x,int O,int IN,int S){ // dW += dy@x^T
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,O,IN,S,1.0f,dy,S,x,S,1.0f,dW,IN); }

static void rms_fwd(float*o,const float*x,const float*w,int C,int S){
    for(int t=0;t<S;t++){ float ss=0; for(int i=0;i<C;i++){float v=x[i*S+t]; ss+=v*v;}
        float r=1.0f/sqrtf(ss/C+NORM_EPS); for(int i=0;i<C;i++) o[i*S+t]=x[i*S+t]*r*w[i]; } }
// dx (+= into provided), dw (+=). returns nothing.
static void rms_bwd(float*dx,float*dw,const float*dy,const float*x,const float*w,int C,int S){
    for(int t=0;t<S;t++){ float ss=0; for(int i=0;i<C;i++){float v=x[i*S+t]; ss+=v*v;}
        float ms=ss/C+NORM_EPS, r=1.0f/sqrtf(ms);
        float c=0; for(int i=0;i<C;i++) c+=w[i]*dy[i*S+t]*x[i*S+t];
        for(int i=0;i<C;i++){ float xd=x[i*S+t], dyy=dy[i*S+t];
            dx[i*S+t]+= w[i]*dyy*r - xd*c*r*r*r/C; dw[i]+= dyy*xd*r; } } }

static void rope_fwd(float*x,int S){ int nh=Q_DIM/HD; // generic over a [*,S] with HD heads
    (void)nh; }
// interleaved RoPE on [C,S], C = nheads*HD. forward rotation.
static void rope_apply(float*x,int C,int S,int inv){
    int nh=C/HD;
    for(int h=0;h<nh;h++) for(int i=0;i<HD/2;i++){
        float freq=1.0f/powf(10000.0f,2.0f*i/(float)HD);
        for(int p=0;p<S;p++){ float th=p*freq, c=cosf(th), s=sinf(th);
            int a=(h*HD+2*i)*S+p, b=(h*HD+2*i+1)*S+p; float v0=x[a],v1=x[b];
            if(!inv){ x[a]=v0*c-v1*s; x[b]=v0*s+v1*c; }
            else    { x[a]=v0*c+v1*s; x[b]=-v0*s+v1*c; } } } }

// attention core (no sink/qknorm), causal, GQA. da/dQ etc layout [*,S].
static void attn_fwd(float*ao,const float*Q,const float*K,const float*V,int S){
    float scale=1.0f/sqrtf((float)HD);
    for(int h=0;h<HEADS;h++){int kvh=h%KV_HEADS;
        for(int q=0;q<S;q++){ float sc[SEQ],m=-1e30f;
            for(int j=0;j<=q;j++){float d=0;for(int e=0;e<HD;e++)d+=Q[(h*HD+e)*S+q]*K[(kvh*HD+e)*S+j];sc[j]=d*scale;if(sc[j]>m)m=sc[j];}
            float Z=0;for(int j=0;j<=q;j++){sc[j]=expf(sc[j]-m);Z+=sc[j];} float inv=1.0f/Z;
            for(int e=0;e<HD;e++){float acc=0;for(int j=0;j<=q;j++)acc+=sc[j]*V[(kvh*HD+e)*S+j];ao[(h*HD+e)*S+q]=acc*inv;} } } }
static void attn_bwd(const float*da,const float*Q,const float*K,const float*V,float*dQ,float*dK,float*dV,int S){
    float scale=1.0f/sqrtf((float)HD);
    memset(dQ,0,Q_DIM*S*4);memset(dK,0,KV_DIM*S*4);memset(dV,0,KV_DIM*S*4);
    for(int h=0;h<HEADS;h++){int kvh=h%KV_HEADS;
        for(int q=0;q<S;q++){ float sc[SEQ],m=-1e30f;
            for(int j=0;j<=q;j++){float d=0;for(int e=0;e<HD;e++)d+=Q[(h*HD+e)*S+q]*K[(kvh*HD+e)*S+j];sc[j]=d*scale;if(sc[j]>m)m=sc[j];}
            float Z=0;for(int j=0;j<=q;j++){sc[j]=expf(sc[j]-m);Z+=sc[j];} float inv=1.0f/Z;
            float p[SEQ];for(int j=0;j<=q;j++)p[j]=sc[j]*inv;
            float dp[SEQ];for(int j=0;j<=q;j++){float acc=0;for(int e=0;e<HD;e++){float dad=da[(h*HD+e)*S+q];acc+=dad*V[(kvh*HD+e)*S+j];dV[(kvh*HD+e)*S+j]+=p[j]*dad;}dp[j]=acc;}
            float g=0;for(int j=0;j<=q;j++)g+=p[j]*dp[j];
            for(int j=0;j<=q;j++){float dsc=p[j]*(dp[j]-g)*scale;for(int e=0;e<HD;e++){dQ[(h*HD+e)*S+q]+=dsc*K[(kvh*HD+e)*S+j];dK[(kvh*HD+e)*S+j]+=dsc*Q[(h*HD+e)*S+q];}} } } }

// ---- one transformer block (matches mlx_ref _block) ----
typedef struct { float Wq[Q_DIM*DIM],Wk[KV_DIM*DIM],Wv[KV_DIM*DIM],Wo[DIM*Q_DIM];
    float W1[HIDDEN*DIM],W2[DIM*HIDDEN],W3[HIDDEN*DIM],rms_att[DIM],rms_ffn[DIM]; } Block;
typedef struct { float xin[DIM*SEQ],h1[DIM*SEQ],Q[Q_DIM*SEQ],K[KV_DIM*SEQ],V[KV_DIM*SEQ],
    attn[Q_DIM*SEQ],o[DIM*SEQ],x2[DIM*SEQ],h2[DIM*SEQ],g1[HIDDEN*SEQ],g3[HIDDEN*SEQ],
    silu[HIDDEN*SEQ],gate[HIDDEN*SEQ],ff[DIM*SEQ]; int S; } BlockAct;

static void block_fwd(float*out,const float*x,const Block*b,BlockAct*a,int S){
    a->S=S; memcpy(a->xin,x,DIM*S*4);
    rms_fwd(a->h1,x,b->rms_att,DIM,S);
    mm(a->Q,b->Wq,a->h1,Q_DIM,DIM,S); mm(a->K,b->Wk,a->h1,KV_DIM,DIM,S); mm(a->V,b->Wv,a->h1,KV_DIM,DIM,S);
    rope_apply(a->Q,Q_DIM,S,0); rope_apply(a->K,KV_DIM,S,0);
    attn_fwd(a->attn,a->Q,a->K,a->V,S);
    mm(a->o,b->Wo,a->attn,DIM,Q_DIM,S);
    for(int i=0;i<DIM*S;i++) a->x2[i]=x[i]+RES_ALPHA*a->o[i];
    rms_fwd(a->h2,a->x2,b->rms_ffn,DIM,S);
    mm(a->g1,b->W1,a->h2,HIDDEN,DIM,S); mm(a->g3,b->W3,a->h2,HIDDEN,DIM,S);
    for(int i=0;i<HIDDEN*S;i++){ float s=1.0f/(1.0f+expf(-a->g1[i])); a->silu[i]=a->g1[i]*s; a->gate[i]=a->silu[i]*a->g3[i]; }
    mm(a->ff,b->W2,a->gate,DIM,HIDDEN,S);
    for(int i=0;i<DIM*S;i++) out[i]=a->x2[i]+RES_ALPHA*a->ff[i];
}
// dout -> dx (+= into dx), accumulate weight grads into gb.
static void block_bwd(float*dx,const float*dout,const Block*b,const BlockAct*a,Block*gb,int S){
    float dx2[DIM*SEQ],dff[DIM*SEQ],dgate[HIDDEN*SEQ],dsilu[HIDDEN*SEQ],dg1[HIDDEN*SEQ],dg3[HIDDEN*SEQ];
    float dh2[DIM*SEQ],dattn[Q_DIM*SEQ],dQ[Q_DIM*SEQ],dK[KV_DIM*SEQ],dV[KV_DIM*SEQ],dh1[DIM*SEQ],dxa[DIM*SEQ];
    // ffn residual: out = x2 + ra*ff
    for(int i=0;i<DIM*S;i++){ dx2[i]=dout[i]; dff[i]=RES_ALPHA*dout[i]; }
    // ff = W2@gate
    dW_acc(gb->W2,dff,a->gate,DIM,HIDDEN,S); mmWT(dgate,b->W2,dff,DIM,HIDDEN,S);
    // gate = silu*g3 ; silu = g1*sigmoid(g1)
    for(int i=0;i<HIDDEN*S;i++){ float s=1.0f/(1.0f+expf(-a->g1[i]));
        dsilu[i]=dgate[i]*a->g3[i]; dg3[i]=dgate[i]*a->silu[i];
        float dsilu_dg1=s*(1.0f+a->g1[i]*(1.0f-s)); dg1[i]=dsilu[i]*dsilu_dg1; }
    // g1=W1@h2, g3=W3@h2
    dW_acc(gb->W1,dg1,a->h2,HIDDEN,DIM,S); dW_acc(gb->W3,dg3,a->h2,HIDDEN,DIM,S);
    mmWT(dh2,b->W1,dg1,HIDDEN,DIM,S); { float t[DIM*SEQ]; mmWT(t,b->W3,dg3,HIDDEN,DIM,S); for(int i=0;i<DIM*S;i++) dh2[i]+=t[i]; }
    // h2 = rmsnorm(x2,rms_ffn)
    memset(dxa,0,DIM*S*4); rms_bwd(dxa,gb->rms_ffn,dh2,a->x2,b->rms_ffn,DIM,S);
    for(int i=0;i<DIM*S;i++) dx2[i]+=dxa[i];
    // x2 = xin + ra*o
    float dxin[DIM*SEQ]; for(int i=0;i<DIM*S;i++){ dxin[i]=dx2[i]; }
    float do_[DIM*SEQ]; for(int i=0;i<DIM*S;i++) do_[i]=RES_ALPHA*dx2[i];
    // o = Wo@attn
    dW_acc(gb->Wo,do_,a->attn,DIM,Q_DIM,S); mmWT(dattn,b->Wo,do_,DIM,Q_DIM,S);
    // attention
    attn_bwd(dattn,a->Q,a->K,a->V,dQ,dK,dV,S);
    // rope backward (inverse-transpose) on dQ,dK
    rope_apply(dQ,Q_DIM,S,1); rope_apply(dK,KV_DIM,S,1);
    // Q=Wq@h1 etc.
    dW_acc(gb->Wq,dQ,a->h1,Q_DIM,DIM,S); dW_acc(gb->Wk,dK,a->h1,KV_DIM,DIM,S); dW_acc(gb->Wv,dV,a->h1,KV_DIM,DIM,S);
    mmWT(dh1,b->Wq,dQ,Q_DIM,DIM,S);
    { float t[DIM*SEQ]; mmWT(t,b->Wk,dK,KV_DIM,DIM,S); for(int i=0;i<DIM*S;i++) dh1[i]+=t[i];
      mmWT(t,b->Wv,dV,KV_DIM,DIM,S); for(int i=0;i<DIM*S;i++) dh1[i]+=t[i]; }
    // h1 = rmsnorm(xin,rms_att)
    memset(dxa,0,DIM*S*4); rms_bwd(dxa,gb->rms_att,dh1,a->xin,b->rms_att,DIM,S);
    for(int i=0;i<DIM*S;i++) dx[i]+= dxin[i]+dxa[i];
}

// ---- MTP params ----
typedef struct { float rms_h[DIM],rms_e[DIM],proj[DIM*2*DIM]; Block blk; } MtpDepth;

// head: logits[VOCAB,S] = embed @ rmsnorm(h,rms_final); CE vs tgt over VOCAB.
// returns loss; fills dh (grad into h), accumulates dembed (via head) and drms_final.
static float head_ce(const float*h,const float*rms_final,const float*embed,const int*tgt,int S,
                     float*dh,float*dembed,float*drms_final,float wscale){
    float hn[DIM*SEQ]; rms_fwd(hn,h,rms_final,DIM,S);
    float logits[VOCAB*SEQ]; mm(logits,embed,hn,VOCAB,DIM,S);
    float dlogits[VOCAB*SEQ]; float loss=0;
    for(int t=0;t<S;t++){ float m=-1e30f; for(int v=0;v<VOCAB;v++){float l=logits[v*S+t];if(l>m)m=l;}
        float Z=0; for(int v=0;v<VOCAB;v++) Z+=expf(logits[v*S+t]-m); float lse=m+logf(Z);
        loss += lse - logits[tgt[t]*S+t];
        for(int v=0;v<VOCAB;v++){ float p=expf(logits[v*S+t]-m)/Z; dlogits[v*S+t]=(p-(v==tgt[t]?1.0f:0.0f))*wscale/S; } }
    // logits=embed@hn : dembed += dlogits@hn^T ; dhn = embed^T@dlogits
    dW_acc(dembed,dlogits,hn,VOCAB,DIM,S);
    float dhn[DIM*SEQ]; mmWT(dhn,embed,dlogits,VOCAB,DIM,S);
    rms_bwd(dh,drms_final,dhn,h,rms_final,DIM,S);
    return loss/S;
}

// Full MTP forward+backward. trunk[DIM,SEQ], targets[SEQ] (token ids, the main
// loss targets). Accumulates grads. Returns the MTP term (lambda*mean(losses)).
static float mtp_run(const float*trunk,const int*targets,const float*embed,const float*rms_final,
                     MtpDepth*mtp, float*dtrunk,float*dembed,float*drms_final,
                     MtpDepth*gmtp, int do_bwd){
    // forward, saving per-depth activations
    static float hk[MTP_DEPTH][DIM*SEQ], hkin[MTP_DEPTH][DIM*SEQ], hp[MTP_DEPTH][DIM*SEQ];
    static float ev[MTP_DEPTH][DIM*SEQ], nh[MTP_DEPTH][DIM*SEQ], ne[MTP_DEPTH][DIM*SEQ];
    static float normed[MTP_DEPTH][2*DIM*SEQ]; static BlockAct acts[MTP_DEPTH];
    int Sk_arr[MTP_DEPTH]; const float*h_prev=trunk; int prevS=SEQ;
    float losses[MTP_DEPTH]; int ndep=0;
    for(int kk=1;kk<=MTP_DEPTH;kk++){ int Sk=SEQ-kk; if(Sk<=0) break; int d=kk-1; Sk_arr[d]=Sk; ndep++;
        MtpDepth*M=&mtp[d];
        for(int i=0;i<DIM;i++) for(int t=0;t<Sk;t++) hp[d][i*Sk+t]=h_prev[i*prevS+t];
        for(int i=0;i<DIM;i++) for(int t=0;t<Sk;t++) ev[d][i*Sk+t]=embed[targets[kk-1+t]*DIM+i];
        rms_fwd(nh[d],hp[d],M->rms_h,DIM,Sk); rms_fwd(ne[d],ev[d],M->rms_e,DIM,Sk);
        for(int i=0;i<DIM;i++) for(int t=0;t<Sk;t++) normed[d][i*Sk+t]=nh[d][i*Sk+t];
        for(int i=0;i<DIM;i++) for(int t=0;t<Sk;t++) normed[d][(DIM+i)*Sk+t]=ne[d][i*Sk+t];
        mm(hkin[d],M->proj,normed[d],DIM,2*DIM,Sk);
        block_fwd(hk[d],hkin[d],&M->blk,&acts[d],Sk);
        // loss target tgt_k = targets[kk .. kk+Sk]
        int tgt[SEQ]; for(int t=0;t<Sk;t++) tgt[t]=targets[kk+t];
        float dum_dh[DIM*SEQ],dum_de[VOCAB*DIM]={0},dum_dr[DIM]={0};
        (void)dum_dh;
        // forward-only loss (recompute in bwd); use a scratch head for the value
        float hn[DIM*SEQ]; rms_fwd(hn,hk[d],rms_final,DIM,Sk);
        float logits[VOCAB*SEQ]; mm(logits,embed,hn,VOCAB,DIM,Sk); float L=0;
        for(int t=0;t<Sk;t++){ float m=-1e30f;for(int v=0;v<VOCAB;v++){float l=logits[v*Sk+t];if(l>m)m=l;}
            float Z=0;for(int v=0;v<VOCAB;v++)Z+=expf(logits[v*Sk+t]-m); L+= m+logf(Z)-logits[tgt[t]*Sk+t]; }
        losses[d]=L/Sk; (void)dum_de;(void)dum_dr;
        h_prev=hk[d]; prevS=Sk;
    }
    float mtp_term=0; for(int d=0;d<ndep;d++) mtp_term+=losses[d]; mtp_term=MTP_LAMBDA*mtp_term/ndep;
    if(!do_bwd) return mtp_term;
    // backward
    float dh_next[DIM*SEQ]; int dh_next_S=0;
    for(int d=ndep-1; d>=0; d--){ int kk=d+1; int Sk=Sk_arr[d]; MtpDepth*M=&mtp[d]; MtpDepth*G=&gmtp[d];
        int tgt[SEQ]; for(int t=0;t<Sk;t++) tgt[t]=targets[kk+t];
        float dhk[DIM*SEQ]; memset(dhk,0,DIM*Sk*4);
        head_ce(hk[d],rms_final,embed,tgt,Sk,dhk,dembed,drms_final, MTP_LAMBDA/ndep);
        if(d<ndep-1){ for(int i=0;i<DIM;i++) for(int t=0;t<dh_next_S;t++) dhk[i*Sk+t]+=dh_next[i*dh_next_S+t]; }
        float dhkin[DIM*SEQ]; memset(dhkin,0,DIM*Sk*4);
        block_bwd(dhkin,dhk,&M->blk,&acts[d],&G->blk,Sk);
        // hkin = proj@normed
        dW_acc(G->proj,dhkin,normed[d],DIM,2*DIM,Sk);
        float dnormed[2*DIM*SEQ]; mmWT(dnormed,M->proj,dhkin,DIM,2*DIM,Sk);
        // split -> dnh,dne
        float dnh[DIM*SEQ],dne[DIM*SEQ];
        for(int i=0;i<DIM;i++) for(int t=0;t<Sk;t++){ dnh[i*Sk+t]=dnormed[i*Sk+t]; dne[i*Sk+t]=dnormed[(DIM+i)*Sk+t]; }
        // rms_h: dnh -> dhp(+=) ; rms_e: dne -> de -> dembed scatter
        float dhp[DIM*SEQ]; memset(dhp,0,DIM*Sk*4); rms_bwd(dhp,G->rms_h,dnh,hp[d],M->rms_h,DIM,Sk);
        float de[DIM*SEQ]; memset(de,0,DIM*Sk*4); rms_bwd(de,G->rms_e,dne,ev[d],M->rms_e,DIM,Sk);
        for(int i=0;i<DIM;i++) for(int t=0;t<Sk;t++) dembed[targets[kk-1+t]*DIM+i]+=de[i*Sk+t];
        // dhp flows into h_prev (hk[d-1] or trunk)
        if(d==0){ for(int i=0;i<DIM;i++) for(int t=0;t<Sk;t++) dtrunk[i*SEQ+t]+=dhp[i*Sk+t]; }
        else { memcpy(dh_next,dhp,DIM*Sk*4); dh_next_S=Sk; }
    }
    return mtp_term;
}

// ---- FD harness ----
static float frand(){ return 2.0f*((float)rand()/(float)RAND_MAX)-1.0f; }
static float trunk[DIM*SEQ], embed[VOCAB*DIM], rms_final[DIM]; static int targets[SEQ];
static MtpDepth mtp[MTP_DEPTH];
static float lossval(){ float dt[DIM*SEQ]={0},de[VOCAB*DIM]={0},dr[DIM]={0}; MtpDepth g[MTP_DEPTH]; memset(g,0,sizeof g);
    return mtp_run(trunk,targets,embed,rms_final,mtp,dt,de,dr,g,0); }

int main(void){ srand(5);
    for(int i=0;i<DIM*SEQ;i++) trunk[i]=0.3f*frand();
    for(int i=0;i<VOCAB*DIM;i++) embed[i]=0.3f*frand();
    for(int i=0;i<DIM;i++) rms_final[i]=1.0f+0.1f*frand();
    for(int t=0;t<SEQ;t++) targets[t]=rand()%VOCAB;
    for(int d=0;d<MTP_DEPTH;d++){ MtpDepth*M=&mtp[d];
        for(int i=0;i<DIM;i++){M->rms_h[i]=1.0f+0.1f*frand();M->rms_e[i]=1.0f+0.1f*frand();}
        for(int i=0;i<DIM*2*DIM;i++)M->proj[i]=0.2f*frand();
        float*w[]={M->blk.Wq,M->blk.Wk,M->blk.Wv,M->blk.Wo,M->blk.W1,M->blk.W2,M->blk.W3};
        int n[]={Q_DIM*DIM,KV_DIM*DIM,KV_DIM*DIM,DIM*Q_DIM,HIDDEN*DIM,DIM*HIDDEN,HIDDEN*DIM};
        for(int k=0;k<7;k++) for(int i=0;i<n[k];i++) w[k][i]=0.2f*frand();
        for(int i=0;i<DIM;i++){M->blk.rms_att[i]=1.0f+0.1f*frand();M->blk.rms_ffn[i]=1.0f+0.1f*frand();} }
    // analytic grads
    float dt[DIM*SEQ]={0},de[VOCAB*DIM]={0},dr[DIM]={0}; MtpDepth g[MTP_DEPTH]; memset(g,0,sizeof g);
    mtp_run(trunk,targets,embed,rms_final,mtp,dt,de,dr,g,1);
    float eps=1e-3f,worst=0; const char*wname="";
    #define CK(arr,grad,n,nm) do{ for(int i=0;i<n;i++){ float sv=arr[i]; arr[i]=sv+eps; float lp=lossval(); arr[i]=sv-eps; float lm=lossval(); arr[i]=sv; \
        float num=(lp-lm)/(2*eps); float e=fabsf(num-(grad)[i]); if(e>worst){worst=e;wname=nm;} }}while(0)
    CK(trunk,dt,DIM*SEQ,"trunk"); CK(embed,de,VOCAB*DIM,"embed"); CK(rms_final,dr,DIM,"rms_final");
    for(int d=0;d<MTP_DEPTH;d++){ MtpDepth*M=&mtp[d]; MtpDepth*G=&g[d];
        CK(M->rms_h,G->rms_h,DIM,"rms_h"); CK(M->rms_e,G->rms_e,DIM,"rms_e"); CK(M->proj,G->proj,DIM*2*DIM,"proj");
        CK(M->blk.Wq,G->blk.Wq,Q_DIM*DIM,"Wq"); CK(M->blk.Wk,G->blk.Wk,KV_DIM*DIM,"Wk"); CK(M->blk.Wv,G->blk.Wv,KV_DIM*DIM,"Wv");
        CK(M->blk.Wo,G->blk.Wo,DIM*Q_DIM,"Wo"); CK(M->blk.W1,G->blk.W1,HIDDEN*DIM,"W1"); CK(M->blk.W2,G->blk.W2,DIM*HIDDEN,"W2");
        CK(M->blk.W3,G->blk.W3,HIDDEN*DIM,"W3"); CK(M->blk.rms_att,G->blk.rms_att,DIM,"rms_att"); CK(M->blk.rms_ffn,G->blk.rms_ffn,DIM,"rms_ffn"); }
    printf("MTP end-to-end FD: max|analytic-numerical| = %.2e (worst:%s)  %s\n",worst,wname,worst<5e-3?"OK":"FAIL");
    return worst<5e-3?0:1;
}
