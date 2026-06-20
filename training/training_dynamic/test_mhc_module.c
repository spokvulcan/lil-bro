// FD check of the mhc.h split API (issue #11): premap → external sub-layer F →
// recombine, and the matching backward recombine_bwd → F_bwd → premap_bwd. F is a
// per-position linear map (stand-in for the ANE sub-layer). Verifies grads of
// loss=<dout,Xout> w.r.t. X and every map param + F's weight, with multi-position
// non-square dims (DIM≠SEQ≠N_HC) to catch [N_HC,DIM,SEQ] layout bugs.
#include <stdio.h>
#define N_HC 4
#define DIM  5
#define SEQ  3
#include "mhc.h"

static MhcMap P; static MHC_F Wf[DIM*DIM];
static MHC_F X0[N_HC*DIM*SEQ], dout[N_HC*DIM*SEQ];

// forward: returns loss, fills Xout
static MHC_F fwd(MHC_F *Xout){
    static MhcTape T; static MHC_F u[DIM*SEQ], F[DIM*SEQ];
    mhc_premap(X0,&P,&T,u);
    for(int d=0;d<DIM;d++)for(int t=0;t<SEQ;t++){ MHC_F s=0;
        for(int q=0;q<DIM;q++) s+=Wf[d*DIM+q]*u[q*SEQ+t]; F[d*SEQ+t]=s; }
    mhc_recombine(X0,F,&T,Xout);
    MHC_F l=0; for(int i=0;i<N_HC*DIM*SEQ;i++) l+=dout[i]*Xout[i]; return l;
}
static MHC_F loss(){ static MHC_F Xo[N_HC*DIM*SEQ]; return fwd(Xo); }

static unsigned sd=7;
static MHC_F rnd(){ sd=sd*1103515245u+12345u; return (MHC_F)((sd>>9))/(MHC_F)8388608.0-(MHC_F)1.0; }

int main(void){
    for(int i=0;i<N_HC*DIM*SEQ;i++){ X0[i]=0.5f*rnd(); dout[i]=rnd(); }
    P=mhc_map_alloc(); unsigned s2=3; mhc_map_init(&P,&s2);
    P.a_pre=0.4; P.a_res=0.5; P.a_post=0.3;             // exercise nonzero gates
    for(int i=0;i<DIM*DIM;i++) Wf[i]=(MHC_F)0.3*rnd();
    // analytic
    static MhcTape T; static MHC_F u[DIM*SEQ], F[DIM*SEQ], Xo[N_HC*DIM*SEQ];
    mhc_premap(X0,&P,&T,u);
    for(int d=0;d<DIM;d++)for(int t=0;t<SEQ;t++){ MHC_F ss=0;
        for(int q=0;q<DIM;q++) ss+=Wf[d*DIM+q]*u[q*SEQ+t]; F[d*SEQ+t]=ss; }
    mhc_recombine(X0,F,&T,Xo);
    static MHC_F dX[N_HC*DIM*SEQ], dF[DIM*SEQ], dB[SEQ*N_HC*N_HC], dC[SEQ*N_HC];
    mhc_recombine_bwd(X0,F,&T,dout,dX,dF,dB,dC);
    MHC_F du[DIM*SEQ]={0}, dWf[DIM*DIM]={0};
    for(int d=0;d<DIM;d++)for(int t=0;t<SEQ;t++){ MHC_F gf=dF[d*SEQ+t];
        for(int q=0;q<DIM;q++){ du[q*SEQ+t]+=gf*Wf[d*DIM+q]; dWf[d*DIM+q]+=gf*u[q*SEQ+t]; } }
    MhcGrad G=mhc_grad_alloc();
    mhc_premap_bwd(X0,&P,&T,du,dB,dC,dX,&G);
    // FD
    MHC_F worst=0; const char*wn=""; MHC_F eps=(MHC_F)(sizeof(MHC_F)>4?1e-6:2e-4);
    #define CK(arr,grad,n,nm) do{ for(int i=0;i<n;i++){ MHC_F sv=(arr)[i]; (arr)[i]=sv+eps; MHC_F lp=loss(); \
        (arr)[i]=sv-eps; MHC_F lm=loss(); (arr)[i]=sv; MHC_F num=(lp-lm)/(2*eps); MHC_F e=fabs(num-(grad)[i]); \
        if(e>worst){worst=e;wn=nm;} }}while(0)
    CK(X0,dX,N_HC*DIM*SEQ,"X");
    CK(P.Wpre,G.Wpre,MHC_M*N_HC,"Wpre"); CK(P.Wres,G.Wres,MHC_M*N_HC*N_HC,"Wres");
    CK(P.Wpost,G.Wpost,MHC_M*N_HC,"Wpost");
    CK(P.Spre,G.Spre,N_HC,"Spre"); CK(P.Sres,G.Sres,N_HC*N_HC,"Sres"); CK(P.Spost,G.Spost,N_HC,"Spost");
    CK((&P.a_pre),(&G.a_pre),1,"a_pre"); CK((&P.a_res),(&G.a_res),1,"a_res"); CK((&P.a_post),(&G.a_post),1,"a_post");
    CK(Wf,dWf,DIM*DIM,"Wf");
    MHC_F ds=mhc_ds_residual(&T);
    MHC_F tol=(MHC_F)(sizeof(MHC_F)>4?1e-7:5e-3);   // double: tight (Sinkhorn FD floor ~1e-9)
    printf("mhc.h module FD (%s): max|analytic-numerical| = %.2e (worst:%s)  doubly-stoch residual = %.2e  %s\n",
           sizeof(MHC_F)>4?"double":"float", (double)worst, wn, (double)ds, (worst<tol && ds<(MHC_F)1e-5)?"OK":"FAIL");
    return (worst<tol && ds<(MHC_F)1e-5)?0:1;
}
