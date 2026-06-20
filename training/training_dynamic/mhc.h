// mhc.h — manifold-constrained hyper-connections (issue #11), CPU-first per ADR 0001.
//
// Wraps one transformer sub-layer F (attention or FFN) in the V4 mHC residual
// coupling (PDF §2.2 eqs 1-7). The residual stream is expanded to N_HC parallel
// streams X ∈ ℝ^{N_HC×DIM} per position; three maps generated *per position* from
// X̂ = RMSNorm(vec X) reshape the streams around F:
//
//     u      = A·X                       (collapse N_HC streams → one DIM-vector)
//     X_out  = B·X + C ⊗ F(u)            (B mixes streams, C broadcasts F's output)
//
//   A = σ(α_pre ·(X̂ W^pre) + S^pre)            ∈ (0,1)^{N_HC}     input map
//   B = Sinkhorn(α_res·Mat(X̂ W^res)+S^res)     doubly-stochastic   residual map
//   C = 2σ(α_post·(X̂ W^post)+S^post)           ∈ (0,2)^{N_HC}     output map
//
// The Sinkhorn is the #5 spike's verified log-domain code (τ≥0.5, t_max=20). The
// whole module is CPU; F stays on the ANE. Forward/backward is FD-verified:
// test_mhc_module.c (this header's split API) → ~1e-12 in double, doubly-stoch ~1e-7.
//
// Layout: X is [N_HC, DIM, SEQ] channel-major within each stream, i.e.
//   X[i*DIM*SEQ + d*SEQ + t]   = stream i, channel d, position t.
//   u/Fout are [DIM, SEQ]:  u[d*SEQ + t].
// Requires N_HC, DIM, SEQ defined before include (config.h in the trainer; the
// unit test defines them directly so this header pulls in no ObjC/Accelerate).
#pragma once
#include <math.h>
#include <string.h>
#include <stdlib.h>

// Scalar type: float in the trainer (fp32 CPU, matches attn_cpu/MTP); the unit
// test overrides to double (-DMHC_F=double) to FD-verify transcription tightly.
#ifndef MHC_F
#define MHC_F float
#endif

#define MHC_M    (N_HC*DIM)
#define MHC_TMAX 20
#ifndef MHC_TAU
#define MHC_TAU  ((MHC_F)0.5)
#endif
#ifndef MHC_EPS
#define MHC_EPS  ((MHC_F)1e-6)
#endif

// One sub-layer's map parameters (flat buffers, heap-allocated).
typedef struct {
    MHC_F *Wpre;   // [MHC_M * N_HC]
    MHC_F *Wres;   // [MHC_M * N_HC*N_HC]
    MHC_F *Wpost;  // [MHC_M * N_HC]
    MHC_F *Spre;   // [N_HC]
    MHC_F *Sres;   // [N_HC*N_HC]
    MHC_F *Spost;  // [N_HC]
    MHC_F a_pre, a_res, a_post;   // scalar gates
} MhcMap;

typedef struct {     // gradients, same shapes; scalars accumulate
    MHC_F *Wpre, *Wres, *Wpost, *Spre, *Sres, *Spost;
    MHC_F a_pre, a_res, a_post;
} MhcGrad;

// Per-position saved state for the backward pass.
typedef struct {
    MHC_F Xhat[MHC_M];
    MHC_F rms;
    MHC_F A[N_HC], C[N_HC];
    MHC_F B[N_HC*N_HC];                       // doubly-stochastic result
    MHC_F P[MHC_TMAX*N_HC*N_HC];              // Sinkhorn row-normalize tape
    MHC_F Q[MHC_TMAX*N_HC*N_HC];              // Sinkhorn col-normalize tape
    MHC_F L[N_HC*N_HC];                        // log-domain L = Btilde/tau
    MHC_F f[N_HC], g[N_HC];                    // final dual potentials
} MhcPos;

typedef struct { MhcPos pos[SEQ]; } MhcTape;

static inline MHC_F mhc_sigm(MHC_F x){ return (MHC_F)1.0/((MHC_F)1.0+exp(-x)); }
static inline MHC_F mhc_lse(const MHC_F *x,int n){ MHC_F m=x[0];
    for(int i=1;i<n;i++) if(x[i]>m)m=x[i];
    MHC_F s=0; for(int i=0;i<n;i++) s+=exp(x[i]-m); return m+log(s); }

// ---- Sinkhorn fwd/bwd (verified spike, generalized to N=N_HC) ----
static void mhc_sk_fwd(const MHC_F *Btil, MhcPos *P){
    const int N=N_HC;
    for(int i=0;i<N*N;i++) P->L[i]=Btil[i]/MHC_TAU;
    for(int i=0;i<N;i++){ P->f[i]=0; P->g[i]=0; }
    for(int t=0;t<MHC_TMAX;t++){
        for(int i=0;i<N;i++){ MHC_F row[N_HC];
            for(int j=0;j<N;j++) row[j]=P->L[i*N+j]+P->g[j];
            MHC_F l=mhc_lse(row,N); P->f[i]=-l;
            for(int j=0;j<N;j++) P->P[t*N*N+i*N+j]=exp(row[j]-l); }
        for(int j=0;j<N;j++){ MHC_F col[N_HC];
            for(int i=0;i<N;i++) col[i]=P->L[i*N+j]+P->f[i];
            MHC_F l=mhc_lse(col,N); P->g[j]=-l;
            for(int i=0;i<N;i++) P->Q[t*N*N+i*N+j]=exp(col[i]-l); } }
    for(int i=0;i<N;i++)for(int j=0;j<N;j++)
        P->B[i*N+j]=exp(P->L[i*N+j]+P->f[i]+P->g[j]);
}
// dB (N*N) -> dBtil (N*N), reverse over the unrolled iterations.
static void mhc_sk_bwd(const MhcPos *P, const MHC_F *dB, MHC_F *dBtil){
    const int N=N_HC;
    MHC_F dL[N_HC*N_HC]={0}, df[N_HC]={0}, dg[N_HC]={0};
    for(int i=0;i<N;i++)for(int j=0;j<N;j++){ MHC_F t=dB[i*N+j]*P->B[i*N+j];
        dL[i*N+j]+=t; df[i]+=t; dg[j]+=t; }
    for(int t=MHC_TMAX-1;t>=0;t--){
        MHC_F dfn[N_HC]={0};
        for(int j=0;j<N;j++)for(int i=0;i<N;i++){ MHC_F c=-dg[j]*P->Q[t*N*N+i*N+j];
            dL[i*N+j]+=c; dfn[i]+=c; }
        for(int i=0;i<N;i++)df[i]+=dfn[i];
        for(int j=0;j<N;j++)dg[j]=0;
        MHC_F dgn[N_HC]={0};
        for(int i=0;i<N;i++)for(int j=0;j<N;j++){ MHC_F c=-df[i]*P->P[t*N*N+i*N+j];
            dL[i*N+j]+=c; dgn[j]+=c; }
        for(int j=0;j<N;j++)dg[j]=dgn[j];
        for(int i=0;i<N;i++)df[i]=0; }
    for(int i=0;i<N*N;i++) dBtil[i]=dL[i]/MHC_TAU;
}

// ---- alloc / init ----
static MhcMap mhc_map_alloc(void){
    MhcMap m; size_t z=sizeof(MHC_F);
    m.Wpre =(MHC_F*)calloc(MHC_M*N_HC,z);
    m.Wres =(MHC_F*)calloc(MHC_M*N_HC*N_HC,z);
    m.Wpost=(MHC_F*)calloc(MHC_M*N_HC,z);
    m.Spre =(MHC_F*)calloc(N_HC,z);
    m.Sres =(MHC_F*)calloc(N_HC*N_HC,z);
    m.Spost=(MHC_F*)calloc(N_HC,z);
    m.a_pre=m.a_res=m.a_post=0.0;
    return m;
}
static MhcGrad mhc_grad_alloc(void){
    MhcGrad g; size_t z=sizeof(MHC_F);
    g.Wpre =(MHC_F*)calloc(MHC_M*N_HC,z);
    g.Wres =(MHC_F*)calloc(MHC_M*N_HC*N_HC,z);
    g.Wpost=(MHC_F*)calloc(MHC_M*N_HC,z);
    g.Spre =(MHC_F*)calloc(N_HC,z);
    g.Sres =(MHC_F*)calloc(N_HC*N_HC,z);
    g.Spost=(MHC_F*)calloc(N_HC,z);
    g.a_pre=g.a_res=g.a_post=0.0;
    return g;
}
static void mhc_grad_zero(MhcGrad *g){ size_t z=sizeof(MHC_F);
    memset(g->Wpre,0,MHC_M*N_HC*z); memset(g->Wres,0,MHC_M*N_HC*N_HC*z);
    memset(g->Wpost,0,MHC_M*N_HC*z); memset(g->Spre,0,N_HC*z);
    memset(g->Sres,0,N_HC*N_HC*z); memset(g->Spost,0,N_HC*z);
    g->a_pre=g->a_res=g->a_post=0.0;
}
// Init near "plain residual": maps start mild, scalar gates small. At S*=0:
// A=σ(0)=.5, C=2σ(0)=1, B=Sinkhorn(0)=1/N (uniform). Streams start symmetric.
static void mhc_map_init(MhcMap *m, unsigned *seed){
    #define MHC_RND ( (MHC_F)((*seed=*seed*1103515245u+12345u)>>9) / (MHC_F)8388608.0 - (MHC_F)1.0 )
    for(int i=0;i<MHC_M*N_HC;i++){ m->Wpre[i]=(MHC_F)0.02*MHC_RND; m->Wpost[i]=(MHC_F)0.02*MHC_RND; }
    for(int i=0;i<MHC_M*N_HC*N_HC;i++) m->Wres[i]=(MHC_F)0.02*MHC_RND;
    for(int i=0;i<N_HC;i++){ m->Spre[i]=0.0; m->Spost[i]=0.0; }
    for(int i=0;i<N_HC*N_HC;i++) m->Sres[i]=0.0;
    m->a_pre=0.1; m->a_res=0.1; m->a_post=0.1;
    #undef MHC_RND
}

// ---- forward ----
// u[DIM,SEQ] = A·X, maps saved per position into T.
static void mhc_premap(const MHC_F *X, const MhcMap *p, MhcTape *T, MHC_F *u){
    const int N=N_HC;
    for(int t=0;t<SEQ;t++){
        MhcPos *sv=&T->pos[t];
        MHC_F v[MHC_M];
        for(int m=0;m<MHC_M;m++) v[m]=X[(m/DIM)*DIM*SEQ + (m%DIM)*SEQ + t];
        MHC_F ss=0; for(int m=0;m<MHC_M;m++) ss+=v[m]*v[m];
        sv->rms=sqrt(ss/MHC_M+MHC_EPS);
        for(int m=0;m<MHC_M;m++) sv->Xhat[m]=v[m]/sv->rms;
        // A
        for(int i=0;i<N;i++){ MHC_F s=0; for(int m=0;m<MHC_M;m++) s+=sv->Xhat[m]*p->Wpre[m*N+i];
            sv->A[i]=mhc_sigm(p->a_pre*s + p->Spre[i]); }
        // B
        MHC_F Btil[N_HC*N_HC];
        for(int i=0;i<N;i++)for(int j=0;j<N;j++){ MHC_F s=0;
            for(int m=0;m<MHC_M;m++) s+=sv->Xhat[m]*p->Wres[m*N*N+i*N+j];
            Btil[i*N+j]=p->a_res*s + p->Sres[i*N+j]; }
        mhc_sk_fwd(Btil, sv);
        // C
        for(int i=0;i<N;i++){ MHC_F s=0; for(int m=0;m<MHC_M;m++) s+=sv->Xhat[m]*p->Wpost[m*N+i];
            sv->C[i]=(MHC_F)2.0*mhc_sigm(p->a_post*s + p->Spost[i]); }
        // u = A·X
        for(int d=0;d<DIM;d++){ MHC_F s=0;
            for(int i=0;i<N;i++) s+=sv->A[i]*X[i*DIM*SEQ + d*SEQ + t];
            u[d*SEQ+t]=s; }
    }
}
// Xout[N_HC,DIM,SEQ] = B·X + C⊗Fout.   Fout is [DIM,SEQ]. Xout may not alias X.
static void mhc_recombine(const MHC_F *X, const MHC_F *Fout, const MhcTape *T, MHC_F *Xout){
    const int N=N_HC;
    for(int t=0;t<SEQ;t++){
        const MhcPos *sv=&T->pos[t];
        for(int i=0;i<N;i++)for(int d=0;d<DIM;d++){
            MHC_F s=0; for(int j=0;j<N;j++) s+=sv->B[i*N+j]*X[j*DIM*SEQ + d*SEQ + t];
            Xout[i*DIM*SEQ + d*SEQ + t] = s + sv->C[i]*Fout[d*SEQ+t];
        }
    }
}

// ---- backward ----
// dXout -> {dX (B·X part, ZEROES then accumulates), dFout, dB_all[SEQ*N*N], dC_all[SEQ*N]}.
static void mhc_recombine_bwd(const MHC_F *X, const MHC_F *Fout, const MhcTape *T,
                              const MHC_F *dXout, MHC_F *dX, MHC_F *dFout,
                              MHC_F *dB_all, MHC_F *dC_all){
    const int N=N_HC; size_t z=sizeof(MHC_F);
    memset(dX,0,(size_t)N_HC*DIM*SEQ*z);
    memset(dFout,0,(size_t)DIM*SEQ*z);
    memset(dB_all,0,(size_t)SEQ*N*N*z);
    memset(dC_all,0,(size_t)SEQ*N*z);
    for(int t=0;t<SEQ;t++){
        const MhcPos *sv=&T->pos[t];
        MHC_F *dB=dB_all+t*N*N, *dC=dC_all+t*N;
        for(int i=0;i<N;i++)for(int d=0;d<DIM;d++){
            MHC_F go=dXout[i*DIM*SEQ + d*SEQ + t];
            for(int j=0;j<N;j++){ dB[i*N+j]+=go*X[j*DIM*SEQ + d*SEQ + t];
                dX[j*DIM*SEQ + d*SEQ + t]+=go*sv->B[i*N+j]; }
            dC[i]+=go*Fout[d*SEQ+t];
            dFout[d*SEQ+t]+=go*sv->C[i];
        }
    }
}
// du (grad of collapsed input) + dB_all/dC_all -> accumulate map grads + dX (+=).
static void mhc_premap_bwd(const MHC_F *X, const MhcMap *p, const MhcTape *T,
                           const MHC_F *du, const MHC_F *dB_all, const MHC_F *dC_all,
                           MHC_F *dX, MhcGrad *g){
    const int N=N_HC;
    for(int t=0;t<SEQ;t++){
        const MhcPos *sv=&T->pos[t];
        const MHC_F *dB=dB_all+t*N*N, *dC=dC_all+t*N;
        MHC_F dA[N_HC]={0}, dXhat[MHC_M]={0};
        // u = A·X
        for(int d=0;d<DIM;d++){ MHC_F gu=du[d*SEQ+t];
            for(int i=0;i<N;i++){ dA[i]+=gu*X[i*DIM*SEQ + d*SEQ + t];
                dX[i*DIM*SEQ + d*SEQ + t]+=gu*sv->A[i]; } }
        // C = 2σ(Cpre); dCpre = dC * 2σ' = dC*2*σ*(1-σ), σ=C/2
        for(int i=0;i<N;i++){ MHC_F sg=sv->C[i]*(MHC_F)0.5;
            MHC_F dcpre=dC[i]*(MHC_F)2.0*sg*((MHC_F)1.0-sg);
            g->Spost[i]+=dcpre;
            MHC_F s=0; for(int m=0;m<MHC_M;m++) s+=sv->Xhat[m]*p->Wpost[m*N+i];
            g->a_post+=dcpre*s;
            for(int m=0;m<MHC_M;m++){ g->Wpost[m*N+i]+=dcpre*p->a_post*sv->Xhat[m];
                dXhat[m]+=dcpre*p->a_post*p->Wpost[m*N+i]; } }
        // A = σ(Apre); dApre = dA*σ*(1-σ)
        for(int i=0;i<N;i++){ MHC_F dapre=dA[i]*sv->A[i]*((MHC_F)1.0-sv->A[i]);
            g->Spre[i]+=dapre;
            MHC_F s=0; for(int m=0;m<MHC_M;m++) s+=sv->Xhat[m]*p->Wpre[m*N+i];
            g->a_pre+=dapre*s;
            for(int m=0;m<MHC_M;m++){ g->Wpre[m*N+i]+=dapre*p->a_pre*sv->Xhat[m];
                dXhat[m]+=dapre*p->a_pre*p->Wpre[m*N+i]; } }
        // B = Sinkhorn(Btil); Btil = a_res*(Xhat·Wres)+Sres
        MHC_F dBtil[N_HC*N_HC]; mhc_sk_bwd(sv, dB, dBtil);
        for(int i=0;i<N;i++)for(int j=0;j<N;j++){ MHC_F dbt=dBtil[i*N+j];
            g->Sres[i*N+j]+=dbt;
            MHC_F s=0; for(int m=0;m<MHC_M;m++) s+=sv->Xhat[m]*p->Wres[m*N*N+i*N+j];
            g->a_res+=dbt*s;
            for(int m=0;m<MHC_M;m++){ g->Wres[m*N*N+i*N+j]+=dbt*p->a_res*sv->Xhat[m];
                dXhat[m]+=dbt*p->a_res*p->Wres[m*N*N+i*N+j]; } }
        // Xhat = v/rms; back to v=vec(X) -> dX
        MHC_F v[MHC_M]; for(int m=0;m<MHC_M;m++) v[m]=X[(m/DIM)*DIM*SEQ + (m%DIM)*SEQ + t];
        MHC_F c=0; for(int m=0;m<MHC_M;m++) c+=dXhat[m]*v[m];
        MHC_F r=sv->rms;
        for(int m=0;m<MHC_M;m++){ MHC_F dv=dXhat[m]/r - v[m]*c/(MHC_M*r*r*r);
            dX[(m/DIM)*DIM*SEQ + (m%DIM)*SEQ + t]+=dv; }
    }
}

// Doubly-stochasticity probe: max |rowsum-1|, |colsum-1| over all saved B (debug).
static MHC_F mhc_ds_residual(const MhcTape *T){
    const int N=N_HC; MHC_F worst=0;
    for(int t=0;t<SEQ;t++){ const MhcPos *sv=&T->pos[t];
        for(int i=0;i<N;i++){ MHC_F rs=0,cs=0; for(int j=0;j<N;j++){ rs+=sv->B[i*N+j]; cs+=sv->B[j*N+i]; }
            MHC_F er=fabs(rs-(MHC_F)1.0), ec=fabs(cs-(MHC_F)1.0);
            if(er>worst)worst=er; if(ec>worst)worst=ec; } }
    return worst;
}
