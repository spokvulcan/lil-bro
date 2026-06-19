// Finite-difference check of the SwiGLU-clamp VJP (issue #9).
// Verifies the analytic backward used in train.m matches numerical gradients,
// and that the gradient is exactly zero in the clamped regions — across inputs
// that straddle both clamp boundaries (silu>10 cap, |h3|>10 linear clamp).
#include <stdio.h>
#include <math.h>

static double silu(double x){ return x / (1.0 + exp(-x)); }
static double siluprime(double x){ double s=1.0/(1.0+exp(-x)); return s*(1.0 + x*(1.0-s)); }
static double clampv(double v,double lo,double hi){ return v>hi?hi:(v<lo?lo:v); }

// Forward: gate = min(silu(h1),10) * clamp(h3,-10,10)
static double gate_fwd(double h1,double h3){
    double siluc = fmin(silu(h1),10.0);
    double h3c   = clampv(h3,-10.0,10.0);
    return siluc*h3c;
}

int main(void){
    // (h1,h3) cases: in-range, gate-capped (h1 huge -> silu>10), linear-clamped
    // (|h3|>10), and both clamped.
    double pts[][2] = {
        {0.5, 0.3}, {2.0, -1.5}, {-3.0, 4.0},   // in range
        {30.0, 2.0},                            // silu(h1)>10 -> gate capped
        {1.0, 25.0}, {1.0, -25.0},              // |h3|>10 -> linear clamped
        {40.0, 50.0},                           // both clamped
        {12.0, 9.99}, {1.0, 10.0001},           // near boundaries
    };
    int npts = sizeof(pts)/sizeof(pts[0]);
    double dgate = 1.7;       // arbitrary upstream grad
    double eps = 1e-5, tol = 2e-3;
    int fails = 0;
    printf("  h1        h3      | dh1(an)   dh1(num) | dh3(an)   dh3(num) | clamp\n");
    for (int i=0;i<npts;i++){
        double h1=pts[i][0], h3=pts[i][1];
        double s = silu(h1);
        double siluc = fmin(s,10.0), h3c = clampv(h3,-10.0,10.0);
        double gatemask = s<10.0?1.0:0.0;
        double linmask  = (h3<10.0 && h3>-10.0)?1.0:0.0;
        // analytic (matches train.m clamp backward)
        double dh1_an = dgate * h3c * gatemask * siluprime(h1);
        double dh3_an = dgate * siluc * linmask;
        // numerical (central difference through the clamped forward)
        double dh1_num = dgate*(gate_fwd(h1+eps,h3)-gate_fwd(h1-eps,h3))/(2*eps);
        double dh3_num = dgate*(gate_fwd(h1,h3+eps)-gate_fwd(h1,h3-eps))/(2*eps);
        int clamped = (gatemask==0.0)||(linmask==0.0);
        int ok = fabs(dh1_an-dh1_num)<tol && fabs(dh3_an-dh3_num)<tol;
        // in clamped regions the corresponding analytic grad must be exactly 0
        if (gatemask==0.0 && dh1_an!=0.0) ok=0;
        if (linmask==0.0  && dh3_an!=0.0) ok=0;
        printf("%7.3f %8.3f | %8.4f %8.4f | %8.4f %8.4f | %s %s\n",
               h1,h3,dh1_an,dh1_num,dh3_an,dh3_num, clamped?"Y":"-", ok?"ok":"FAIL");
        if(!ok) fails++;
    }
    printf(fails? "\nFAIL: %d cases\n" : "\nPASS: all %d cases match (incl. zero-grad in clamped regions)\n",
           fails?fails:npts);
    return fails?1:0;
}
