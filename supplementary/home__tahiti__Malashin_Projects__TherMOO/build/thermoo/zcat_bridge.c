#include "zcat-benchmark.h"
void thermoo_zcat(int p,int n,int m,int rows,const double *x,double *f) {
    static int prev_n=0,prev_m=0;
    if(n!=prev_n || m!=prev_m) {
        if(prev_n) zcat_unset();
        double *lb=0,*ub=0; zcat_set(n,m,1,0,1,0,&lb,&ub);
        prev_n=n; prev_m=m;
    }
    for(int i=0;i<rows;++i) ZCAT_MOP[p-1]((double*)(x+i*n),f+i*m,0,0);
}
