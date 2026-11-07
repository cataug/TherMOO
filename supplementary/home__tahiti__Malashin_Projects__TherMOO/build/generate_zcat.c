
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "zcat-benchmark.h"
#include "zcat-rnd-opt-sol.h"

int main(int argc, char **argv) {
    if (argc != 4) return 2;

    int m = atoi(argv[1]), count = atoi(argv[2]), d = 10*m;
    if (m < 2 || m > 5 || count < 1) return 2;

    double *lb = NULL, *ub = NULL;
    zcat_set(d, m, 1, 0, 1, 0, &lb, &ub);

    double *x = malloc(d*sizeof(double));
    double *f = malloc(m*sizeof(double));
    if (!x || !f) return 3;

    srand(20261104u + (unsigned)m);

    for (int p=zcat1; p<=zcat20; ++p) {
        char name[4096];
        snprintf(
            name, sizeof(name), "%s/%s.%dD.pf",
            argv[3], ZCAT_MOP_STR[p], m
        );

        FILE *out = fopen(name, "w");
        if (!out) { perror(name); return 4; }

        fprintf(
            out,
            "# M=%d D=%d level=1 bias=0 complicatedPS=1 "
            "imbalance=0 seed=%u\n",
            m, d, 20261104u+(unsigned)m
        );

        for (int i=0; i<count; ++i) {
            zcat_rnd_opt_sol(p, x, m, d);
            ZCAT_MOP[p](x, f, NULL, NULL);

            for (int j=0; j<m; ++j) {
                if (!isfinite(f[j])) return 5;
                fprintf(out, j ? " %.17g" : "%.17g", f[j]);
            }
            fputc('\n', out);
        }

        fclose(out);
        printf("[ZCAT] %s: %d points\n", ZCAT_MOP_STR[p], count);
        fflush(stdout);
    }

    free(x);
    free(f);
    zcat_unset();
    return 0;
}
