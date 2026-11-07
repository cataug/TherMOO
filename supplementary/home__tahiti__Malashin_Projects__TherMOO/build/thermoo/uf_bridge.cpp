/* Copyright 2017-2026 PaGMO development team

This file is part of the PaGMO library.

The PaGMO library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 3 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The PaGMO library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received codetail::pi()es of the GNU General Public License and the
GNU Lesser General Public License along with the PaGMO library.  If not,
see https://www.gnu.org/licenses/. */


#include <cmath>
#include <vector>
using vector_double=std::vector<double>;
namespace detail { constexpr double pi(){return 3.141592653589793238462643383279502884;} }
struct cec2009 { unsigned m_dim;
void UF1(vector_double&,const vector_double&) const;
void UF2(vector_double&,const vector_double&) const;
void UF3(vector_double&,const vector_double&) const;
void UF4(vector_double&,const vector_double&) const;
void UF5(vector_double&,const vector_double&) const;
void UF6(vector_double&,const vector_double&) const;
void UF7(vector_double&,const vector_double&) const;
void UF8(vector_double&,const vector_double&) const;
void UF9(vector_double&,const vector_double&) const;
void UF10(vector_double&,const vector_double&) const;
};
void cec2009::UF1(vector_double &f, const vector_double &x) const
{
    double count1, count2;
    double sum1, sum2, yj;

    sum1 = sum2 = 0.0;
    count1 = count2 = 0;
    for (decltype(m_dim) j = 2u; j <= m_dim; ++j) {
        yj = x[j - 1] - std::sin(6.0 * detail::pi() * x[0] + j * detail::pi() / (double)m_dim);
        yj = yj * yj;
        if (j % 2 == 0u) {
            sum2 += yj;
            count2++;
        } else {
            sum1 += yj;
            count1++;
        }
    }
    f[0] = x[0] + 2.0 * sum1 / count1;
    f[1] = 1.0 - std::sqrt(x[0]) + 2.0 * sum2 / count2;
}

void cec2009::UF2(vector_double &f, const vector_double &x) const
{
    double count1, count2;
    double sum1, sum2, yj;

    sum1 = sum2 = 0.0;
    count1 = count2 = 0;
    for (decltype(m_dim) j = 2u; j <= m_dim; ++j) {
        if (j % 2 == 0u) {
            yj = x[j - 1]
                 - 0.3 * x[0]
                       * (x[0] * std::cos(24.0 * detail::pi() * x[0] + 4.0 * j * detail::pi() / (double)m_dim) + 2.0)
                       * std::sin(6.0 * detail::pi() * x[0] + j * detail::pi() / (double)m_dim);
            sum2 += yj * yj;
            count2++;
        } else {
            yj = x[j - 1]
                 - 0.3 * x[0]
                       * (x[0] * std::cos(24.0 * detail::pi() * x[0] + 4.0 * j * detail::pi() / (double)m_dim) + 2.0)
                       * std::cos(6.0 * detail::pi() * x[0] + j * detail::pi() / (double)m_dim);
            sum1 += yj * yj;
            count1++;
        }
    }
    f[0] = x[0] + 2.0 * sum1 / count1;
    f[1] = 1.0 - std::sqrt(x[0]) + 2.0 * sum2 / count2;
}

void cec2009::UF3(vector_double &f, const vector_double &x) const
{
    double count1, count2;
    double sum1, sum2, prod1, prod2, yj, pj;

    sum1 = sum2 = 0.0;
    count1 = count2 = 0;
    prod1 = prod2 = 1.0;
    for (decltype(m_dim) j = 2u; j <= m_dim; ++j) {
        yj = x[j - 1] - std::pow(x[0], 0.5 * (1.0 + 3.0 * (j - 2.0) / ((double)m_dim - 2.0)));
        pj = std::cos(20.0 * yj * detail::pi() / std::sqrt(j + 0.0));
        if (j % 2 == 0u) {
            sum2 += yj * yj;
            prod2 *= pj;
            count2++;
        } else {
            sum1 += yj * yj;
            prod1 *= pj;
            count1++;
        }
    }
    f[0] = x[0] + 2.0 * (4.0 * sum1 - 2.0 * prod1 + 2.0) / count1;
    f[1] = 1.0 - std::sqrt(x[0]) + 2.0 * (4.0 * sum2 - 2.0 * prod2 + 2.0) / count2;
}

void cec2009::UF4(vector_double &f, const vector_double &x) const
{
    double count1, count2;
    double sum1, sum2, yj, hj;

    sum1 = sum2 = 0.0;
    count1 = count2 = 0;
    for (decltype(m_dim) j = 2u; j <= m_dim; ++j) {
        yj = x[j - 1] - std::sin(6.0 * detail::pi() * x[0] + j * detail::pi() / (double)m_dim);
        hj = std::abs(yj) / (1.0 + std::exp(2.0 * std::abs(yj)));
        if (j % 2 == 0u) {
            sum2 += hj;
            count2++;
        } else {
            sum1 += hj;
            count1++;
        }
    }
    f[0] = x[0] + 2.0 * sum1 / count1;
    f[1] = 1.0 - x[0] * x[0] + 2.0 * sum2 / count2;
}

void cec2009::UF5(vector_double &f, const vector_double &x) const
{
    double count1, count2;
    double sum1, sum2, yj, hj, N, E;

    sum1 = sum2 = 0.0;
    count1 = count2 = 0;
    N = 10.0;
    E = 0.1;
    for (decltype(m_dim) j = 2u; j <= m_dim; ++j) {
        yj = x[j - 1] - std::sin(6.0 * detail::pi() * x[0] + j * detail::pi() / (double)m_dim);
        hj = 2.0 * yj * yj - std::cos(4.0 * detail::pi() * yj) + 1.0;
        if (j % 2 == 0u) {
            sum2 += hj;
            count2++;
        } else {
            sum1 += hj;
            count1++;
        }
    }
    hj = (0.5 / N + E) * std::abs(std::sin(2.0 * N * detail::pi() * x[0]));
    f[0] = x[0] + hj + 2.0 * sum1 / count1;
    f[1] = 1.0 - x[0] + hj + 2.0 * sum2 / count2;
}

void cec2009::UF6(vector_double &f, const vector_double &x) const
{
    double count1, count2;
    double sum1, sum2, prod1, prod2, yj, hj, pj, N, E;
    N = 2.0;
    E = 0.1;

    sum1 = sum2 = 0.0;
    count1 = count2 = 0;
    prod1 = prod2 = 1.0;
    for (decltype(m_dim) j = 2u; j <= m_dim; ++j) {
        yj = x[j - 1] - std::sin(6.0 * detail::pi() * x[0] + j * detail::pi() / (double)m_dim);
        pj = std::cos(20.0 * yj * detail::pi() / std::sqrt(j + 0.0));
        if (j % 2 == 0u) {
            sum2 += yj * yj;
            prod2 *= pj;
            count2++;
        } else {
            sum1 += yj * yj;
            prod1 *= pj;
            count1++;
        }
    }

    hj = 2.0 * (0.5 / N + E) * std::sin(2.0 * N * detail::pi() * x[0]);
    if (hj < 0.0) hj = 0.0;
    f[0] = x[0] + hj + 2.0 * (4.0 * sum1 - 2.0 * prod1 + 2.0) / count1;
    f[1] = 1.0 - x[0] + hj + 2.0 * (4.0 * sum2 - 2.0 * prod2 + 2.0) / count2;
}

void cec2009::UF7(vector_double &f, const vector_double &x) const
{
    double count1, count2;
    double sum1, sum2, yj;

    sum1 = sum2 = 0.0;
    count1 = count2 = 0;
    for (decltype(m_dim) j = 2u; j <= m_dim; ++j) {
        yj = x[j - 1] - std::sin(6.0 * detail::pi() * x[0] + j * detail::pi() / (double)m_dim);
        if (j % 2 == 0u) {
            sum2 += yj * yj;
            count2++;
        } else {
            sum1 += yj * yj;
            count1++;
        }
    }
    yj = std::pow(x[0], 0.2);
    f[0] = yj + 2.0 * sum1 / count1;
    f[1] = 1.0 - yj + 2.0 * sum2 / count2;
}

void cec2009::UF8(vector_double &f, const vector_double &x) const
{
    double count1, count2, count3;
    double sum1, sum2, sum3, yj;

    sum1 = sum2 = sum3 = 0.0;
    count1 = count2 = count3 = 0;
    for (decltype(m_dim) j = 3u; j <= m_dim; ++j) {
        yj = x[j - 1] - 2.0 * x[1] * std::sin(2.0 * detail::pi() * x[0] + j * detail::pi() / (double)m_dim);
        if (j % 3 == 1u) {
            sum1 += yj * yj;
            count1++;
        } else if (j % 3 == 2u) {
            sum2 += yj * yj;
            count2++;
        } else {
            sum3 += yj * yj;
            count3++;
        }
    }
    f[0] = std::cos(0.5 * detail::pi() * x[0]) * std::cos(0.5 * detail::pi() * x[1]) + 2.0 * sum1 / count1;
    f[1] = std::cos(0.5 * detail::pi() * x[0]) * std::sin(0.5 * detail::pi() * x[1]) + 2.0 * sum2 / count2;
    f[2] = std::sin(0.5 * detail::pi() * x[0]) + 2.0 * sum3 / count3;
}

void cec2009::UF9(vector_double &f, const vector_double &x) const
{
    double count1, count2, count3;
    double sum1, sum2, sum3, yj, E;

    E = 0.1;
    sum1 = sum2 = sum3 = 0.0;
    count1 = count2 = count3 = 0;
    for (decltype(m_dim) j = 3u; j <= m_dim; ++j) {
        yj = x[j - 1] - 2.0 * x[1] * std::sin(2.0 * detail::pi() * x[0] + j * detail::pi() / (double)m_dim);
        if (j % 3 == 1u) {
            sum1 += yj * yj;
            count1++;
        } else if (j % 3 == 2u) {
            sum2 += yj * yj;
            count2++;
        } else {
            sum3 += yj * yj;
            count3++;
        }
    }
    yj = (1.0 + E) * (1.0 - 4.0 * (2.0 * x[0] - 1.0) * (2.0 * x[0] - 1.0));
    if (yj < 0.0) yj = 0.0;
    f[0] = 0.5 * (yj + 2 * x[0]) * x[1] + 2.0 * sum1 / count1;
    f[1] = 0.5 * (yj - 2 * x[0] + 2.0) * x[1] + 2.0 * sum2 / count2;
    f[2] = 1.0 - x[1] + 2.0 * sum3 / count3;
}

void cec2009::UF10(vector_double &f, const vector_double &x) const
{
    double count1, count2, count3;
    double sum1, sum2, sum3, yj, hj;

    sum1 = sum2 = sum3 = 0.0;
    count1 = count2 = count3 = 0;
    for (decltype(m_dim) j = 3u; j <= m_dim; ++j) {
        yj = x[j - 1] - 2.0 * x[1] * std::sin(2.0 * detail::pi() * x[0] + j * detail::pi() / (double)m_dim);
        hj = 4.0 * yj * yj - std::cos(8.0 * detail::pi() * yj) + 1.0;
        if (j % 3 == 1u) {
            sum1 += hj;
            count1++;
        } else if (j % 3 == 2u) {
            sum2 += hj;
            count2++;
        } else {
            sum3 += hj;
            count3++;
        }
    }
    f[0] = std::cos(0.5 * detail::pi() * x[0]) * std::cos(0.5 * detail::pi() * x[1]) + 2.0 * sum1 / count1;
    f[1] = std::cos(0.5 * detail::pi() * x[0]) * std::sin(0.5 * detail::pi() * x[1]) + 2.0 * sum2 / count2;
    f[2] = std::sin(0.5 * detail::pi() * x[0]) + 2.0 * sum3 / count3;
}

/****************************************************************************/
// constraint test instances
/****************************************************************************/

extern "C" void thermoo_uf(int p,int n,int rows,const double *in,double *out){
        cec2009 obj; obj.m_dim=n; int m=p<8?2:3; vector_double x(n),f(m);
        for(int i=0;i<rows;++i){for(int j=0;j<n;++j)x[j]=in[i*n+j];
        switch(p){case 1: obj.UF1(f,x); break;
case 2: obj.UF2(f,x); break;
case 3: obj.UF3(f,x); break;
case 4: obj.UF4(f,x); break;
case 5: obj.UF5(f,x); break;
case 6: obj.UF6(f,x); break;
case 7: obj.UF7(f,x); break;
case 8: obj.UF8(f,x); break;
case 9: obj.UF9(f,x); break;
case 10: obj.UF10(f,x); break;}
for(int j=0;j<m;++j)out[i*m+j]=f[j];}}
