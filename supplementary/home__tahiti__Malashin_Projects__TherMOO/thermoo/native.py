"""Compile batch evaluators from the author's ZCAT and PaGMO's CEC2009 code."""
from .runtime import ROOT
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

_UF = None
_ZCAT = None


def build():
    build_dir = ROOT / "build/thermoo"
    build_dir.mkdir(parents=True, exist_ok=True)
    cc, cxx = shutil.which("gcc"), shutil.which("g++")
    if not cc or not cxx:
        raise RuntimeError("gcc and g++ are required for the native benchmark wrappers")
    source = ROOT / "sources/ZCAT/src"
    bridge = build_dir / "zcat_bridge.c"
    bridge.write_text('''#include "zcat-benchmark.h"
void thermoo_zcat(int p,int n,int m,int rows,const double *x,double *f) {
    static int prev_n=0,prev_m=0;
    if(n!=prev_n || m!=prev_m) {
        if(prev_n) zcat_unset();
        double *lb=0,*ub=0; zcat_set(n,m,1,0,1,0,&lb,&ub);
        prev_n=n; prev_m=m;
    }
    for(int i=0;i<rows;++i) ZCAT_MOP[p-1]((double*)(x+i*n),f+i*m,0,0);
}
''')
    files = sorted(p for p in source.glob("*.c") if p.name != "main.c")
    subprocess.run([cc, "-O3", "-std=gnu11", "-shared", "-fPIC", "-I", str(source), str(bridge), *map(str, files), "-lm", "-o", str(build_dir / "libzcat.so")], check=True)
    vendor = Path(__file__).parents[1] / "vendor/cec2009_pagmo.cpp"
    original = vendor.read_text()
    start = original.index("void cec2009::UF1(")
    end = original.index("void cec2009::CF1(")
    definitions = original[start:end]
    header = original[:original.index("#include")]
    declarations = "\n".join(f"void UF{i}(vector_double&,const vector_double&) const;" for i in range(1, 11))
    calls = "\n".join(f"case {i}: obj.UF{i}(f,x); break;" for i in range(1, 11))
    wrapper = header + '''\n#include <cmath>\n#include <vector>\nusing vector_double=std::vector<double>;
namespace detail { constexpr double pi(){return 3.141592653589793238462643383279502884;} }
struct cec2009 { unsigned m_dim;\n''' + declarations + "\n};\n" + definitions
    wrapper += '''\nextern "C" void thermoo_uf(int p,int n,int rows,const double *in,double *out){
        cec2009 obj; obj.m_dim=n; int m=p<8?2:3; vector_double x(n),f(m);
        for(int i=0;i<rows;++i){for(int j=0;j<n;++j)x[j]=in[i*n+j];
        switch(p){''' + calls + '''}\nfor(int j=0;j<m;++j)out[i*m+j]=f[j];}}
'''
    cpp = build_dir / "uf_bridge.cpp"
    cpp.write_text(wrapper)
    subprocess.run([cxx, "-O3", "-std=c++14", "-shared", "-fPIC", str(cpp), "-o", str(build_dir / "libuf.so")], check=True)
    (build_dir / "source_hashes.json").write_text(json.dumps({"pagmo_cec2009": hashlib.sha256(vendor.read_bytes()).hexdigest(), "zcat_sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}, indent=2))


def uf(p, X):
    global _UF
    X = np.ascontiguousarray(X, dtype=np.float64)
    if _UF is None:
        _UF = ctypes.CDLL(str(ROOT / "build/thermoo/libuf.so")).thermoo_uf
        _UF.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
        _UF.restype = None
    F = np.empty((len(X), 2 if p < 8 else 3))
    _UF(p, X.shape[1], len(X), X.ctypes.data, F.ctypes.data)
    return F


def zcat(p, X, m):
    global _ZCAT
    X = np.ascontiguousarray(X, dtype=np.float64)
    if _ZCAT is None:
        _ZCAT = ctypes.CDLL(str(ROOT / "build/thermoo/libzcat.so")).thermoo_zcat
        _ZCAT.argtypes = [ctypes.c_int] * 4 + [ctypes.c_void_p] * 2
        _ZCAT.restype = None
    F = np.empty((len(X), m))
    _ZCAT(p, X.shape[1], m, len(X), X.ctypes.data, F.ctypes.data)
    return F
