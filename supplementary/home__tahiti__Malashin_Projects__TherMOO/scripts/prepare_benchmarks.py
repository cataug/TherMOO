import csv
import hashlib
import importlib.metadata as metadata
import json
import math
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for name in (
    "sources", "data/reference_fronts", "configs",
    "reports", "build", "outputs", "logs"
):
    (ROOT / name).mkdir(parents=True, exist_ok=True)


def run(args, cwd=None):
    print("\n[RUN] " + " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


# Скачиваем исходники. Существующие копии сохраняем без обновления.
repos = {
    "jMetalPy": "https://github.com/jMetal/jMetalPy.git",
    "pymoo": "https://github.com/anyoptimization/pymoo.git",
    "ZCAT": "https://github.com/evo-mx/ZCAT.git",
    "jMetal": "https://github.com/jMetal/jMetal.git",
}

commits = {}
for name, url in repos.items():
    dest = ROOT / "sources" / name

    if not dest.exists():
        options = ["--depth", "1", "--no-tags"]
        if name == "jMetal":
            options += ["--filter=blob:none", "--sparse"]
        run(["git", "clone", *options, url, dest])
    elif not (dest / ".git").exists():
        raise RuntimeError(f"{dest} уже существует и не является Git-копией")
    else:
        print(f"[KEEP] {dest}", flush=True)

    if name == "jMetal":
        run([
            "git", "-C", dest, "sparse-checkout", "set",
            "resources/referenceFronts"
        ])

    commits[name] = subprocess.check_output(
        ["git", "-C", str(dest), "rev-parse", "HEAD"],
        text=True
    ).strip()


groups = {
    "ZDT": [f"ZDT{i}" for i in range(1, 7)],
    "DTLZ": [f"DTLZ{i}" for i in range(1, 8)],
    "WFG": [f"WFG{i}" for i in range(1, 10)],
    "LZ09": [f"LZ09_F{i}" for i in range(1, 10)],
    "UF": [f"UF{i}" for i in range(1, 11)],
    "ZCAT": [f"ZCAT{i}" for i in range(1, 21)],
}

expected = [p for family in groups.values() for p in family]
canonical = {p.replace("_", "").upper(): p for p in expected}

pattern = re.compile(
    r"^(ZDT[1-6]|DTLZ[1-7]|WFG[1-9]|LZ09[_-]?F[1-9]|"
    r"UF(?:10|[1-9])|ZCAT(?:20|1[0-9]|[1-9]))(?:[._-]|$)",
    re.I
)

# Копируем готовые фронты, сохраняя их происхождение.
refs = ROOT / "data/reference_fronts"

for provider, relative in [
    ("jMetalPy", "resources/reference_fronts"),
    ("jMetal", "resources/referenceFronts"),
]:
    source = ROOT / "sources" / provider / relative
    if not source.is_dir():
        raise RuntimeError(f"Не найдена папка фронтов: {source}")

    target = refs / provider
    target.mkdir(exist_ok=True)
    count = 0

    for path in sorted(source.rglob("*")):
        if (
            path.is_file()
            and path.suffix.lower() in (".pf", ".csv", ".txt")
            and pattern.match(path.name)
        ):
            output = target / path.relative_to(source)
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, output)
            count += 1

    print(f"[FRONTS] {provider}: {count} файлов", flush=True)


generated = refs / "generated"
generated.mkdir(exist_ok=True)

# Точный фронт бинарной ZDT5 в исходных, ненормированных координатах.
with (generated / "ZDT5.raw.2D.pf").open("w") as stream:
    stream.write(
        "# Native unnormalized ZDT5 objectives; "
        "use normalize=False in pymoo\n"
    )
    for f1 in range(1, 32):
        stream.write(f"{f1:.17g} {10.0 / f1:.17g}\n")


# Генерация ZCAT через авторскую C-реализацию.
cc = shutil.which("gcc") or shutil.which("cc")
if not cc:
    raise RuntimeError(
        "Исходники скачаны. Для генерации фронтов ZCAT нужен gcc/cc."
    )

src = ROOT / "sources/ZCAT/src"
driver = ROOT / "build/generate_zcat.c"

driver.write_text(r'''
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
''')

binary = ROOT / "build/generate_zcat"
c_files = [
    p for p in sorted(src.glob("*.c"))
    if p.name != "main.c"
]

run([
    cc, "-O2", "-std=gnu11", "-I", src,
    driver, *c_files, "-lm", "-o", binary
])

for m, points in [(2, 5000), (3, 10000), (5, 20000)]:
    # Авторскому генератору нужен доступ к src/seg.
    run([binary, m, points, generated], cwd=src)


# Проверяем числовой формат и строим индекс всех фронтов.
index = []

for path in sorted(refs.rglob("*")):
    match = pattern.match(path.name)
    if (
        not path.is_file()
        or path.suffix.lower() not in (".pf", ".csv", ".txt")
        or not match
    ):
        continue

    key = match.group(1).replace("_", "").replace("-", "").upper()
    problem = canonical[key]
    rows, columns = 0, None

    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "%")):
            continue

        tokens = re.split(r"[\s,;]+", line)
        if rows == 0 and all(
            t.lower().startswith(("f", "obj")) for t in tokens
        ):
            continue

        values = [float(t) for t in tokens]
        columns = columns or len(values)

        if (
            len(values) != columns
            or not all(math.isfinite(v) for v in values)
        ):
            raise RuntimeError(f"Некорректный фронт: {path}")

        rows += 1

    if not rows:
        raise RuntimeError(f"Пустой фронт: {path}")

    index.append({
        "problem": problem,
        "objectives": columns,
        "points": rows,
        "provider": path.relative_to(refs).parts[0],
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })


# Проверяем покрытие всех 61 задач основного эксперимента.
selected = {}

for family, problems in groups.items():
    for problem in problems:
        if family == "ZDT":
            m = 2
        elif family == "LZ09":
            m = 3 if problem == "LZ09_F6" else 2
        elif family == "UF":
            m = 2 if int(problem[2:]) <= 7 else 3
        else:
            m = 3

        candidates = [
            r for r in index
            if r["problem"] == problem and r["objectives"] == m
        ]

        if not candidates:
            raise RuntimeError(
                f"Нет фронта основного эксперимента: {problem}, M={m}"
            )

        selected[problem] = max(
            candidates,
            key=lambda r: (r["provider"] == "generated", r["points"])
        )


# Только проверяем версии установленных пакетов.
packages = {}
for name in (
    "numpy", "scipy", "pandas", "matplotlib",
    "pymoo", "jmetalpy", "torch"
):
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = None

manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.executable,
    "platform": platform.platform(),
    "repositories": {
        name: {"url": repos[name], "commit": commits[name]}
        for name in repos
    },
    "installed_packages": packages,
    "benchmark_families": groups,
    "zcat_generation": {
        "points": {"2D": 5000, "3D": 10000, "5D": 20000},
        "seed": "20261104 + M",
        "level": 1,
        "bias": False,
        "complicated_PS": True,
        "imbalance": False,
    },
    "reference_fronts": index,
}

for relative, content in [
    ("configs/benchmarks.json", groups),
    ("configs/reference_fronts_main.json", selected),
    ("reports/download_manifest.json", manifest),
]:
    (ROOT / relative).write_text(
        json.dumps(content, indent=2) + "\n"
    )

with (ROOT / "reports/reference_fronts_index.csv").open(
    "w", newline=""
) as stream:
    writer = csv.DictWriter(stream, fieldnames=list(index[0]))
    writer.writeheader()
    writer.writerows(index)

print("\n[READY] Main reference fronts: 61/61")
print(f"[READY] Reference-front files: {len(index)}")
print("[ENV] " + json.dumps(packages))
print(f"[ROOT] {ROOT}")
print("[MANIFEST] reports/download_manifest.json")
