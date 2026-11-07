#!/usr/bin/env python3
"""Copy all TherMOO research artifacts to GitHub, with large data in Release assets.
Runs in the foreground, never modifies the experiment directories.
Requires Python 3.9+, git, GitHub CLI (gh), and configured authentication.
"""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time

REPO = os.getenv("THERMOO_GH_REPO", "cataug/TherMOO")
REMOTE = os.getenv("THERMOO_GIT_REMOTE", f"https://github.com/{REPO}.git")
WORK = Path(os.getenv("THERMOO_GIT_WORK", "/home/tahiti/TherMOO_GITHUB_UPLOAD")).expanduser().resolve()
SOURCES = [Path(x).expanduser().resolve() for x in (
    os.getenv("THERMOO_ROOT1", "/home/tahiti/TherMOO"),
    os.getenv("THERMOO_ROOT2", "/home/tahiti/Malashin_Projects/TherMOO"),
)]
DIRECT_LIMIT = int(float(os.getenv("THERMOO_DIRECT_MIB", "45")) * 1024**2)
GIT_BUDGET = int(float(os.getenv("THERMOO_GIT_BUDGET_GIB", "3")) * 1024**3)
GIT_FILE_BUDGET = int(os.getenv("THERMOO_GIT_FILE_BUDGET", "25000"))
LOG_LIMIT = int(float(os.getenv("THERMOO_LOG_MIB", "4")) * 1024**2)
BATCH_LIMIT = int(float(os.getenv("THERMOO_BATCH_MIB", "300")) * 1024**2)
BATCH_FILES = int(os.getenv("THERMOO_BATCH_FILES", "750"))
PART_LIMIT = int(float(os.getenv("THERMOO_RELEASE_PART_MIB", "1536")) * 1024**2)
MIN_AGE = int(os.getenv("THERMOO_MIN_AGE_SECONDS", "30"))
PUSH_DELAY = float(os.getenv("THERMOO_PUSH_DELAY", "11"))
DRY_RUN = os.getenv("THERMOO_DRY_RUN", "0") == "1"

DIRECT_EXT = {".py", ".sh", ".md", ".rst", ".tex", ".bib", ".toml", ".ini", ".cfg", ".yaml", ".yml", ".c", ".cpp", ".h", ".hpp", ".js", ".ts", ".css", ".html", ".ipynb", ".gitignore", ".gitattributes", ".txt"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg", ".pdf", ".eps", ".tif", ".tiff", ".bmp"}
DATA_EXT = {".csv", ".tsv", ".json", ".jsonl", ".log", ".parquet", ".pkl", ".pickle", ".npz", ".npy", ".zip", ".gz", ".xz", ".zst"}
IGNORE_DIR = {".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".cache", "node_modules", "site-packages", ".ipynb_checkpoints", ".idea", ".vscode"}
IGNORE_FILE = {".env", ".env.local", ".env.production", ".env.development", "id_rsa", "id_ed25519", "credentials.json", "secrets.json", ".netrc", ".pypirc", "token.txt", "token.json"}
SECRET_NAME = re.compile(r"(?:^|[_.-])(?:secret|api[_-]?key|access[_-]?token|private[_-]?key|credential|password)(?:[_.-]|$)", re.I)
SECRET_BODY = re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----)")


def run(args, cwd=None, capture=False):
    print("$", " ".join(map(str, args)), flush=True)
    return subprocess.run([str(a) for a in args], cwd=cwd, check=True,
                          text=True, capture_output=capture)


def sha_file(path, bufsize=4 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bufsize), b""):
            h.update(b)
    return h.hexdigest()


def atomic_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    temp = dst.with_name(dst.name + ".thermoo-upload-tmp")
    try:
        with open(src, "rb") as inp, open(temp, "wb") as out:
            shutil.copyfileobj(inp, out, 4 * 1024 * 1024)
        if src.stat().st_size != temp.stat().st_size:
            raise RuntimeError(f"Source changed during copy: {src}")
        os.replace(temp, dst)
    finally:
        if temp.exists():
            temp.unlink()


def eligible(src, relative, cutoff, skipped):
    name = src.name
    if name.lower() in IGNORE_FILE or SECRET_NAME.search(name):
        skipped.append((str(relative), "possible secret filename"))
        return False
    if src.is_symlink():
        skipped.append((str(relative), "symlink (not dereferenced)"))
        return False
    if not src.is_file():
        skipped.append((str(relative), "not a regular file"))
        return False
    st = src.stat()
    if st.st_mtime > cutoff:
        skipped.append((str(relative), "file modified recently; rerun after writer finishes"))
        return False
    if st.st_size <= 16 * 1024**2 and (src.suffix.lower() in DIRECT_EXT or src.suffix.lower() in {".json", ".jsonl", ".log", ".txt", ".yaml", ".toml"}):
        with open(src, "rb") as f:
            if SECRET_BODY.search(f.read()):
                skipped.append((str(relative), "possible credential content; REVIEW"))
                return False
    return True


def get_entries(roots, cutoff):
    skipped = []
    entries = []
    seen = set()
    primary = next((p for p in roots if (p / "thermoo").is_dir()), roots[0])
    print(f"[SOURCE] Main project: {primary}")
    for root in [primary] + [p for p in roots if p != primary]:
        prefix = Path() if root == primary else Path("supplementary") / root.as_posix().strip("/").replace("/", "__")
        for base, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in IGNORE_DIR and not d.startswith(".venv"))
            for filename in sorted(files):
                source = Path(base) / filename
                relative = prefix / source.relative_to(root)
                if str(relative) in seen:
                    raise RuntimeError(f"Duplicate path {relative}")
                seen.add(str(relative))
                try:
                    if eligible(source, relative, cutoff, skipped):
                        entries.append((relative, source, source.stat().st_size))
                except (OSError, PermissionError) as err:
                    skipped.append((str(relative), f"read failure: {err}"))
    return entries, skipped


def priority(relative, size, crowded):
    suffix = relative.suffix.lower()
    parts = [p.lower() for p in relative.parts]
    if size > DIRECT_LIMIT or crowded:
        return 99
    if suffix in IMAGE_EXT:
        return 0
    if suffix in DIRECT_EXT:
        return 1
    if (suffix in {".json", ".csv", ".tsv"} and size <= LOG_LIMIT):
        return 2
    if suffix in DATA_EXT and size <= LOG_LIMIT:
        return 3
    if size <= 256 * 1024:
        return 4
    return 9


def split_choices(entries):
    dir_counts = collections.Counter(str(rel.parent) for rel, _, _ in entries)
    rows = []
    for rel, src, sz in entries:
        crowded = dir_counts[str(rel.parent)] > 2900 and rel.suffix.lower() not in IMAGE_EXT and rel.suffix.lower() not in DIRECT_EXT
        rows.append((priority(rel, sz, crowded), rel, src, sz))
    rows.sort(key=lambda x: (x[0], str(x[1])))
    direct, release = [], []
    used = 0
    for rank, rel, src, size in rows:
        if rank <= 4 and used + size <= GIT_BUDGET and len(direct) < GIT_FILE_BUDGET:
            direct.append((rel, src, size))
            used += size
        else:
            release.append((rel, src, size))
    return direct, release


def upload_git(work, paths, message, branch):
    pending = []
    bytes_pending = 0
    pushed = 0
    def flush():
        nonlocal pending, bytes_pending, pushed
        if not pending:
            return
        # Stage individual file paths forcibly, so source .gitignore cannot hide results.
        for start in range(0, len(pending), 200):
            run(["git", "add", "-f", "--", *pending[start:start + 200]], cwd=work)
        dirty = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=work)
        if dirty.returncode:
            run(["git", "commit", "-m", f"{message} (batch {pushed + 1})"], cwd=work)
            run(["git", "push", "-u", "origin", f"HEAD:{branch}"], cwd=work)
            pushed += 1
            print(f"[GIT] Uploaded batch {pushed}: {len(pending)} paths / {bytes_pending / 1024**2:.1f} MiB", flush=True)
            if PUSH_DELAY:
                time.sleep(PUSH_DELAY)
        pending, bytes_pending = [], 0
    for rel, size in paths:
        if pending and (bytes_pending + size > BATCH_LIMIT or len(pending) >= BATCH_FILES):
            flush()
        pending.append(str(rel))
        bytes_pending += size
    flush()


class HashingReader:
    def __init__(self, fileobj):
        self.fileobj = fileobj
        self.h = hashlib.sha256()
    def read(self, n=-1):
        b = self.fileobj.read(n)
        self.h.update(b)
        return b


class ReleasePartWriter:
    def __init__(self, repo, tag, parts_dir):
        self.repo, self.tag, self.parts_dir = repo, tag, parts_dir
        self.file = None
        self.hasher = None
        self.index = 0
        self.count = 0
        self.parts = []
        self.closed = False
        self.parts_dir.mkdir(parents=True, exist_ok=True)
    def writable(self):
        return True
    def _open(self):
        self.index += 1
        self.count = 0
        self.hasher = hashlib.sha256()
        self.path = self.parts_dir / f"thermoo-raw.tar.gz.part{self.index:04d}"
        self.file = open(self.path, "wb")
        print(f"[RELEASE] Writing {self.path.name}", flush=True)
    def _complete(self):
        if self.file is None:
            return
        self.file.flush()
        self.file.close()
        digest = self.hasher.hexdigest()
        size = self.count
        run(["gh", "release", "upload", self.tag, str(self.path), "--repo", self.repo, "--clobber"])
        self.parts.append((self.path.name, digest, size))
        print(f"[RELEASE] Uploaded {self.path.name} / {size / 1024**2:.1f} MiB", flush=True)
        self.path.unlink()
        self.file = None
    def write(self, b):
        if self.closed:
            raise ValueError("closed release stream")
        if not isinstance(b, (bytes, bytearray, memoryview)):
            raise TypeError(type(b))
        view = memoryview(b)
        while view:
            if self.file is None:
                self._open()
            n = min(len(view), PART_LIMIT - self.count)
            piece = view[:n]
            self.file.write(piece)
            self.hasher.update(piece)
            self.count += n
            view = view[n:]
            if self.count >= PART_LIMIT:
                self._complete()
        return len(b)
    def flush(self):
        if self.file:
            self.file.flush()
    def close(self):
        if not self.closed:
            self._complete()
            self.closed = True


def main():
    roots = list(dict.fromkeys(p for p in SOURCES if p.is_dir()))
    if not roots:
        raise RuntimeError("No TherMOO directory found. Set THERMOO_ROOT1/THERMOO_ROOT2.")
    if any(WORK == p or p in WORK.parents or WORK in p.parents for p in roots):
        raise RuntimeError("Working clone must be OUTSIDE all source folders")
    cutoff = time.time() - MIN_AGE
    print(f"[SCAN] Roots: {[str(x) for x in roots]} (minimum file age: {MIN_AGE}s)", flush=True)
    entries, skipped = get_entries(roots, cutoff)
    if not entries:
        raise RuntimeError("No eligible source files")
    direct, release = split_choices(entries)
    print(f"[PLAN] {len(entries):,} regular files / {sum(x[2] for x in entries) / 1024**3:.3f} GiB")
    print(f"[PLAN] Git: {len(direct):,} files / {sum(x[2] for x in direct) / 1024**3:.3f} GiB")
    print(f"[PLAN] Release archive: {len(release):,} files / {sum(x[2] for x in release) / 1024**3:.3f} GiB")
    print(f"[PLAN] Skipped: {len(skipped)} (see manifest)", flush=True)
    if DRY_RUN:
        for name, reason in skipped[:30]:
            print("  SKIPPED", name, reason)
        print("[DRY RUN] No files copied or uploaded.")
        return
    for command in ("git", "gh"):
        if not shutil.which(command):
            raise RuntimeError(f"Required executable not found: {command}")
    run(["gh", "auth", "status"])
    run(["gh", "auth", "setup-git"])
    if not (WORK / ".git").is_dir():
        if WORK.exists() and any(WORK.iterdir()):
            raise RuntimeError(f"Working directory is nonempty, not a Git clone: {WORK}")
        WORK.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", REMOTE, str(WORK)])
    else:
        actual = run(["git", "remote", "get-url", "origin"], cwd=WORK, capture=True).stdout.strip()
        if actual != REMOTE:
            raise RuntimeError(f"Wrong origin {actual}; expected {REMOTE}")
        if subprocess.run(["git", "status", "--porcelain"], cwd=WORK, capture_output=True).stdout.strip():
            print("[NOTICE] Existing uncommitted changes found in upload clone; new upload files will be staged selectively.")
        run(["git", "pull", "--ff-only"], cwd=WORK)
    branch = run(["git", "branch", "--show-current"], cwd=WORK, capture=True).stdout.strip() or "main"
    for field in ("user.name", "user.email"):
        result = subprocess.run(["git", "config", field], cwd=WORK, capture_output=True, text=True)
        if not result.stdout.strip():
            raise RuntimeError(f"Git {field} is not configured. Set it with git config --global {field} ...")

    manifest = []
    git_paths = []
    for i, (rel, src, size) in enumerate(direct, 1):
        dst = WORK / rel
        if not dst.exists() or dst.stat().st_size != size or sha_file(dst) != sha_file(src):
            atomic_copy(src, dst)
        digest = sha_file(dst)
        manifest.append(dict(path=rel.as_posix(), size=size, sha256=digest, storage="git"))
        git_paths.append((rel, size))
        if i % 500 == 0:
            print(f"[COPY] {i:,}/{len(direct):,} Git files prepared", flush=True)

    tag = ""
    if release:
        tag = "thermoo-full-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
    info = ("# TherMOO research export\n\n"
            "Scripts, reports, small logs, and visualizations are stored directly in this repository. "
            "Larger experimental artifacts and high-cardinality result directories are stored losslessly "
            "in a GitHub Release attached to this repository.\n\n"
            f"Release tag: `{tag or 'not required'}`.\n\n"
            "Primary TherMOO paths are unchanged. Files from a second TherMOO directory "
            "are placed under `supplementary/`.\n\n")
    if tag:
        info += ("After cloning, restore the archive with GitHub CLI:\n\n```bash\n"
                 f"gh release download {tag} -R {REPO} "
                 "-p 'thermoo-raw.tar.gz.part*' -D /tmp/thermoo-parts\n"
                 "cat /tmp/thermoo-parts/thermoo-raw.tar.gz.part* | tar -xz -C .\n"
                 "python3 verify_thermoo_export.py\n```\n\n")
    else:
        info += "No Release archive was required. Verify with `python3 verify_thermoo_export.py`.\n\n"
    info += ("Files excluded for safety (e.g., virtual environments, keys, credentials, active writes, or symlinks) "
             "are reported in `thermoo_export_exclusions.json`. Their source locations are not modified.\n"
             "Every archived original and every Git-tracked original is recorded with size and SHA-256 "
             "in `thermoo_export_manifest.jsonl`.\n")
    (WORK / "THERMOO_UPLOAD_README.md").write_text(info, encoding="utf-8")
    (WORK / "thermoo_export_exclusions.json").write_text(json.dumps([{"path": p, "reason": why} for p, why in skipped], indent=2), encoding="utf-8")
    verify_script = '''#!/usr/bin/env python3
import hashlib, json, pathlib, sys
root = pathlib.Path(__file__).resolve().parent
missing = []; mismatch = []; count = 0
for line in (root / "thermoo_export_manifest.jsonl").open(encoding="utf-8"):
    record = json.loads(line); path = root / record["path"]
    if not path.is_file():
        missing.append(record["path"]); continue
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4*1024*1024), b""):
            h.update(block)
    if path.stat().st_size != record["size"] or h.hexdigest() != record["sha256"]:
        mismatch.append(record["path"])
    count += 1
print(f"Verified {count} originals; missing={len(missing)}, mismatched={len(mismatch)}")
for label, items in [("MISSING",missing),("MISMATCH",mismatch)]:
    for name in items[:30]: print(label,name)
sys.exit(1 if missing or mismatch else 0)
'''
    (WORK / "verify_thermoo_export.py").write_text(verify_script, encoding="utf-8")
    git_paths.extend([(Path("THERMOO_UPLOAD_README.md"), (WORK / "THERMOO_UPLOAD_README.md").stat().st_size),
                      (Path("thermoo_export_exclusions.json"), (WORK / "thermoo_export_exclusions.json").stat().st_size),
                      (Path("verify_thermoo_export.py"), (WORK / "verify_thermoo_export.py").stat().st_size)])
    upload_git(WORK, git_paths, "Add TherMOO sources, figures and logs", branch)

    if release:
        run(["gh", "release", "create", tag, "--repo", REPO, "--target", branch,
             "--title", f"TherMOO full experimental data {tag}",
             "--notes", "Lossless multipart tar.gz archive of experiment logs, results and large files. See THERMOO_UPLOAD_README.md for restoration."])
        writer = ReleasePartWriter(REPO, tag, WORK.parent / (WORK.name + "_release_parts"))
        try:
            with tarfile.open(fileobj=writer, mode="w|gz", compresslevel=3, format=tarfile.PAX_FORMAT) as tar:
                for i, (rel, src, size) in enumerate(release, 1):
                    current = src.stat()
                    if current.st_size != size or current.st_mtime > cutoff:
                        raise RuntimeError(f"Source changed during export: {src}; stop/wait and rerun")
                    meta = tar.gettarinfo(str(src), arcname=rel.as_posix())
                    meta.uid = meta.gid = 0
                    meta.uname = meta.gname = ""
                    with open(src, "rb") as f:
                        reader = HashingReader(f)
                        tar.addfile(meta, reader)
                        digest = reader.h.hexdigest()
                    if src.stat().st_size != size or src.stat().st_mtime_ns != current.st_mtime_ns:
                        raise RuntimeError(f"Source changed while archiving: {src}")
                    manifest.append(dict(path=rel.as_posix(), size=size, sha256=digest, storage="release", tag=tag))
                    if i % 250 == 0:
                        print(f"[ARCHIVE] {i:,}/{len(release):,} files processed", flush=True)
        finally:
            writer.close()
        (WORK / "thermoo_release_parts.sha256").write_text(
            "".join(f"{digest}  {name}\n" for name, digest, size in writer.parts), encoding="utf-8")
        print(f"[RELEASE] Uploaded {len(writer.parts)} archive parts to {tag}", flush=True)
    (WORK / "thermoo_export_manifest.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in sorted(manifest, key=lambda x:x["path"])), encoding="utf-8")
    metadata_paths = [Path("thermoo_export_manifest.jsonl")]
    if release:
        metadata_paths.append(Path("thermoo_release_parts.sha256"))
    upload_git(WORK, [(p, (WORK / p).stat().st_size) for p in metadata_paths], "Add TherMOO export SHA-256 manifest", branch)
    print("\n[SUCCESS] Repository:", f"https://github.com/{REPO}")
    if release:
        print("[SUCCESS] Raw data Release:", f"https://github.com/{REPO}/releases/tag/{tag}")
    print("[SUCCESS] Local project directories were not modified.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (subprocess.CalledProcessError, OSError, RuntimeError, tarfile.TarError) as exc:
        print(f"[FAILED] {exc}", file=sys.stderr, flush=True)
        raise
