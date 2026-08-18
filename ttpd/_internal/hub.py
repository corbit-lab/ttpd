"""Locate heavy artifacts (model weights, benchmark data) locally or on the Hub.

Weights and benchmark data are NOT tracked in git. They live in two Hugging Face
repos and are pulled on demand:

    weights     model   repo -> $TTPD_WEIGHTS_REPO  (default below)
    benchmarks  dataset repo -> $TTPD_BENCH_REPO

Call sites keep computing the same local paths they always did; they just wrap
the result once:

    from ttpd.hub import ensure_local
    ckpt = ensure_local(ckpt)          # local file if present, else downloaded

If the file exists on disk it is returned untouched and nothing is downloaded,
so a machine that already has the full tree behaves exactly as before.

This module is stdlib-only at import time; `huggingface_hub` is imported lazily
inside the download path, so `--plan` and classification work without it.

    python3 -m ttpd._internal.hub --plan            # show local -> remote mapping
    python3 -m ttpd._internal.hub --check           # verify mapping has no collisions
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = "artifacts"


def _repo_root() -> str:
    """Where the artifacts/ tree lives.

    Normally the checkout one level above this package. When ttpd is pip-installed
    that parent is site-packages, which has no artifacts/ and is not writable, so
    fall back to $TTPD_ROOT and then to the current directory. Downloads are
    unaffected either way -- they resolve through the huggingface_hub cache.
    """
    env = os.environ.get("TTPD_ROOT")
    if env:
        return os.path.abspath(env)
    parent = os.path.dirname(os.path.dirname(HERE))
    if os.path.isdir(os.path.join(parent, ARTIFACT_DIR)):
        return parent
    if os.path.isdir(os.path.join(os.getcwd(), ARTIFACT_DIR)):
        return os.path.abspath(os.getcwd())
    return parent


REPO_ROOT = _repo_root()

WEIGHTS_REPO = os.environ.get("TTPD_WEIGHTS_REPO", "Murjani/ttpd-weights")
BENCH_REPO = os.environ.get("TTPD_BENCH_REPO", "Murjani/ttpd-benchmarks")
REVISION = os.environ.get("TTPD_HUB_REVISION", "main")

# Set TTPD_OFFLINE=1 to make a missing file a hard error instead of a download.
OFFLINE = os.environ.get("TTPD_OFFLINE", "").lower() in ("1", "true", "yes")

MODEL, DATASET = "model", "dataset"


# --------------------------------------------------------------------------
# classification: repo-relative local path -> (repo kind, path inside HF repo)
#
# artifacts/ mirrors the two Hub repos exactly, so the mapping is a prefix
# strip and the local and remote layouts cannot drift:
#
#     artifacts/weights/<p>  <->  model repo    <p>
#     artifacts/data/<p>     <->  dataset repo  <p>
# --------------------------------------------------------------------------

PREFIXES = {f"{ARTIFACT_DIR}/weights/": MODEL, f"{ARTIFACT_DIR}/data/": DATASET}


def classify(rel_path: str) -> tuple[str, str] | None:
    """Map a repo-relative path to (repo kind, path inside that HF repo).

    Returns None for files that stay in git.
    """
    rel = rel_path.replace(os.sep, "/").lstrip("./")
    for prefix, kind in PREFIXES.items():
        if rel.startswith(prefix):
            return kind, rel[len(prefix):]
    return None


def local_path(kind: str, remote: str) -> str:
    """Inverse of classify(): where `remote` belongs on disk."""
    sub = "weights" if kind == MODEL else "data"
    return os.path.join(REPO_ROOT, ARTIFACT_DIR, sub, remote)


def iter_artifacts(root: str = REPO_ROOT):
    """Yield (abs_path, rel_path, repo_kind, remote_path) for every Hub artifact."""
    skip = {".git", "__pycache__", ".venv", "venv", "_holding", "_trash"}
    for dirpath, dirnames, filenames in os.walk(os.path.join(root, ARTIFACT_DIR)):
        dirnames[:] = [d for d in dirnames if d not in skip]
        for name in sorted(filenames):
            abs_p = os.path.join(dirpath, name)
            rel = os.path.relpath(abs_p, root).replace(os.sep, "/")
            hit = classify(rel)
            if hit is not None:
                yield abs_p, rel, hit[0], hit[1]


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_plan(root: str = REPO_ROOT) -> dict:
    """Group artifacts by remote path and flag genuine collisions.

    Two local files mapping to one remote path is fine when their contents are
    identical (that is how the intentional duplicate trees are deduplicated);
    it is an error when they differ, because one would silently overwrite the
    other on upload.
    """
    by_remote: dict[tuple[str, str], list[str]] = {}
    for abs_p, rel, kind, remote in iter_artifacts(root):
        by_remote.setdefault((kind, remote), []).append(rel)

    uploads, duplicates, collisions = [], [], []
    for (kind, remote), rels in sorted(by_remote.items()):
        if len(rels) == 1:
            uploads.append((kind, remote, rels[0]))
            continue
        digests = {r: _sha256(os.path.join(root, r)) for r in rels}
        if len(set(digests.values())) == 1:
            uploads.append((kind, remote, rels[0]))
            duplicates.extend(rels[1:])
        else:
            collisions.append((kind, remote, digests))
    return {"uploads": uploads, "duplicates": duplicates, "collisions": collisions}


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------

def _download(kind: str, remote: str) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "huggingface_hub is required to fetch missing artifacts "
            "(pip install huggingface_hub), or place the file locally."
        ) from exc
    repo_id = WEIGHTS_REPO if kind == MODEL else BENCH_REPO
    return hf_hub_download(repo_id=repo_id, filename=remote, revision=REVISION,
                           repo_type=kind)


def ensure_local(path: str) -> str:
    """Return a readable local path for `path`, downloading from the Hub if needed.

    `path` is whatever the caller already computed -- absolute or relative to
    the repo root. An existing file is returned unchanged.
    """
    if path and os.path.exists(path):
        return path

    abs_p = path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
    if os.path.exists(abs_p):
        return abs_p

    rel = os.path.relpath(abs_p, REPO_ROOT).replace(os.sep, "/")
    hit = classify(rel)
    if hit is None:
        raise FileNotFoundError(
            f"{path} is missing and is not a known Hub artifact "
            f"(resolved repo-relative path: {rel})")
    if OFFLINE:
        raise FileNotFoundError(
            f"{path} is missing and TTPD_OFFLINE=1 forbids downloading it "
            f"(would fetch {hit[1]})")

    kind, remote = hit
    repo_id = WEIGHTS_REPO if kind == MODEL else BENCH_REPO
    print(f"[ttpd.hub] {rel} not found locally -> {repo_id}:{remote}",
          file=sys.stderr)
    return _download(kind, remote)


def ensure_bench_dir(variant: str) -> str:
    """Return a directory holding every instance file for `variant`.

    Used by runners that take a --bench-dir and glob it, where per-file
    resolution is not enough. Falls back to snapshotting the instance subset of
    the dataset repo.
    """
    if variant not in ("a280", "ttd300"):
        raise ValueError(f"unknown variant {variant!r}; expected a280 or ttd300")
    local = os.path.join(REPO_ROOT, ARTIFACT_DIR, "data", "instances", variant)
    pattern = {"a280": "bench_*.txt", "ttd300": "ttd300_*.txt"}[variant]
    import glob
    if glob.glob(os.path.join(local, pattern)):
        return local
    if OFFLINE:
        raise FileNotFoundError(
            f"no {variant} instances at {local} and TTPD_OFFLINE=1")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("huggingface_hub is required to fetch instances "
                           "(pip install huggingface_hub)") from exc
    print(f"[ttpd.hub] no {variant} instances locally -> {BENCH_REPO}",
          file=sys.stderr)
    root = snapshot_download(repo_id=BENCH_REPO, repo_type=DATASET,
                             revision=REVISION,
                             allow_patterns=[f"instances/{variant}/*"])
    return os.path.join(root, "instances", variant)


def weights_dir(variant: str, model: str, run: str = "") -> str:
    """Canonical directory for a (variant, model) weight family.

        weights_dir("a280", "gat", "benchmark-tuned")  -> artifacts/weights/a280/gat/bench-tuned

    Returns the path whether or not it exists yet; individual files inside it
    are resolved by ensure_local(), which fetches them on demand.
    """
    parts = [REPO_ROOT, ARTIFACT_DIR, "weights", variant, model]
    if run:
        parts.append(run)
    return os.path.join(*parts)


def instance(variant: str, name: str) -> str:
    """Resolve a single instance file by name, e.g. instance("a280", "bench_20_5.txt").

    Runners used to read instances from their own directory (the self-contained
    self-contained per-runner layout); they now go through here instead.
    """
    if not name.endswith(".txt"):
        name += ".txt"
    return ensure_local(os.path.join(REPO_ROOT, ARTIFACT_DIR, "data",
                                     "instances", variant, name))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", action="store_true",
                   help="print the local -> remote mapping and exit")
    p.add_argument("--check", action="store_true",
                   help="fail if two differing files map to one remote path")
    p.add_argument("--root", default=REPO_ROOT)
    args = p.parse_args()

    plan = build_plan(args.root)
    if args.plan:
        for kind, remote, rel in plan["uploads"]:
            print(f"{kind:7s}  {rel}\n         -> {remote}")
        print(f"\n{len(plan['uploads'])} artifacts, "
              f"{len(plan['duplicates'])} duplicates deduplicated")
    if plan["collisions"]:
        print(f"\n{len(plan['collisions'])} COLLISIONS "
              f"(differing files -> same remote path):", file=sys.stderr)
        for kind, remote, digests in plan["collisions"]:
            print(f"  {kind}:{remote}", file=sys.stderr)
            for rel, dig in digests.items():
                print(f"    {dig[:12]}  {rel}", file=sys.stderr)
        sys.exit(1)
    if args.check:
        print("no collisions")


if __name__ == "__main__":
    main()
