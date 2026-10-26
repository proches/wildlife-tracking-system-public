"""Keep the PUBLIC code copy on GitHub up to date (proches/wildlife-tracking-system-public).

The user chose on 26/10/2026: code public, camera pictures private.  The private
repository (origin, proches/wildlife-tracking-system) keeps everything.  This copies
the files git tracks, MINUS the camera pictures and the system's own records, into a
local mirror and pushes it:
* left out: training_data/ (hand-checked camera crops), logs/ (detection history,
  learned clutter, camera settings), every picture or video (ptz_presets.jpg ...);
* public commits use the GitHub noreply address, never the private email.
Called by tools/git_autosync.py after each successful push; can also run alone:

    .venv313\\Scripts\\python.exe tools\\publish_public.py [--dry-run]
Mirror: %LOCALAPPDATA%\\melia_public_mirror (re-created when missing).  Log: logs/autosync.log.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIRROR = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "melia_public_mirror"
REMOTE = "https://github.com/proches/wildlife-tracking-system-public.git"
AUTHOR_NAME, AUTHOR_EMAIL = "proches", "83775689+proches@users.noreply.github.com"
EXCLUDE_DIRS = ("training_data/", "logs/")
EXCLUDE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".mp4", ".avi", ".mov", ".mkv")
LOG = ROOT / "logs" / "autosync.log"
NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  public: {msg}\n")


def git(cwd, *args, timeout=300, stdin=None):
    env = dict(os.environ, GCM_INTERACTIVE="never", GIT_TERMINAL_PROMPT="0", GIT_EDITOR="true",
               GIT_AUTHOR_NAME=AUTHOR_NAME, GIT_AUTHOR_EMAIL=AUTHOR_EMAIL,
               GIT_COMMITTER_NAME=AUTHOR_NAME, GIT_COMMITTER_EMAIL=AUTHOR_EMAIL)
    return subprocess.run(["git", "-c", "credential.helper=manager", *args], cwd=cwd, input=stdin,
                          capture_output=True, text=True, timeout=timeout, env=env, creationflags=NO_WINDOW)


def is_public(rel):
    rel = rel.replace("\\", "/")
    return not rel.startswith(EXCLUDE_DIRS) and not rel.lower().endswith(EXCLUDE_EXT)


def public_files():
    return [p for p in git(ROOT, "ls-files").stdout.splitlines() if p and is_public(p)]


def ensure_mirror():
    if (MIRROR / ".git").exists():
        return True
    MIRROR.mkdir(parents=True, exist_ok=True)
    git(MIRROR, "init", "-q", "-b", "main")
    git(MIRROR, "remote", "add", "origin", REMOTE)
    if git(MIRROR, "fetch", "-q", "origin", "main", timeout=900).returncode == 0:
        git(MIRROR, "reset", "-q", "origin/main")          # continue the public history
    return True


def sync(dry_run=False):
    files = public_files()
    want = set(files)
    changed = []
    for rel in files:
        src, dst = ROOT / rel, MIRROR / rel
        if not src.is_file():
            continue
        s = src.stat()
        if dst.is_file():
            d = dst.stat()
            if d.st_size == s.st_size and int(d.st_mtime) == int(s.st_mtime):
                continue
        changed.append(rel)
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    removed = []
    for path in MIRROR.rglob("*"):
        rel = path.relative_to(MIRROR).as_posix()
        if rel.startswith(".git/") or rel == ".git" or not path.is_file():
            continue
        if rel not in want:
            removed.append(rel)
            if not dry_run:
                path.unlink()
    if dry_run:
        return changed, removed, None
    git(MIRROR, "add", "-A")
    if not git(MIRROR, "status", "--porcelain").stdout.strip():
        return changed, removed, None
    src_sha = git(ROOT, "rev-parse", "--short", "HEAD").stdout.strip()
    names = sorted(set(changed) | set(removed))
    body = "\n".join(f"  {p}" for p in names[:40]) + ("\n  ..." if len(names) > 40 else "")
    msg = (f"Public code update: {len(names)} file{'s' if len(names) != 1 else ''}\n\n"
           f"From the private repository, commit {src_sha}.  Camera pictures and logs are not published.\n"
           f"Files:\n{body}\n")
    r = git(MIRROR, "commit", "-q", "-F", "-", stdin=msg)
    if r.returncode != 0:
        raise RuntimeError(f"commit failed: {(r.stderr or r.stdout).strip()[:300]}")
    return changed, removed, git(MIRROR, "rev-parse", "--short", "HEAD").stdout.strip()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    ensure_mirror()
    changed, removed, commit = sync(a.dry_run)
    if a.dry_run:
        print(f"would copy {len(changed)} and remove {len(removed)} files")
        for p in (changed + removed)[:60]:
            print("  ", p)
        return 0
    ahead = git(MIRROR, "rev-list", "--count", "origin/main..HEAD").stdout.strip()
    if commit is None and ahead == "0":
        return 0
    r = git(MIRROR, "push", "-q", "origin", "main", timeout=1800)
    if r.returncode != 0:
        log(f"push failed (will retry next run): {(r.stderr or r.stdout).strip()[:300]}")
        return 1
    log(f"pushed {commit or 'pending commits'} ({len(changed)} copied, {len(removed)} removed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
