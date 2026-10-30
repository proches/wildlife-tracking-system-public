"""Commit and push every change of the project to GitHub (origin main).

Run by:
- the Claude Code Stop hook (.claude/settings.json) at the end of each Claude turn
  (--reason claude-session): every change Claude makes reaches GitHub;
- task MeliaGitAutoSync every 30 min (--reason scheduled): detections
  (logs/events.csv, detections.csv), learned state, an accepted model ...

The user asked for it on 26/10/2026 ("commit every change ... even when anything
happens").  What .gitignore keeps out (the .env password, videos, training data)
stays out.  The script never prompts: no editor, Git Credential Manager in
non-interactive mode.  It refuses a file that contains the camera password from
.env or is over 95 MB, and it does nothing when nothing changed.

    .venv313\\Scripts\\python.exe tools\\git_autosync.py [--reason X] [--dry-run]
Log: logs/autosync.log (not in git).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "autosync.log"
LOCK = ROOT / "logs" / "autosync.lock"
MAX_BYTES = 95 * 1024 * 1024          # GitHub refuses files over 100 MB
NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")


def git(*args, timeout=300, stdin=None):
    env = dict(os.environ, GCM_INTERACTIVE="never", GIT_TERMINAL_PROMPT="0", GIT_EDITOR="true")
    helper = ["-c", "credential.helper=manager"] if os.name == "nt" else []    # Linux: gh auth setup-git
    return subprocess.run(["git", *helper, *args], cwd=ROOT, input=stdin,
                          capture_output=True, text=True, timeout=timeout, env=env, creationflags=NO_WINDOW)


def secrets():
    """Values of the secret keys in .env (never printed or logged)."""
    out = []
    try:
        for line in open(ROOT / ".env", encoding="utf-8"):
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                value = value.strip().strip('"').strip("'")
                if any(w in key.upper() for w in ("PASSWORD", "KEY", "TOKEN", "SECRET")) and len(value) >= 4:
                    out.append(value.encode())
    except OSError:
        pass
    return out


def unsafe(paths):
    """-> {path: reason} for staged files that must not be pushed."""
    bad, keys = {}, secrets()
    for p in paths:
        full = ROOT / p
        if not full.is_file():
            continue                                      # deleted / renamed away
        if full.stat().st_size > MAX_BYTES:
            bad[p] = "over 95 MB"
            continue
        try:
            data = full.read_bytes()
        except OSError:
            continue
        if any(k in data for k in keys):
            bad[p] = "contains a secret from .env"
    return bad


def summary(paths):
    """Short subject line: what kind of change this is."""
    added = 0
    numstat = git("diff", "--cached", "--numstat", "--", "logs/events.csv").stdout.split()
    if numstat and numstat[0].isdigit():
        added = int(numstat[0])
    parts = []
    if added:
        parts.append(f"{added} new detection event{'s' if added != 1 else ''}")
    if any(p.startswith("models/") for p in paths):
        parts.append("model")
    code = [p for p in paths if p.endswith((".py", ".bat", ".vbs", ".md", ".json", ".txt", ".xml")) and not p.startswith("logs/")]
    if code:
        parts.append(f"{len(code)} code/config file{'s' if len(code) != 1 else ''}")
    return ", ".join(parts) or f"{len(paths)} file{'s' if len(paths) != 1 else ''}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reason", default="manual")
    ap.add_argument("--dry-run", action="store_true", help="show what would be committed, change nothing")
    a = ap.parse_args()
    try:                                                   # one run at a time (hook + task may meet)
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        if time.time() - LOCK.stat().st_mtime < 900:
            return 0
        LOCK.unlink(missing_ok=True)                       # left by a run that was killed
        return main()
    try:
        status = git("status", "--porcelain").stdout.strip()
        ahead = git("rev-list", "--count", "origin/main..HEAD").stdout.strip()
        if not status and ahead in ("", "0"):
            return 0                                       # nothing to do: stay silent
        commit = None
        if status:
            if a.dry_run:
                print(status)
                return 0
            git("add", "-A")
            paths = [p for p in git("diff", "--cached", "--name-only").stdout.splitlines() if p]
            bad = unsafe(paths)
            for p, why in bad.items():
                git("reset", "-q", "--", p)
                log(f"NOT committed: {p} ({why})")
            paths = [p for p in paths if p not in bad]
            if paths:
                body = "\n".join(f"  {p}" for p in paths[:40]) + ("\n  ..." if len(paths) > 40 else "")
                subject = summary(paths)
                msg = f"auto-sync: {subject}\n\nReason: {a.reason}. Files:\n{body}\n"
                if a.reason == "claude-session":
                    msg += "\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
                r = git("commit", "-q", "-F", "-", stdin=msg)
                if r.returncode != 0:
                    log(f"commit failed: {(r.stderr or r.stdout).strip()[:300]}")
                    return 1
                commit = git("rev-parse", "--short", "HEAD").stdout.strip()
                log(f"committed {commit} ({a.reason}): {subject}")
        r = git("push", "-q", "origin", "main", timeout=900)
        if r.returncode != 0:
            log(f"push failed (will retry next run): {(r.stderr or r.stdout).strip()[:300]}")
            return 1
        log(f"pushed to GitHub ({a.reason})")
        try:                       # the public code copy (no camera pictures, no logs)
            sys.path.insert(0, str(ROOT))
            from tools.publish_public import main as publish_public
            publish_public([])
        except Exception as exc:   # the private copy is safe; the public one retries next run
            log(f"public copy not updated: {type(exc).__name__}: {exc}")
        if commit and a.reason == "claude-session":
            print(json.dumps({"systemMessage": f"GitHub: changes committed and pushed ({commit})"}))
        return 0
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
