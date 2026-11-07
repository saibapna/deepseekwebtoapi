#!/usr/bin/env python3
"""Opt-in, revision-locked hotfix for the original DeepSeekAPI HTTP lifecycle.

The patch is intentionally separate from the Hermes bridge. It refuses to
modify any upstream revision other than the GitHub blob inspected during this
release. Run in a clean git branch and inspect the diff before deploying.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import os
import tempfile
from pathlib import Path

EXPECTED_GIT_BLOB = "d73ff3c0bbc23eda9a91b972f6781adbbf26db79"
MARKER = "HERMES_BRIDGE_DSK_STABILITY_V1"


class PatchMismatch(RuntimeError):
    pass


def git_blob_sha(text: str) -> str:
    data = text.encode("utf-8")
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def replace_exact(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise PatchMismatch(f"Expected exactly one anchor: {old[:80]!r}")
    return text.replace(old, new, 1)


def transform(text: str) -> str:
    if MARKER in text:
        raise PatchMismatch("Stability patch already installed")
    text = replace_exact(
        text, "import time\n", "import time\nimport os\n\n"
        f"# {MARKER}: bounded timeouts and response cleanup\n"
        "def _bounded_timeout(name: str, default: float) -> float:\n"
        "    value = float(os.environ.get(name, str(default)))\n"
        "    if not (1 <= value <= 600):\n"
        "        raise ValueError(f'{name} must be between 1 and 600 seconds')\n"
        "    return value\n\n"
        "REQUEST_TIMEOUT = _bounded_timeout('DEEPSEEK_REQUEST_TIMEOUT', 60)\n"
        "STREAM_TIMEOUT = _bounded_timeout('DEEPSEEK_STREAM_TIMEOUT', 180)\n"
        "COOKIE_REFRESH_TIMEOUT = _bounded_timeout('DEEPSEEK_COOKIE_REFRESH_TIMEOUT', 90)\n",
    )
    changes = [
        ("subprocess.run([sys.executable, script_path], check=True)",
         "subprocess.run([sys.executable, script_path], check=True, timeout=COOKIE_REFRESH_TIMEOUT)"),
        ("                    timeout=None\n", "                    timeout=REQUEST_TIMEOUT\n"),
        ("                stream=True,\n                timeout=None\n", "                stream=True,\n                timeout=STREAM_TIMEOUT\n"),
        ('raise APIError(f"Server error occurred: {response.text}", response.status_code)',
         'raise APIError("Upstream service error", response.status_code)'),
        ('raise APIError(f"API request failed: {response.text}", response.status_code)',
         'raise APIError("Upstream request rejected", response.status_code)'),
        ('raise APIError(f"API request failed: {error_text}", response.status_code)',
         'raise APIError("Upstream streaming request rejected", response.status_code)'),
        ("                error_text = next(response.iter_lines(), b'').decode('utf-8', 'ignore')\n", ""),
        ('raise APIError(f"Invalid JSON in response chunk: {json_str!r}")',
         'raise APIError("Invalid JSON in streaming response")'),
        ('print(f"[DEEPSEEK_DEBUG] Unhandled chunk: {json_str!r}", file=sys.stderr)',
         'if os.getenv("DEEPSEEK_DEBUG_STREAM", "false").lower() == "true":\n'
         '            print("[DEEPSEEK_DEBUG] Unhandled streaming event", file=sys.stderr)'),
        ('raise NetworkError(f"Network error occurred during streaming: {str(e)}")',
         'raise NetworkError("Network error occurred during streaming") from e'),
        ('raise NetworkError(f"Network error occurred: {str(e)}")',
         'raise NetworkError("Network error occurred") from e'),
    ]
    for old, new in changes:
        text = replace_exact(text, old, new)
    method_at = text.index("    def chat_completion(")
    start = text.index("            if response.status_code != 200:\n", method_at)
    end = text.index("        except requests.exceptions.RequestException as e:\n", start)
    original = text[start:end]
    if original.count("for chunk in response.iter_lines():") != 1:
        raise PatchMismatch("Cannot locate unique SSE iterator")
    indented = "".join("    " + line if line.strip() else line for line in original.splitlines(keepends=True))
    wrapped = "            try:\n" + indented + "            finally:\n                response.close()\n\n"
    text = text[:start] + wrapped + text[end:]
    compile(text, "dsk/api.py", "exec")
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".", help="Repository root")
    parser.add_argument("--apply", action="store_true", help="Write modified dsk/api.py (default is check only)")
    parser.add_argument("--diff", action="store_true", help="Print proposed unified diff")
    args = parser.parse_args()
    target = Path(args.root).resolve() / "dsk" / "api.py"
    original = target.read_text(encoding="utf-8")
    current = git_blob_sha(original)
    if current != EXPECTED_GIT_BLOB:
        raise SystemExit(f"Refusing to patch unknown dsk/api.py revision: {current}; expected {EXPECTED_GIT_BLOB}")
    updated = transform(original)
    if args.diff:
        print("".join(difflib.unified_diff(original.splitlines(keepends=True), updated.splitlines(keepends=True), fromfile="a/dsk/api.py", tofile="b/dsk/api.py")))
    if args.apply:
        fd, filename = tempfile.mkstemp(prefix=".api.py.tmp-", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(updated)
            os.chmod(filename, target.stat().st_mode)
            os.replace(filename, target)
        finally:
            if os.path.exists(filename):
                os.unlink(filename)
        print("Patched dsk/api.py. Run git diff --check and inspect git diff.")
    else:
        print("Revision and patch anchors verified. No files changed; use --apply to write.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
