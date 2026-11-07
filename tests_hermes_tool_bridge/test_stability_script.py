from scripts.apply_dsk_stability import git_blob_sha, PatchMismatch, transform, MARKER
import pytest


def sample():
    return '''import time
import sys
import subprocess
import os
class E(Exception):
    pass
class APIError(Exception):
    pass
class NetworkError(Exception):
    pass
class Stub:
    def _refresh_cookies(self):
        script_path = 'foo'
        subprocess.run([sys.executable, script_path], check=True)
    def _make_request(self):
        try:
            response = request(
                    timeout=None
            )
            if bad():
                raise APIError(f"Server error occurred: {response.text}", response.status_code)
            if worse():
                raise APIError(f"API request failed: {response.text}", response.status_code)
        except requests.exceptions.RequestException as e:
            raise NetworkError(f"Network error occurred: {str(e)}")
    def chat_completion(self):
        try:
            response = post(
                stream=True,
                timeout=None
            )
            if response.status_code != 200:
                error_text = next(response.iter_lines(), b'').decode('utf-8', 'ignore')
                raise APIError(f"API request failed: {error_text}", response.status_code)
            for chunk in response.iter_lines():
                try:
                    yield chunk
                except Exception:
                    raise APIError('oops')
        except requests.exceptions.RequestException as e:
            raise NetworkError(f"Network error occurred during streaming: {str(e)}")
    def _parse_chunk(self, json_str):
        if fail():
            raise APIError(f"Invalid JSON in response chunk: {json_str!r}")
        print(f"[DEEPSEEK_DEBUG] Unhandled chunk: {json_str!r}", file=sys.stderr)
'''


def test_patch_transform_compiles_and_closes_stream():
    result=transform(sample())
    compile(result,'patched_api.py','exec')
    assert MARKER in result
    assert 'finally:\n                response.close()' in result
    assert 'timeout=REQUEST_TIMEOUT' in result
    assert 'timeout=STREAM_TIMEOUT' in result
    assert 'timeout=COOKIE_REFRESH_TIMEOUT' in result
    assert '{response.text}' not in result
    assert "{json_str!r}" not in result


def test_patch_idempotence_refused():
    with pytest.raises(PatchMismatch):
        transform(transform(sample()))


def test_revision_hash_uses_git_blob_format():
    import hashlib
    assert git_blob_sha('hello\n') == hashlib.sha1(b'blob 6\x00hello\n').hexdigest()
