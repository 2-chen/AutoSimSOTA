"""Exercise the real CLI/MCP discovery against a local fake model, with zero API spend."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from autosim.research.agent_roles import role_profile
from autosim.research.agent_runtime import _role_cli_args
from autosim.research.common import atomic_json


class FixtureModel(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get('Content-Length', 0)))
        payload = {'input_tokens': 1} if self.path.endswith('count_tokens') else {
            'id': 'msg_fixture', 'type': 'message', 'role': 'assistant', 'model': 'fixture',
            'content': [{'type': 'text', 'text': '{}'}], 'stop_reason': 'end_turn',
            'stop_sequence': None, 'usage': {'input_tokens': 1, 'output_tokens': 1}}
        body = json.dumps(payload).encode()
        self.send_response(200); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', default=shutil.which('claude'))
    options = parser.parse_args()
    if not options.cli:
        parser.error('claude not on PATH; pass --cli /path/to/claude')
    cli = str(Path(options.cli).absolute())
    with tempfile.TemporaryDirectory(prefix='autosim-discovery-') as temporary:
        root = Path(temporary); workspace = root/'checkout'; workspace.mkdir()
        atomic_json(root/'workspace_snapshot.json', {'schema_version': 2,
            'destination': str(workspace), 'source': str(workspace), 'isolated': True, 'source_entries': []})
        (root/'mcp-home').mkdir(); (root/'mcp-tmp').mkdir()
        server = ThreadingHTTPServer(('127.0.0.1', 0), FixtureModel)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        profile = role_profile('init', read_only=True)
        args = _role_cli_args(profile, workspace=workspace, output=root, python=sys.executable,
                             model='claude-sonnet-4-5', max_budget_usd=.1)
        env = {'PATH': os.pathsep.join((str(Path(cli).parent), str(Path(sys.executable).parent),
                                        '/usr/local/bin', '/usr/bin', '/bin')),
               'HOME': str(root/'agent-home'), 'TMPDIR': str(root/'mcp-tmp'),
               'ANTHROPIC_BASE_URL': f'http://127.0.0.1:{server.server_port}',
               'ANTHROPIC_API_KEY': 'local-fixture-not-a-real-key',
               'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1'}
        try:
            result = subprocess.run([cli, *args, '--debug-file', str(root/'debug.log')],
                input='Return {} without using tools.', text=True, capture_output=True,
                cwd=workspace, env=env, timeout=45)
            print('CLI_exit', result.returncode)
            discovered = []
            for line in result.stdout.splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get('type') == 'system' and row.get('subtype') == 'init':
                    discovered = row.get('tools') or []
                    print('discovered', discovered)
            debug = root/'debug.log'
            if debug.is_file():
                for line in debug.read_text().splitlines():
                    if any(term in line for term in ('inspect_native_environment', 'schema', 'Error', 'error')):
                        print('diagnostic', line[:900])
            missing = sorted(set(profile.mcp_tools)-set(discovered))
            print('missing_configured_tools', missing)
            if result.returncode or missing:
                raise SystemExit(1)
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=2)


if __name__ == '__main__':
    main()
