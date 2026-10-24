import importlib.util
import http.client
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
from pathlib import Path
import re
import threading
import unittest
from urllib.parse import urlencode

MODULE = Path(__file__).with_name('import_source.py')


class ImportSourceTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE.exists(), 'POST source receiver is missing')
        spec = importlib.util.spec_from_file_location('import_source', MODULE)
        self.receiver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.receiver)
        self.editor = Path(__file__).parents[1] / 'src/editor.html'

    def request(self, source='title Test', **changes):
        body = urlencode({'source': source}).encode()
        env = dict(REQUEST_METHOD='POST', CONTENT_TYPE='application/x-www-form-urlencoded',
                   CONTENT_LENGTH=str(len(body)), HTTP_ORIGIN='null')
        env.update(changes)
        return self.receiver.handle_request(env, io.BytesIO(body), self.editor)

    def imported(self, body):
        match = re.search(r'<script id="importedSource" type="application/json">(.*?)</script>', body, re.S)
        self.assertIsNotNone(match)
        return json.loads(match[1])

    def test_source_is_data_with_correct_base_and_no_storage(self):
        source = '</script><script>alert(1)</script>&\r\n雪\u2028$&</textarea>'
        status, headers, body = self.request(source)
        self.assertEqual(status, 200)
        self.assertEqual(self.imported(body), source)
        self.assertNotIn('</script><script>alert(1)', body)
        self.assertIn('<base href="/">', body)
        self.assertEqual(headers['Cache-Control'], 'no-store')
        self.assertEqual(headers['Referrer-Policy'], 'no-referrer')
        self.assertIn("script-src-attr 'none'", headers['Content-Security-Policy'])
        self.assertNotIn('Access-Control-Allow-Origin', headers)

    def test_large_source_and_hosted_origin(self):
        source = 'x' * (5 * 1024 * 1024)
        status, _, body = self.request(source, HTTP_ORIGIN='https://example.org')
        self.assertEqual(status, 200)
        self.assertEqual(self.imported(body), source)

    def test_unbuffered_request_reads_all_chunks(self):
        class ShortReads(io.BytesIO):
            def read(self, size):
                return super().read(min(size, 7))
        body = b'source=title+Chunked+read'
        env = dict(REQUEST_METHOD='POST', CONTENT_TYPE='application/x-www-form-urlencoded', CONTENT_LENGTH=str(len(body)))
        status, _, document = self.receiver.handle_request(env, ShortReads(body), self.editor)
        self.assertEqual(status, 200)
        self.assertEqual(self.imported(document), 'title Chunked read')

    def test_validation(self):
        cases = [({'REQUEST_METHOD': 'GET'}, 405),
                 ({'CONTENT_TYPE': 'text/plain'}, 415),
                 ({'CONTENT_LENGTH': 'bad'}, 400),
                 ({'CONTENT_LENGTH': '-1'}, 400),
                 ({'CONTENT_LENGTH': str(32 * 1024 * 1024 + 1)}, 413),
                 ({'CONTENT_LENGTH': '999'}, 400)]
        for changes, expected in cases:
            with self.subTest(changes=changes):
                status, headers, _ = self.request(**changes)
                self.assertEqual(status, expected)
                self.assertEqual(headers['Cache-Control'], 'no-store')

    def test_malformed_forms(self):
        for body in [b'other=test', b'source=a&source=b', b'source=%FF', b'source=%xx', b'source=', b'source=a&other=b']:
            with self.subTest(body=body):
                env = dict(REQUEST_METHOD='POST', CONTENT_TYPE='application/x-www-form-urlencoded',
                           CONTENT_LENGTH=str(len(body)))
                status, _, _ = self.receiver.handle_request(env, io.BytesIO(body), self.editor)
                self.assertEqual(status, 400)

    def test_missing_editor(self):
        status, _, body = self.receiver.handle_request(
            dict(REQUEST_METHOD='POST', CONTENT_TYPE='application/x-www-form-urlencoded', CONTENT_LENGTH='8'),
            io.BytesIO(b'source=x'), Path('/missing/editor.html'))
        self.assertEqual(status, 503)
        self.assertNotIn('/missing/', body)

    def test_http_adapter_statuses_and_large_post(self):
        self.receiver.EDITOR_PATH = self.editor
        receiver = self.receiver

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                receiver.handle_http(self)

            def log_message(self, *args):
                pass

        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            for body, expected in [(urlencode({'source': 'x' * (5 * 1024 * 1024)}), 200),
                                   ('source=%FF', 400)]:
                connection = http.client.HTTPConnection(*server.server_address, timeout=20)
                try:
                    connection.request('POST', '/editor.html', body,
                                       {'Content-Type': 'application/x-www-form-urlencoded', 'Origin': 'null'})
                    response = connection.getresponse()
                    self.assertEqual(response.status, expected)
                    self.assertEqual(response.getheader('Cache-Control'), 'no-store')
                    document = response.read().decode('utf-8')
                    if expected == 200:
                        self.assertEqual(self.imported(document), 'x' * (5 * 1024 * 1024))
                finally:
                    connection.close()
        finally:
            server.shutdown()
            thread.join()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
