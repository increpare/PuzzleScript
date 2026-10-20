import errno
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket


class PreviewServer(ThreadingHTTPServer):
    # Avoid sharing an occupied port on Windows.
    allow_reuse_address = False


def main():
    handler = partial(SimpleHTTPRequestHandler, directory=str(Path(__file__).resolve().parent))
    try:
        server = PreviewServer(("0.0.0.0", 8000), handler)
    except OSError as error:
        if error.errno not in (errno.EADDRINUSE, errno.EACCES):
            raise
        # Port zero asks the operating system to allocate a free port.
        server = PreviewServer(("0.0.0.0", 0), handler)

    with server:
        port = server.server_address[1]
        print(f"Local: http://127.0.0.1:{port}/", flush=True)
        try:
            addresses = sorted({
                result[4][0] for result in socket.getaddrinfo(
                    socket.gethostname(), None, socket.AF_INET, socket.SOCK_STREAM
                )
            })
        except OSError:
            addresses = []
        for address in addresses:
            if not address.startswith("127."):
                print(f"LAN:   http://{address}:{port}/", flush=True)
        print("Press Ctrl+C to stop.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
