"""Serve the built FEMCubes documentation without browser caching.

Used by ``scripts/start_doc_server.sh``; run directly as::

    python scripts/doc_server.py <html folder> <port> [host]

Every request is answered in full and marked as not cacheable, so a page rebuilt
with ``scripts/build_docs.sh`` shows on the next reload.
"""

import functools
import http.server
import sys


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    """Static file handler that disables browser caching."""

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self):
        # Ignore conditional requests, so a rebuilt page is always sent in full
        for header in ("If-Modified-Since", "If-None-Match"):
            if header in self.headers:
                del self.headers[header]
        super().do_GET()


def main():
    if len(sys.argv) not in (3, 4):
        sys.exit("Usage: python scripts/doc_server.py <html folder> <port> [host]")
    directory, port = sys.argv[1], int(sys.argv[2])
    host = sys.argv[3] if len(sys.argv) == 4 else "127.0.0.1"
    handler = functools.partial(NoCacheHandler, directory=directory)
    http.server.ThreadingHTTPServer((host, port), handler).serve_forever()


if __name__ == "__main__":
    main()
