"""A range-serving HTTP server, for testing the parallel downloader.

Python's own http.server does not do Range, and the whole point of the test is
that ranges are assembled correctly — so this serves a deterministic file and
honours (or, on request, refuses) the header.

    ranges.py <port> <size> [--no-ranges] [--drop-after N]

Prints the port on stdout once listening. The body is a repeating pattern
derived from the byte offset, so a slice written to the wrong place is visible
rather than being plausible zeroes.
"""
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SIZE = int(sys.argv[2])
NO_RANGES = "--no-ranges" in sys.argv

# Built once. Generating it per request in a Python loop takes longer than the
# download being measured, which made the tests look like the downloader was
# slow when it was the fixture.
BODY = bytes(((i * 7 + 11) & 0xFF) for i in range(SIZE))


def body(start, end):
    return BODY[start : end + 1]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def do_GET(self):
        rng = self.headers.get("Range")
        if rng and not NO_RANGES:
            spec = rng.split("=", 1)[1]
            first, _, last = spec.partition("-")
            start = int(first)
            end = int(last) if last else SIZE - 1
            end = min(end, SIZE - 1)
            data = body(start, end)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{SIZE}")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(data)
            return

        data = body(0, SIZE - 1)
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        if not NO_RANGES:
            self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    # Threading, or eight parallel range requests queue behind each other and
    # the test measures the server instead of the client.
    server = ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler)
    print(server.server_address[1], flush=True)
    server.serve_forever()
