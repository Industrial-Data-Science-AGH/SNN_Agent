"""Shared test doubles: a scriptable local HTTP server that records what it receives."""

import base64
import json
import socketserver
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):  # clients that hang up early are expected here
        pass


class FakeHttp:
    """Answers from `script` (a list of (status, payload, headers)); records (method, path, headers, body).

    Recorded header names are lower-case."""

    def __init__(self):
        self.requests, self.script, self.delay = [], [], 0.0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _serve(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length).decode() if length else ""
                headers = {name.lower(): value for name, value in self.headers.items()}  # HTTP names are case-insensitive
                outer.requests.append((self.command, self.path, headers, body))
                if outer.delay:
                    threading.Event().wait(outer.delay)
                status, payload, extra = outer.script.pop(0) if outer.script else (200, {"ok": True}, {})
                raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(raw)))
                for name, value in extra.items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(raw)

            do_GET = do_POST = _serve

            def log_message(self, *args):
                pass

        self.httpd = QuietServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class FakeSmtp:
    """A minimal SMTP server whose behaviour tests configure. Records accepted messages.

    data_reply: a reply line for the message body, or "DROP" (close without answering) or "HANG" (never answer).
    data_command_reply: the reply to the DATA command itself (default 354).
    """

    def __init__(self, *, greeting="220 fake ESMTP", credentials=None):
        self.greeting, self.credentials = greeting, credentials
        self.rcpt_reply, self.data_reply, self.data_command_reply = {}, "250 queued", "354 go ahead"
        self.hang_at = None
        self.messages, self.commands = [], []
        outer = self

        class Handler(socketserver.StreamRequestHandler):
            def say(self, line):
                self.wfile.write(line.encode() + b"\r\n")
                self.wfile.flush()

            def handle(self):
                self.say(outer.greeting)
                sender, recipients = None, []
                while True:
                    line = self.rfile.readline()
                    if not line:
                        return
                    text = line.decode(errors="replace").strip()
                    verb = text.split(" ")[0].upper()
                    outer.commands.append(verb)
                    if outer.hang_at == verb:
                        time.sleep(5)
                        return
                    if verb == "EHLO":
                        self.say("250-fake")
                        self.say("250 AUTH PLAIN" if outer.credentials else "250 OK")
                    elif verb == "AUTH":
                        _, user, password = base64.b64decode(text.split(" ")[2]).decode().split("\x00")
                        ok = (user, password) == outer.credentials
                        self.say("235 authenticated" if ok else "535 authentication failed")
                    elif verb == "MAIL":
                        sender, recipients = text, []
                        self.say("250 OK")
                    elif verb == "RCPT":
                        address = text.split("<", 1)[1].rstrip(">")
                        reply = outer.rcpt_reply.get(address, "250 OK")
                        if reply.startswith("250"):
                            recipients.append(address)
                        self.say(reply)
                    elif verb == "DATA":
                        self.say(outer.data_command_reply)
                        if not outer.data_command_reply.startswith("354"):
                            continue
                        body = b""
                        while not body.endswith(b"\r\n.\r\n"):
                            chunk = self.rfile.readline()
                            if not chunk:
                                return
                            body += chunk
                        if outer.data_reply == "DROP":
                            return
                        if outer.data_reply == "HANG":
                            time.sleep(5)
                            return
                        if outer.data_reply.startswith("250"):
                            outer.messages.append({"sender": sender, "recipients": recipients, "data": body[:-5]})
                        self.say(outer.data_reply)
                    elif verb == "QUIT":
                        self.say("221 bye")
                        return
                    else:
                        self.say("250 OK")

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True
            block_on_close = False

        self.server = Server(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def make_jpeg(width=640, height=480, *, precision=8, components=3, extra=b"", scan=b"\x01\x02\x03", eoi=True):
    """A structurally valid JPEG (not decodable, which is not needed): SOI, APP0, SOF0, SOS, data, EOI."""
    app0 = b"\xff\xe0" + (16).to_bytes(2, "big") + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof_body = bytes([precision]) + height.to_bytes(2, "big") + width.to_bytes(2, "big") + bytes([components])
    sof_body += b"".join(bytes([i + 1, 0x11, 0]) for i in range(components))
    sof = b"\xff\xc0" + (len(sof_body) + 2).to_bytes(2, "big") + sof_body
    sos_body = bytes([components]) + b"".join(bytes([i + 1, 0]) for i in range(components)) + b"\x00\x3f\x00"
    sos = b"\xff\xda" + (len(sos_body) + 2).to_bytes(2, "big") + sos_body
    return b"\xff\xd8" + app0 + extra + sof + sos + scan + (b"\xff\xd9" if eoi else b"")


class FakeVision:
    """Scripted vision provider: each call takes the next item (an Observation, or an exception to raise)."""

    deployment, prompt_version = "gpt-fake", "fake-v1"

    def __init__(self, script=None, *, block=None):
        self.script, self.calls, self.block, self.entered = list(script or []), [], block, threading.Event()

    def analyze(self, jpeg):
        self.calls.append(jpeg)
        self.entered.set()
        if self.block is not None:
            assert self.block.wait(10), "the test never released the vision call"
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeNotifier:
    def __init__(self, script=None):
        self.sent, self.script = [], list(script or [])

    def send(self, notification):
        self.sent.append(notification)
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, BaseException):
                raise item
        return f"<event-{notification.event_id}@fake>"
