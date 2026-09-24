"""Start the backend: `python -m rpi_agents.cloud.app.server api|worker|both [--check-config]`.

The API and the worker are the same image with different commands (architecture, section 27): the API scales to at
most one replica, the worker scales with the queue. A configuration error is reported by name and stops the process
with exit code 2, before anything is served. `--check-config` validates the environment and the manifest and exits, for
a deployment pipeline. SIGTERM stops the worker loop and lets the API finish its requests.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading

from rpi_agents.cloud.app.auth import install_log_redaction
from rpi_agents.cloud.app.backend_config import BackendConfigError, build, from_env
from rpi_agents.cloud.app.keyvault import KeyVaultError, resolve

log = logging.getLogger("snn_backend.server")


def main(argv: list[str] | None = None, environ=None) -> int:
    parser = argparse.ArgumentParser(description="SNN Agent backend")
    parser.add_argument("role", choices=["api", "worker", "both"])
    parser.add_argument("--check-config", action="store_true", help="validate the configuration and exit")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, stream=sys.stderr, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    install_log_redaction()
    # The Azure SDK logs every storage request with its headers at INFO: hundreds of lines a second that bury the
    # service's own events (the Authorization header is redacted by the SDK, the volume is the problem).
    for noisy in ("azure", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        backend = build(from_env(resolve(os.environ if environ is None else environ)))
    except (BackendConfigError, KeyVaultError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    if args.check_config:
        print("configuration ok")
        return 0

    stop = threading.Event()
    worker = None
    if args.role in ("worker", "both"):
        worker = threading.Thread(target=backend.worker.run_forever, args=(stop, 1.0), name="snn-worker")
        worker.start()
    try:
        if args.role in ("api", "both"):
            import uvicorn

            from rpi_agents.cloud.app.api import create_app

            app = create_app(backend.services, backend.operator, backend.api_settings)
            # Our own rate limiting and logging read X-Forwarded-For with an explicit trust count, so uvicorn's
            # proxy-header rewriting is off; the server banner and access log (which can carry paths) are off too.
            uvicorn.run(app, host=backend.config.host, port=backend.config.port, proxy_headers=False, server_header=False, access_log=False, log_config=None)
        else:
            signal.signal(signal.SIGTERM, lambda *_: stop.set())
            signal.signal(signal.SIGINT, lambda *_: stop.set())
            stop.wait()
    finally:
        stop.set()
        if worker is not None:
            worker.join(timeout=30)
        log.info("stopped; drained=%s", backend.worker.drained())
    return 0


if __name__ == "__main__":
    sys.exit(main())
