"""
Launcher: start the local server and open the browser.

Run directly:      python launcher.py
Build an exe:      see build.bat (uses PyInstaller)
"""

import time
import threading
import webbrowser

import server


def main():
    cfg = server.CONFIG
    t = threading.Thread(target=server.run, kwargs={"open_browser": False}, daemon=True)
    t.start()

    url = "http://%s:%d" % (cfg["host"], cfg["port"])
    time.sleep(0.8)
    webbrowser.open(url)

    # Keep the process (and server) alive.
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
