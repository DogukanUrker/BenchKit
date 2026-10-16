"""Run controls the front-ends flip while the engine works."""

from __future__ import annotations

import threading


class RunControls:
    """Thread-safe pause / skip / stop / force-unload switches for the UI."""

    def __init__(self, force_unload: bool = False) -> None:
        self._stop = threading.Event()
        self._skip = threading.Event()
        self._cancel_request = threading.Event()
        self._running = threading.Event()
        self._running.set()
        self._force_unload = force_unload

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    @property
    def cancel_event(self) -> threading.Event:
        """Event passed to clients to kill every active model request."""
        return self._cancel_request

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_request.is_set()

    @property
    def paused(self) -> bool:
        return not self._running.is_set()

    @property
    def force_unload(self) -> bool:
        return self._force_unload

    def toggle_force_unload(self) -> bool:
        """Flip force-unload and return the new state."""
        self._force_unload = not self._force_unload
        return self._force_unload

    def stop(self) -> None:
        self._stop.set()
        self._cancel_request.set()
        self._running.set()

    def skip_job(self) -> None:
        self._skip.set()
        self._cancel_request.set()
        self._running.set()

    @property
    def skip_requested(self) -> bool:
        return self._skip.is_set()

    def pause(self) -> None:
        self._running.clear()

    def resume(self) -> None:
        self._running.set()

    def toggle_pause(self) -> bool:
        """Flip the paused state and return ``True`` when now paused."""
        if self.paused:
            self.resume()
            return False
        self.pause()
        return True

    def take_skip(self) -> bool:
        """Consume a pending skip request."""
        if self._skip.is_set():
            self._skip.clear()
            self._cancel_request.clear()
            return True
        return False

    def cancel_active(self) -> None:
        """Cancel in-flight requests while unwinding an internal job failure."""
        self._cancel_request.set()
        self._running.set()

    def wait_while_paused(self) -> None:
        self._running.wait()
