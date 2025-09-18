import time
from contextlib import contextmanager
from typing import Iterator, Optional, Dict


class StepTimer:
    """Collects named timing measurements in seconds.

    Use with the time_step() context manager to record durations.
    """

    def __init__(self) -> None:
        self._durations: Dict[str, float] = {}

    @contextmanager
    def time_step(self, name: str, echo: bool = True) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            duration = end - start
            self._durations[name] = self._durations.get(name, 0.0) + duration
            if echo:
                print(f"[TIME] {name}: {duration:.3f}s")

    def get(self, name: str) -> Optional[float]:
        return self._durations.get(name)

    def to_lines(self) -> list[str]:
        lines: list[str] = []
        for key, seconds in self._durations.items():
            lines.append(f"{key}: {seconds:.3f}s")
        return lines

    def write_to_file(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for line in self.to_lines():
                f.write(line + "\n")


