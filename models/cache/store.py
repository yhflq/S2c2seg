import threading
from collections import OrderedDict


class MemoryStore:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self._entries = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if self.capacity <= 0:
            return None
        with self._lock:
            if key not in self._entries:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return self._entries[key]

    def put(self, key, value):
        if self.capacity <= 0:
            return
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self.capacity:
                self._entries.popitem(last=False)

    def clear(self):
        with self._lock:
            self._entries.clear()

    def stats(self):
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total else 0.0,
            "size": len(self._entries),
        }
