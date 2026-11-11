"""A small LSM-tree key-value store.

Write path : WAL append -> memtable -> (when big) flush to an immutable SSTable
Read path  : memtable -> SSTables newest to oldest, skipping tables whose bloom
             filter or key range rules the key out
Compaction : merge a window of adjacent tables, drop shadowed versions, and drop
             tombstones only when the window includes the oldest table
Recovery   : manifest lists live tables; the WAL is replayed into the memtable.

Crash-safety ordering on flush:
  1. write SSTable to *.tmp, fsync, rename       (data durable, not yet referenced)
  2. write MANIFEST.tmp, fsync, rename, fsync dir (table becomes live atomically)
  3. truncate WAL
A crash between 1 and 2 leaves an unreferenced file that is deleted on open and
the WAL still has the data. A crash between 2 and 3 replays WAL entries that are
already in the table; replay is idempotent because newer data shadows older.
"""
from __future__ import annotations

import heapq
import os
import threading
from bisect import bisect_left, insort
from typing import Dict, Iterator, List, Optional, Tuple

from .sstable import MISSING, SSTable, write_sstable
from .wal import WAL

MANIFEST = "MANIFEST"


class DB:
    def __init__(self, path: str, memtable_bytes: int = 1 << 20, fanout: int = 4,
                 max_tables: int = 8, sync: bool = False, block_records: int = 64):
        self.path = path
        self.memtable_limit = memtable_bytes
        self.fanout, self.max_tables, self.block_records = fanout, max_tables, block_records
        self._lock = threading.RLock()
        self.stats: Dict[str, int] = dict(
            user_bytes=0, wal_bytes=0, flush_bytes=0, compaction_bytes=0, flushes=0, compactions=0,
            bloom_skipped=0, bloom_passed=0, bloom_false_positive=0, sst_reads=0)
        os.makedirs(path, exist_ok=True)

        names = self._read_manifest()
        self._cleanup_strays(set(names))
        self.tables: List[SSTable] = [SSTable(os.path.join(path, n)) for n in names]  # newest first
        self._next_file = 1 + max((int(n.split(".")[0]) for n in names), default=0)

        self.mem: Dict[bytes, Optional[bytes]] = {}
        self.mem_keys: List[bytes] = []
        self.mem_size = 0
        wal_path = os.path.join(path, "wal.log")
        for k, v in WAL.replay(wal_path):
            self._mem_put(k, v)
        self.wal = WAL(wal_path, sync=sync)

    # ----------------------------------------------------------------- public
    def put(self, key: bytes, value: bytes) -> None:
        self._check(key, value)
        with self._lock:
            self.stats["user_bytes"] += len(key) + len(value)
            self.wal.append(key, value)
            self._mem_put(key, value)
            self._maybe_flush()

    def delete(self, key: bytes) -> None:
        self._check(key, b"")
        with self._lock:
            self.stats["user_bytes"] += len(key)
            self.wal.append(key, None)
            self._mem_put(key, None)
            self._maybe_flush()

    def get(self, key: bytes) -> Optional[bytes]:
        with self._lock:
            if key in self.mem:
                return self.mem[key]
            for t in self.tables:
                if not t.may_contain(key):
                    self.stats["bloom_skipped"] += 1
                    continue
                self.stats["bloom_passed"] += 1
                self.stats["sst_reads"] += 1
                r = t.get(key)
                if r is not MISSING:
                    return r  # may be None: a tombstone
                self.stats["bloom_false_positive"] += 1
            return None

    def scan(self, start: Optional[bytes] = None, end: Optional[bytes] = None
             ) -> Iterator[Tuple[bytes, bytes]]:
        """Ordered iterator over start <= key < end.

        Iterates a consistent snapshot: the memtable slice is copied and the
        tables are pinned, so concurrent flushes/compactions cannot pull files
        out from under it.
        """
        with self._lock:
            lo = 0 if start is None else bisect_left(self.mem_keys, start)
            hi = len(self.mem_keys) if end is None else bisect_left(self.mem_keys, end)
            mem_items = [(k, self.mem[k]) for k in self.mem_keys[lo:hi]]
            tables = list(self.tables)
            for t in tables:
                t.pins += 1
        try:
            def tagged(it, rank):
                for k, v in it:
                    yield k, rank, v
            sources = [tagged(iter(mem_items), -1)]
            sources += [tagged(t.scan(start, end), i) for i, t in enumerate(tables)]
            last = None
            for k, _, v in heapq.merge(*sources, key=lambda x: (x[0], x[1])):
                if k == last:
                    continue  # older version of a key we already emitted/skipped
                last = k
                if v is not None:
                    yield k, v
        finally:
            with self._lock:
                for t in tables:
                    t.pins -= 1
                    if t.retired and t.pins == 0:
                        t.close()

    def flush(self) -> None:
        with self._lock:
            self._flush()

    def compact(self) -> bool:
        """Merge one window of tables. Returns False if there is nothing to do."""
        with self._lock:
            return self._compact_once()

    def close(self) -> None:
        with self._lock:
            self._flush()
            self.wal.close()
            for t in self.tables:
                t.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def write_amplification(self) -> float:
        s = self.stats
        return (s["wal_bytes"] + s["flush_bytes"] + s["compaction_bytes"]) / max(1, s["user_bytes"])

    # --------------------------------------------------------------- internals
    @staticmethod
    def _check(key, value) -> None:
        if not isinstance(key, bytes) or not isinstance(value, bytes):
            raise TypeError("keys and values must be bytes")
        if not key:
            raise ValueError("empty key")

    def _mem_put(self, key: bytes, value: Optional[bytes]) -> None:
        if key not in self.mem:
            insort(self.mem_keys, key)
            self.mem_size += len(key) + 16
        else:
            old = self.mem[key]
            self.mem_size -= 0 if old is None else len(old)
        self.mem[key] = value
        self.mem_size += 0 if value is None else len(value)

    def _maybe_flush(self) -> None:
        self.stats["wal_bytes"] = self.wal.bytes_written
        if self.mem_size >= self.memtable_limit:
            self._flush()

    def _new_name(self) -> str:
        name = f"{self._next_file:06d}.sst"
        self._next_file += 1
        return name

    def _flush(self) -> None:
        if not self.mem:
            return
        name = self._new_name()
        items = ((k, self.mem[k]) for k in self.mem_keys)
        written = write_sstable(os.path.join(self.path, name), items, len(self.mem_keys), self.block_records)
        self.stats["flush_bytes"] += written
        self.stats["flushes"] += 1
        self.tables.insert(0, SSTable(os.path.join(self.path, name)))
        self._write_manifest()
        self.wal.reset()
        self.stats["wal_bytes"] = self.wal.bytes_written
        self.mem, self.mem_keys, self.mem_size = {}, [], 0
        while len(self.tables) > self.max_tables:
            if not self._compact_once():
                break

    def _compact_once(self) -> bool:
        n = len(self.tables)
        if n < 2:
            return False
        w = min(self.fanout, n)
        sizes = [t.size for t in self.tables]
        i = min(range(n - w + 1), key=lambda j: sum(sizes[j:j + w]))  # cheapest adjacent window
        window = self.tables[i:i + w]
        drop_tombstones = i + w == n  # nothing older could be shadowed by a dropped tombstone

        def merged():
            def tagged(it, rank):
                for k, v in it:
                    yield k, rank, v
            srcs = [tagged(t.scan(), r) for r, t in enumerate(window)]
            last = None
            for k, _, v in heapq.merge(*srcs, key=lambda x: (x[0], x[1])):
                if k == last:
                    continue
                last = k
                if v is None and drop_tombstones:
                    continue
                yield k, v

        name = self._new_name()
        path = os.path.join(self.path, name)
        written = write_sstable(path, merged(), sum(t.count for t in window), self.block_records)
        self.stats["compaction_bytes"] += written
        self.stats["compactions"] += 1
        new = SSTable(path)
        if new.count == 0:           # everything cancelled out
            new.close()
            os.unlink(path)
            replacement = []
        else:
            replacement = [new]
        self.tables[i:i + w] = replacement
        self._write_manifest()
        for t in window:             # POSIX: unlink now, fd stays valid for pinned scans
            os.unlink(t.path)
            t.retired = True
            if t.pins == 0:
                t.close()
        return True

    # ---------------------------------------------------------------- manifest
    def _read_manifest(self) -> List[str]:
        p = os.path.join(self.path, MANIFEST)
        if not os.path.exists(p):
            return []
        with open(p) as f:
            return [ln.strip() for ln in f if ln.strip()]

    def _write_manifest(self) -> None:
        p = os.path.join(self.path, MANIFEST)
        with open(p + ".tmp", "w") as f:
            f.write("\n".join(os.path.basename(t.path) for t in self.tables) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(p + ".tmp", p)
        dfd = os.open(self.path, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)

    def _cleanup_strays(self, live: set) -> None:
        for n in os.listdir(self.path):
            if n.endswith(".tmp") or (n.endswith(".sst") and n not in live):
                os.unlink(os.path.join(self.path, n))
