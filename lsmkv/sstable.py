import os
import struct
from bisect import bisect_right
from typing import Iterable, Iterator, List, Optional, Tuple

from .bloom import Bloom

REC = struct.Struct("<II")
FOOTER = struct.Struct("<QQQQ")
MAGIC = 0x4C534D4B56303031  # "LSMKV001"
TOMB = 0xFFFFFFFF
MISSING = object()  # distinguishes "not in this table" from "deleted in this table"


def write_sstable(path: str, items: Iterable[Tuple[bytes, Optional[bytes]]], expected: int,
                  block_records: int = 64, fp_rate: float = 0.01) -> int:
    """Write items (must be sorted, unique keys) atomically. Returns bytes written."""
    bloom = Bloom(expected, fp_rate)
    index: List[Tuple[bytes, int]] = []
    tmp = path + ".tmp"
    count, last_key, off = 0, b"", 0
    with open(tmp, "wb") as f:
        for key, val in items:
            if count % block_records == 0:
                index.append((key, off))
            vlen = TOMB if val is None else len(val)
            rec = REC.pack(len(key), vlen) + key + (b"" if val is None else val)
            f.write(rec)
            off += len(rec)
            bloom.add(key)
            last_key = key
            count += 1
        index_off = off
        buf = [struct.pack("<I", len(index))]
        for k, o in index:
            buf.append(struct.pack("<IQ", len(k), o) + k)
        buf.append(struct.pack("<I", len(last_key)) + last_key)
        index_bytes = b"".join(buf)
        f.write(index_bytes)
        bloom_off = index_off + len(index_bytes)
        f.write(bloom.to_bytes())
        f.write(FOOTER.pack(index_off, bloom_off, count, MAGIC))
        f.flush()
        os.fsync(f.fileno())
        total = f.tell()
    os.replace(tmp, path)
    return total


class SSTable:
    def __init__(self, path: str):
        self.path = path
        self.pins = 0          # active scans using this table
        self.retired = False   # removed from the manifest; close when pins hit 0
        self.fd = os.open(path, os.O_RDONLY)
        size = os.fstat(self.fd).st_size
        index_off, bloom_off, self.count, magic = FOOTER.unpack(os.pread(self.fd, FOOTER.size, size - FOOTER.size))
        if magic != MAGIC:
            raise ValueError(f"bad sstable magic in {path}")
        self.data_end = index_off
        self.size = size
        raw = os.pread(self.fd, size - FOOTER.size - index_off, index_off)
        (n,) = struct.unpack_from("<I", raw, 0)
        pos, self.keys, self.offsets = 4, [], []
        for _ in range(n):
            klen, o = struct.unpack_from("<IQ", raw, pos)
            pos += 12
            self.keys.append(raw[pos:pos + klen])
            self.offsets.append(o)
            pos += klen
        (mlen,) = struct.unpack_from("<I", raw, pos)
        self.max_key = raw[pos + 4:pos + 4 + mlen]
        self.min_key = self.keys[0] if self.keys else b""
        self.bloom = Bloom.from_bytes(raw[bloom_off - index_off:])

    def close(self) -> None:
        os.close(self.fd)

    def _block(self, i: int) -> bytes:
        start = self.offsets[i]
        end = self.offsets[i + 1] if i + 1 < len(self.offsets) else self.data_end
        return os.pread(self.fd, end - start, start)

    @staticmethod
    def _parse(block: bytes) -> Iterator[Tuple[bytes, Optional[bytes]]]:
        pos = 0
        while pos < len(block):
            klen, vlen = REC.unpack_from(block, pos)
            pos += REC.size
            key = block[pos:pos + klen]
            pos += klen
            if vlen == TOMB:
                yield key, None
            else:
                yield key, block[pos:pos + vlen]
                pos += vlen

    def may_contain(self, key: bytes) -> bool:
        return bool(self.keys) and self.min_key <= key <= self.max_key and key in self.bloom

    def get(self, key: bytes):
        """Returns value, None (tombstone) or MISSING."""
        i = bisect_right(self.keys, key) - 1
        if i < 0:
            return MISSING
        for k, v in self._parse(self._block(i)):
            if k == key:
                return v
            if k > key:
                break
        return MISSING

    def scan(self, start: Optional[bytes] = None, end: Optional[bytes] = None
             ) -> Iterator[Tuple[bytes, Optional[bytes]]]:
        """Yield (key, value|None) for start <= key < end, in order."""
        i = 0 if start is None else max(0, bisect_right(self.keys, start) - 1)
        while i < len(self.keys):
            for k, v in self._parse(self._block(i)):
                if start is not None and k < start:
                    continue
                if end is not None and k >= end:
                    return
                yield k, v
            i += 1
