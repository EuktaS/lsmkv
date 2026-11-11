"""Write-ahead log.

Record layout (little endian):  crc32 | klen | vlen | key | value
``vlen == 0xFFFFFFFF`` marks a tombstone (delete). The CRC covers everything
after itself. On replay we stop at the first record that is truncated or fails
its checksum (a torn write from a crash) and truncate the file there, so the
log is always a clean prefix of what was acknowledged.
"""
import os
import struct
import zlib
from typing import Iterator, Optional, Tuple

HDR = struct.Struct("<III")
TOMBSTONE_LEN = 0xFFFFFFFF


class WAL:
    def __init__(self, path: str, sync: bool = False):
        self.path, self.sync = path, sync
        self.f = open(path, "ab", buffering=0)  # unbuffered: write() goes straight to the OS
        self.bytes_written = 0

    def append(self, key: bytes, value: Optional[bytes]) -> None:
        vlen = TOMBSTONE_LEN if value is None else len(value)
        body = struct.pack("<II", len(key), vlen) + key + (b"" if value is None else value)
        rec = struct.pack("<I", zlib.crc32(body)) + body
        self.f.write(rec)
        self.bytes_written += len(rec)
        if self.sync:
            os.fsync(self.f.fileno())

    def reset(self) -> None:
        self.f.close()
        with open(self.path, "wb"):
            pass
        self.f = open(self.path, "ab", buffering=0)

    def close(self) -> None:
        self.f.close()

    @staticmethod
    def replay(path: str) -> Iterator[Tuple[bytes, Optional[bytes]]]:
        if not os.path.exists(path):
            return
        with open(path, "rb") as f:
            data = f.read()
        pos, good = 0, 0
        while pos + HDR.size <= len(data):
            crc, klen, vlen = HDR.unpack_from(data, pos)
            vsz = 0 if vlen == TOMBSTONE_LEN else vlen
            end = pos + HDR.size + klen + vsz
            if end > len(data):
                break
            if zlib.crc32(data[pos + 4:end]) != crc:
                break
            key = data[pos + HDR.size:pos + HDR.size + klen]
            val = None if vlen == TOMBSTONE_LEN else data[pos + HDR.size + klen:end]
            yield key, val
            pos = good = end
        if good < len(data):
            with open(path, "r+b") as f:  # drop the torn tail
                f.truncate(good)
