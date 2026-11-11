"""Bloom filter sized from expected item count and target false-positive rate."""
import hashlib
import math
import struct


class Bloom:
    def __init__(self, n_items: int, fp_rate: float = 0.01, *, bits: bytearray = None, k: int = None):
        if bits is not None:
            self.bits, self.k = bits, k
            self.m = len(bits) * 8
            return
        n = max(n_items, 1)
        self.m = max(64, int(-n * math.log(fp_rate) / (math.log(2) ** 2)))
        self.m = (self.m + 7) // 8 * 8
        self.k = max(1, round(self.m / n * math.log(2)))
        self.bits = bytearray(self.m // 8)

    def _positions(self, key: bytes):
        # double hashing: g_i(x) = h1(x) + i*h2(x) mod m  (Kirsch & Mitzenmacher)
        h = hashlib.blake2b(key, digest_size=16).digest()
        h1, h2 = struct.unpack("<QQ", h)
        h2 |= 1
        return ((h1 + i * h2) % self.m for i in range(self.k))

    def add(self, key: bytes) -> None:
        for p in self._positions(key):
            self.bits[p >> 3] |= 1 << (p & 7)

    def __contains__(self, key: bytes) -> bool:
        return all(self.bits[p >> 3] & (1 << (p & 7)) for p in self._positions(key))

    def to_bytes(self) -> bytes:
        return struct.pack("<I", self.k) + bytes(self.bits)

    @classmethod
    def from_bytes(cls, raw: bytes) -> "Bloom":
        (k,) = struct.unpack("<I", raw[:4])
        return cls(0, bits=bytearray(raw[4:]), k=k)
