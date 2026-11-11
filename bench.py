"""Benchmark: write throughput, write amplification, point reads, bloom effectiveness."""
import os
import random
import shutil
import tempfile
import time

from lsmkv import DB


def main(n: int = 100_000, value_size: int = 100, seed: int = 7) -> None:
    rng = random.Random(seed)
    d = tempfile.mkdtemp()
    try:
        db = DB(d, memtable_bytes=1 << 20, fanout=4, max_tables=8)
        keys = [b"user:%08d" % rng.randrange(n * 4) for _ in range(n)]
        val = os.urandom(value_size)

        t0 = time.perf_counter()
        for k in keys:
            db.put(k, val)
        db.flush()
        w = time.perf_counter() - t0
        print(f"writes : {n:,} puts in {w:.2f}s -> {n / w:,.0f} ops/s (no fsync per write)")
        print(f"         flushes={db.stats['flushes']} compactions={db.stats['compactions']} "
              f"tables={len(db.tables)} write amplification={db.write_amplification():.2f}x")

        sample = [rng.choice(keys) for _ in range(20_000)]
        t0 = time.perf_counter()
        hits = sum(db.get(k) is not None for k in sample)
        r = time.perf_counter() - t0
        print(f"hits   : {len(sample):,} gets in {r:.2f}s -> {len(sample) / r:,.0f} ops/s ({hits:,} found)")

        before = dict(db.stats)
        miss = [b"nokey:%08d" % i for i in range(20_000)]
        t0 = time.perf_counter()
        for k in miss:
            db.get(k)
        m = time.perf_counter() - t0
        skipped = db.stats["bloom_skipped"] - before["bloom_skipped"]
        reads = db.stats["sst_reads"] - before["sst_reads"]
        print(f"misses : {len(miss):,} gets in {m:.2f}s -> {len(miss) / m:,.0f} ops/s")
        print(f"         bloom/range checks avoided {skipped:,} table probes; {reads} block reads were needed "
              f"({reads / max(1, skipped + reads):.2%} of probes)")

        t0 = time.perf_counter()
        cnt = sum(1 for _ in db.scan(b"user:00100000", b"user:00200000"))
        s = time.perf_counter() - t0
        print(f"scan   : {cnt:,} rows in range in {s:.3f}s")
        db.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
