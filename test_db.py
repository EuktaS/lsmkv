import os
import random

import pytest

from lsmkv import DB
from lsmkv.bloom import Bloom
from lsmkv.wal import WAL


def small(path, **kw):
    kw.setdefault("memtable_bytes", 1500)
    kw.setdefault("block_records", 8)
    return DB(str(path), **kw)


def test_put_get_delete_overwrite(tmp_path):
    with small(tmp_path) as db:
        db.put(b"a", b"1")
        db.put(b"a", b"2")
        db.put(b"b", b"x")
        db.delete(b"b")
        assert db.get(b"a") == b"2"
        assert db.get(b"b") is None
        assert db.get(b"nope") is None


def test_rejects_bad_input(tmp_path):
    with small(tmp_path) as db:
        with pytest.raises(TypeError):
            db.put("str", b"x")
        with pytest.raises(ValueError):
            db.put(b"", b"x")


def test_data_survives_clean_reopen(tmp_path):
    with small(tmp_path) as db:
        for i in range(300):
            db.put(b"k%04d" % i, b"v%d" % i)
        db.delete(b"k0007")
    with small(tmp_path) as db:
        assert db.get(b"k0123") == b"v123"
        assert db.get(b"k0007") is None


def test_crash_without_close_recovers_from_wal(tmp_path):
    db = small(tmp_path, memtable_bytes=1 << 20)  # never flushes
    for i in range(50):
        db.put(b"k%02d" % i, b"v%d" % i)
    db.delete(b"k03")
    # simulate a crash: no close(), no flush. Open a second instance on the same dir.
    db2 = small(tmp_path, memtable_bytes=1 << 20)
    assert db2.get(b"k10") == b"v10"
    assert db2.get(b"k03") is None
    assert len(db2.tables) == 0


def test_torn_wal_tail_is_discarded(tmp_path):
    db = small(tmp_path, memtable_bytes=1 << 20)
    for i in range(10):
        db.put(b"k%d" % i, b"v%d" % i)
    wal = os.path.join(tmp_path, "wal.log")
    size = os.path.getsize(wal)
    with open(wal, "r+b") as f:
        f.truncate(size - 3)  # tear the last record
    db2 = small(tmp_path, memtable_bytes=1 << 20)
    assert db2.get(b"k8") == b"v8"
    assert db2.get(b"k9") is None  # torn write was never durable
    db2.put(b"k9", b"new")         # log is usable again after truncation
    assert list(WAL.replay(wal))[-1] == (b"k9", b"new")


def test_corrupt_wal_record_stops_replay(tmp_path):
    db = small(tmp_path, memtable_bytes=1 << 20)
    for i in range(10):
        db.put(b"k%d" % i, b"v%d" % i)
    wal = os.path.join(tmp_path, "wal.log")
    data = bytearray(open(wal, "rb").read())
    data[len(data) // 2] ^= 0xFF
    open(wal, "wb").write(bytes(data))
    got = list(WAL.replay(wal))
    assert 0 < len(got) < 10


def test_orphan_sstable_from_interrupted_flush_is_removed(tmp_path):
    with small(tmp_path) as db:
        db.put(b"a", b"1")
    orphan = os.path.join(tmp_path, "999999.sst")
    open(orphan, "wb").write(b"garbage")
    open(os.path.join(tmp_path, "000123.sst.tmp"), "wb").write(b"half written")
    with small(tmp_path) as db:
        assert db.get(b"a") == b"1"
    assert not os.path.exists(orphan)


def test_scan_is_ordered_deduped_and_hides_tombstones(tmp_path):
    with small(tmp_path) as db:
        for i in range(200):
            db.put(b"k%03d" % i, b"old")
        db.flush()
        for i in range(0, 200, 2):
            db.put(b"k%03d" % i, b"new")
        for i in range(0, 200, 5):
            db.delete(b"k%03d" % i)
        got = list(db.scan(b"k010", b"k030"))
        keys = [k for k, _ in got]
        assert keys == sorted(keys) and len(keys) == len(set(keys))
        assert b"k010" not in keys and b"k015" not in keys
        assert dict(got)[b"k012"] == b"new" and dict(got)[b"k013"] == b"old"


def test_tombstones_dropped_only_when_oldest_table_is_merged(tmp_path):
    with small(tmp_path, memtable_bytes=1 << 30, max_tables=100) as db:
        db.put(b"x", b"v1")
        db.flush()                 # table A (oldest) holds x
        db.put(b"y", b"v")
        db.flush()                 # table B
        db.delete(b"x")
        db.flush()                 # table C holds tombstone for x
        db.put(b"z", b"v")
        db.flush()                 # table D
        db.fanout = 2
        # force a window that does NOT include the oldest table
        sizes = [t.size for t in db.tables]
        assert len(sizes) == 4
        db.tables[3].size = 10**9  # make oldest very expensive so cheap window is elsewhere
        db.compact()
        assert db.get(b"x") is None  # tombstone must still shadow A's value
        db.compact()
        db.compact()
        assert db.get(b"x") is None


def test_compaction_preserves_data_and_reduces_tables(tmp_path):
    with small(tmp_path, max_tables=4) as db:
        model = {}
        for i in range(2000):
            k, v = b"k%05d" % (i % 700), b"v%d" % i
            db.put(k, v)
            model[k] = v
        assert db.stats["compactions"] > 0 and len(db.tables) <= 5
        assert all(db.get(k) == v for k, v in model.items())


def test_scan_survives_compaction_mid_iteration(tmp_path):
    with small(tmp_path, max_tables=100) as db:
        for batch in range(6):
            for i in range(100):
                db.put(b"k%04d" % (i * 6 + batch), b"v")
            db.flush()
        it = db.scan()
        first = [next(it) for _ in range(10)]
        while db.compact():
            pass
        rest = list(it)
        keys = [k for k, _ in first + rest]
        assert keys == sorted(keys) and len(keys) == 600


@pytest.mark.parametrize("seed", range(12))
def test_randomized_against_dict_model_with_reopens(tmp_path, seed):
    rng = random.Random(seed)
    model = {}
    db = small(tmp_path / f"db{seed}", max_tables=rng.choice([2, 4, 6]), fanout=rng.choice([2, 3, 4]))
    keyspace = [b"key%03d" % i for i in range(150)]
    for step in range(2500):
        r = rng.random()
        k = rng.choice(keyspace)
        if r < 0.55:
            v = b"v%d-%s" % (step, os.urandom(rng.randint(0, 20)))
            db.put(k, v)
            model[k] = v
        elif r < 0.75:
            db.delete(k)
            model.pop(k, None)
        elif r < 0.92:
            assert db.get(k) == model.get(k)
        elif r < 0.96:
            a, b = sorted(rng.sample(keyspace, 2))
            assert list(db.scan(a, b)) == sorted((x, y) for x, y in model.items() if a <= x < b)
        elif r < 0.98:
            db.compact()
        else:
            if rng.random() < 0.5:
                db.close()
            # else: simulate crash, leak the old instance
            db = small(tmp_path / f"db{seed}", max_tables=db.max_tables, fanout=db.fanout)
    assert list(db.scan()) == sorted(model.items())
    db.close()


def test_bloom_false_positive_rate_close_to_target():
    n = 5000
    b = Bloom(n, 0.01)
    for i in range(n):
        b.add(b"in%d" % i)
    assert all(b"in%d" % i in b for i in range(n))  # no false negatives, ever
    fp = sum(b"out%d" % i in b for i in range(20000)) / 20000
    assert fp < 0.02


def test_bloom_filters_avoid_disk_reads_for_missing_keys(tmp_path):
    with small(tmp_path, max_tables=100) as db:
        for t in range(5):
            for i in range(100):
                db.put(b"t%d-k%03d" % (t, i), b"v")
            db.flush()
        for i in range(1000):
            assert db.get(b"absent%d" % i) is None
        s = db.stats
        assert s["bloom_skipped"] > 0.95 * (s["bloom_skipped"] + s["bloom_passed"])
