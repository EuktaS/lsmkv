# lsmkv

A small log-structured merge-tree key-value store in pure Python: the same
family of design as LevelDB and RocksDB, built to understand how durability,
compaction and read amplification trade off.

## What's in it

- **WAL** with CRC32 per record; replay stops at the first torn or corrupt
  record and truncates there
- **Memtable** (dict plus sorted key list) flushed to immutable **SSTables**
- **SSTable format**: data blocks, sparse index, Bloom filter, fixed footer
  with a magic number
- **Bloom filters** (double hashing) so most lookups for absent keys never
  touch disk
- **Manifest** with atomic replace plus directory fsync, so a table becomes
  live in one step
- **Size-aware compaction**: merges the cheapest window of adjacent tables;
  tombstones are only dropped when the window includes the oldest table
- **Snapshot scans**: range iterators pin the tables they read, so a
  compaction mid-iteration cannot break them
- Stats for write amplification and bloom filter effectiveness

## Run it

    pip install pytest
    python -m pytest -q
    python bench.py

    from lsmkv import DB
    with DB("./data") as db:
        db.put(b"user:1", b"alice")
        db.delete(b"user:2")
        db.get(b"user:1")
        list(db.scan(b"user:", b"user;"))

## Design decisions

**Crash ordering on flush.** Write the SSTable to a temp file, fsync, rename.
Then replace the manifest and fsync the directory. Only then truncate the WAL.
A crash after the table is written but before the manifest update leaves an
unreferenced file (deleted on open) and an intact WAL. A crash after the
manifest update but before the WAL truncate replays entries that are already
in the table, which is harmless because newer data shadows older data.

**Why tombstones can't always be dropped.** A delete is a tombstone that has
to shadow older values living in older tables. If a compaction only merges a
subset of tables, dropping the tombstone would let an old value in an
unmerged older table reappear. So tombstones are dropped only when the
compaction window includes the oldest table. There's a test for exactly this.

**Bloom filters include tombstoned keys.** Otherwise a delete in a newer table
would be skipped and an older table would answer with the stale value.

**Sparse index.** One index entry per 64 records keeps memory small; a point
lookup is a bloom check, a binary search in memory, one `pread` of one block,
and a short linear scan.

## Testing

25 tests. The main one is a randomized model test: 12 seeds, 2,500 operations
each (put, delete, get, range scan, manual compaction, clean reopen, and
simulated crash by abandoning the instance and reopening the directory),
compared against a plain Python dict after every read. Others cover torn WAL
tails, corrupt records, orphan files from an interrupted flush, scan during
compaction, and Bloom false-positive rate.

## Measured results (this machine, CPython 3.12, 100k puts of ~100B values)

    writes : ~84,000 ops/s (WAL written without fsync per write)
    write amplification: 2.86x (WAL + flush + compaction bytes / user bytes)
    point reads, keys present: ~26,000 ops/s
    point reads, keys absent:  ~366,000 ops/s; 140,000 table probes avoided,
                               0 block reads needed

## Limitations

- Single process. A lock makes it thread-safe but there is no concurrent
  flush or background compaction thread.
- `sync=True` fsyncs every write; the default trades durability of the last
  few writes on power loss for speed, like most stores' default settings.
- Compaction is a simple tiered-window policy, not full leveled compaction
  with non-overlapping key ranges, so read amplification grows with table
  count until compaction catches up.
- Unlink-while-open file handling assumes POSIX semantics.
