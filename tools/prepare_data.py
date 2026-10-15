"""Reconstruct frozen paper datasets from pinned public shards and selected rows.

This replays accepted documents, not historical discovery/deduplication decisions.
All reconstructed token/boundary arrays and holdout metadata must match old hashes.
"""
import argparse
from contextlib import contextmanager
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess

os.environ.setdefault("RAYON_NUM_THREADS", "2")

ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_TIMEOUT_SECONDS = 7500


@contextmanager
def exclusive_lock(path):
    """Refuse concurrent writers without treating a leftover lock file as active."""
    try:
        import fcntl
    except ImportError as error:
        raise RuntimeError("Data preparation requires POSIX file locks (Linux or macOS). "
                           "--verify-only remains available without them.") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another preparation or download holds {path}; "
                               "retry after it exits. Do not remove an active lock file.") from error
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps({"pid": os.getpid()}) + "\n")
        stream.flush()
        yield stream
        # Closing releases the lock after any inherited curl descriptor closes.
        # An explicit LOCK_UN here would also unlock a still-running child.
    # Keep the inode: unlinking a lock file can allow two independent locks.


def load_dependencies():
    """Hash verification and CLI help only need the Python standard library."""
    try:
        numpy = importlib.import_module("numpy")
        parquet = importlib.import_module("pyarrow.parquet")
        tokenizers = importlib.import_module("tokenizers")
    except ImportError as error:
        raise RuntimeError(
            "Data reconstruction requires NumPy, PyArrow and tokenizers. "
            f"Install them with: python -m pip install -r {ROOT / 'reproduction/requirements.txt'} "
            "(use the same Python environment that runs this script). "
            f"Import error: {error}"
        ) from error
    return numpy, parquet, tokenizers.Tokenizer


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check(path, expected):
    path = Path(path)
    require(path.is_file(), f"Missing file: {path}")
    require(sha(path) == expected, f"SHA-256 mismatch: {path}")


def check_source(path, source):
    require(path.is_file(), f"Missing source file: {path}")
    require(path.stat().st_size == source["bytes"],
            f"Source byte count differs: {path}; expected {source['bytes']}, "
            f"got {path.stat().st_size}. The file has been retained for inspection.")
    check(path, source["sha256"])


def source_file(source, raw, download):
    path = raw / source["filename"]
    if path.exists() or path.is_symlink():
        check_source(path, source)
        return path
    require(download, f"Missing {path}; supply the pinned shard or use --download")
    with exclusive_lock(raw / f".{path.name}.download.lock") as lock:
        # A competing process may have finished between the first check and the lock.
        if path.exists() or path.is_symlink():
            check_source(path, source)
            return path
        download_source(source, path, lock)
    return path


def download_source(source, path, lock):
    temporary = path.with_suffix(path.suffix + ".download")
    if temporary.exists():
        require(temporary.is_file(), f"Download cache is not a file: {temporary}")
        require(temporary.stat().st_size <= source["bytes"],
                f"Download cache is larger than the pinned source: {temporary}; "
                "inspect it before retrying. It has not been replaced.")
    if not temporary.exists() or temporary.stat().st_size < source["bytes"]:
        require(shutil.which("curl") is not None,
                "curl is required for --download. Install curl, or download the pinned "
                f"shard manually to {path} and omit --download.")
        try:
            subprocess.run([
                "curl", "--fail", "--location", "--proto", "=https", "--proto-redir", "=https",
                "--continue-at", "-", "--retry", "3", "--retry-max-time", "300",
                "--connect-timeout", "30", "--max-time", "7200",
                "--speed-limit", "1024", "--speed-time", "120",
                "--output", str(temporary), source["url"],
            ], check=True, timeout=DOWNLOAD_TIMEOUT_SECONDS, pass_fds=(lock.fileno(),))
            # curl inherits the lock so an interrupted parent cannot expose its live
            # .download writer to another preparation process.
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError(
                f"Download failed for {source['url']}. Any received bytes remain at "
                f"{temporary}; rerun the same command to resume. If the server does not "
                "support resumed downloads, inspect the cache and remove only that "
                f".download file before retrying. Details: {error}"
            ) from error
    check_source(temporary, source)
    temporary.replace(path)


def rebuild(component, sources, tokenizer, folder):
    np, pq, _ = load_dependencies()
    selection_path = ROOT / component["selection"]
    check(selection_path, component["selection_sha256"])
    with gzip.open(selection_path, "rt", encoding="utf-8") as stream:
        records = json.load(stream)
    require(len(records) == component["documents"], "Selection document count differs")
    split = component["split"]
    token_count = component["tokens"]
    folder.mkdir(parents=True, exist_ok=True)
    token_path = folder / f"tokens_{split}.npy"
    tokens = np.lib.format.open_memmap(token_path, mode="w+", dtype="<u2", shape=(token_count,))
    bounds = np.empty((len(records), 2), dtype="<i8")
    cursor = 0
    for index, record in enumerate(records):
        size = record["length"]
        bounds[index] = cursor, cursor + size
        cursor += size
    require(cursor == token_count, "Selection token total differs")
    metadata = [None] * len(records) if component.get("metadata") else None
    done = set()
    last_report = 0
    for source_key in dict.fromkeys(r["source"] for r in records):
        indexes = [i for i, r in enumerate(records) if r["source"] == source_key]
        lookup = {records[i]["row"]: i for i in indexes}
        require(len(lookup) == len(indexes), "Duplicate selected source row")
        maximum_row = max(lookup)
        parquet = pq.ParquetFile(sources[source_key])
        require(min(lookup) >= 0 and maximum_row < parquet.metadata.num_rows, "Row outside shard")
        offset = 0
        for batch in parquet.iter_batches(batch_size=256, columns=["text"]):
            chosen = [(lookup[offset+j], text) for j, text in enumerate(batch.column(0).to_pylist())
                      if offset+j in lookup]
            offset += batch.num_rows
            if chosen:
                encodings = tokenizer.encode_batch([text for _, text in chosen], add_special_tokens=False)
                for (index, text), encoding in zip(chosen, encodings):
                    record = records[index]
                    ids = encoding.ids + [3]
                    require(len(ids) == record["length"], f"Token length differs at document {index}")
                    require(min(ids) >= 0 and max(ids) < 8192, "Token outside vocabulary")
                    start, end = bounds[index]
                    tokens[start:end] = ids
                    if metadata is not None:
                        row = dict(record["metadata"])
                        require(hashlib.sha256(text.encode()).hexdigest() == row["original_text_sha256"],
                                "Holdout text differs")
                        row["text"] = text  # Existing key retains original serialization order.
                        metadata[index] = json.dumps(row, ensure_ascii=False) + "\n"
                    done.add(index)
                if len(done) - last_report >= 10000:
                    print(f"{component['name']}: {len(done)}/{len(records)} documents encoded", flush=True)
                    last_report = len(done)
            if offset > maximum_row:
                break
        print(f"{component['name']}: {len(done)}/{len(records)} documents encoded", flush=True)
    require(len(done) == len(records), "Not all selected rows were reconstructed")
    tokens.flush()
    del tokens
    np.save(folder / f"docs_{split}.npy", bounds, allow_pickle=False)
    if metadata is not None:
        with (folder / component["metadata"]).open("w", encoding="utf-8", newline="\n") as stream:
            stream.writelines(metadata)
        np.save(folder / "eval_offsets.npy", bounds[:, 0], allow_pickle=False)
        np.save(folder / "eval_document_ids.npy", np.arange(len(records), dtype="<i8"), allow_pickle=False)
    for name, expected in component["files_sha256"].items():
        check(folder / name, expected)
    print(f"PASS: {component['name']} historical payload hashes", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("small", "medium", "all"), required=True)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw", help="Pinned parquet cache")
    parser.add_argument("--output", type=Path, required=True, help="New output root")
    parser.add_argument("--download", action="store_true", help="Download missing public shards using curl")
    parser.add_argument("--verify-only", action="store_true", help="Check an existing prepared output")
    args = parser.parse_args(argv)
    args.raw = args.raw.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    plan_path = ROOT / "reproduction/data_plan.json"
    plan = json.loads(plan_path.read_text())
    components = [c for c in plan["components"] if args.study in ("all", c["study"])]
    tokenizer_path = ROOT / "tokenizer/tokenizer.json"
    check(tokenizer_path, plan["tokenizer_sha256"])
    if args.verify_only:
        for component in components:
            for name, expected in component["files_sha256"].items():
                check(args.output / component["folder"] / name, expected)
        for study in sorted({c["study"] for c in components}):
            check(args.output / f"{study}_corpus/tokenizer.json", plan["tokenizer_sha256"])
        print("PASS: all requested historical payload hashes")
        return
    require(not args.output.exists(), "Output already exists; use --verify-only or a new path")
    staging = args.output.with_name(args.output.name + ".partial")
    require(not staging.exists() and not staging.is_symlink(),
            f"Partial output exists; inspect it before retrying: {staging}")
    _, _, Tokenizer = load_dependencies()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    require(tokenizer.get_vocab_size() == 8192 and tokenizer.token_to_id("<eos>") == 3,
            "Tokenizer contract differs")
    with exclusive_lock(args.output.parent / f".{args.output.name}.prepare.lock"):
        require(not args.output.exists(), "Output already exists; use --verify-only or a new path")
        require(not staging.exists() and not staging.is_symlink(),
                f"Partial output exists; inspect it before retrying: {staging}")
        sources = {}
        for key in dict.fromkeys(k for c in components for k in c["sources"]):
            sources[key] = source_file(plan["sources"][key], args.raw, args.download)
            print(f"PASS: source {key}", flush=True)
        staging.mkdir(parents=True)
        for component in components:
            rebuild(component, sources, tokenizer, staging / component["folder"])
        for study in sorted({c["study"] for c in components}):
            shutil.copyfile(tokenizer_path, staging / f"{study}_corpus/tokenizer.json")
        report = {"scope": "Exact payload reconstruction; not a re-audit of historical exclusions or a training rerun",
                  "plan_sha256": sha(plan_path), "study": args.study,
                  "components": [c["name"] for c in components], "status": "complete"}
        (staging / "reconstruction.json").write_text(json.dumps(report, indent=2) + "\n")
        require(not args.output.exists(),
                f"Output appeared during reconstruction: {args.output}; verified work remains at {staging}")
        staging.rename(args.output)
    print(f"Complete: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Preparation failed: {error}") from error
