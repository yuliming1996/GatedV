"""Bounded reconstruction tests: tiny local Parquet, no network or model training.

Run: python -B -m unittest discover -s tests -p 'test_prepare_data.py' -v
Install reproduction/requirements.txt to include the real tokenizer/Parquet tests.
"""
import contextlib
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "tools/prepare_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_data_under_test", SCRIPT)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)

IMPORT_PREPARE = (
    "import importlib.util,json,sys; from pathlib import Path; "
    "s=importlib.util.spec_from_file_location('prepare',sys.argv[1]); "
    "p=importlib.util.module_from_spec(s);s.loader.exec_module(p); "
)


class SourceIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.raw = Path(self.temporary.name) / "raw"
        self.raw.mkdir()
        self.data = b"tiny pinned source bytes"
        self.source = {
            "filename": "tiny.parquet", "bytes": len(self.data),
            "sha256": hashlib.sha256(self.data).hexdigest(),
            "url": "https://example.invalid/pinned/tiny.parquet",
        }
        self.path = self.raw / self.source["filename"]
        self.partial = self.path.with_suffix(".parquet.download")

    def test_verified_cache_never_downloads(self):
        self.path.write_bytes(self.data)
        with patch.object(prepare.subprocess, "run") as download:
            self.assertEqual(prepare.source_file(self.source, self.raw, True), self.path)
        download.assert_not_called()

    def test_corrupt_cache_is_rejected_without_overwrite(self):
        bad = b"x" * len(self.data)
        self.path.write_bytes(bad)
        with patch.object(prepare.subprocess, "run") as download:
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                prepare.source_file(self.source, self.raw, True)
        download.assert_not_called()
        self.assertEqual(self.path.read_bytes(), bad)

    def test_missing_cache_without_download_is_actionable(self):
        with self.assertRaisesRegex(ValueError, "use --download"):
            prepare.source_file(self.source, self.raw, False)

    def test_missing_curl_is_actionable(self):
        with patch.object(prepare.shutil, "which", return_value=None):
            with self.assertRaisesRegex(ValueError, "curl is required"):
                prepare.source_file(self.source, self.raw, True)

    def test_partial_download_resumes_then_verifies_before_commit(self):
        self.partial.write_bytes(self.data[:5])

        def download(command, **options):
            self.assertEqual(command[command.index("--continue-at") + 1], "-")
            self.assertEqual(options["timeout"], prepare.DOWNLOAD_TIMEOUT_SECONDS)
            self.assertEqual(self.partial.read_bytes(), self.data[:5])
            with self.partial.open("ab") as stream:
                stream.write(self.data[5:])
            return subprocess.CompletedProcess(command, 0)

        with patch.object(prepare.shutil, "which", return_value="/usr/bin/curl"), \
                patch.object(prepare.subprocess, "run", side_effect=download):
            prepare.source_file(self.source, self.raw, True)
        self.assertEqual(self.path.read_bytes(), self.data)
        self.assertFalse(self.partial.exists())

    def test_failed_download_preserves_partial_without_committing(self):
        self.partial.write_bytes(self.data[:5])
        with patch.object(prepare.shutil, "which", return_value="/usr/bin/curl"), \
                patch.object(prepare.subprocess, "run", side_effect=subprocess.TimeoutExpired("curl", 1)):
            with self.assertRaisesRegex(RuntimeError, "rerun the same command to resume"):
                prepare.source_file(self.source, self.raw, True)
        self.assertEqual(self.partial.read_bytes(), self.data[:5])
        self.assertFalse(self.path.exists())

    def test_complete_partial_hash_is_checked_without_redownload(self):
        self.partial.write_bytes(b"x" * len(self.data))
        with patch.object(prepare.subprocess, "run") as download:
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                prepare.source_file(self.source, self.raw, True)
        download.assert_not_called()
        self.assertTrue(self.partial.exists())
        self.assertFalse(self.path.exists())

    def test_complete_valid_partial_is_committed_without_download(self):
        self.partial.write_bytes(self.data)
        with patch.object(prepare.subprocess, "run") as download:
            prepare.source_file(self.source, self.raw, True)
        download.assert_not_called()
        self.assertEqual(self.path.read_bytes(), self.data)

    def test_missing_dependencies_are_actionable(self):
        with patch.object(prepare.importlib, "import_module", side_effect=ImportError("numpy absent")):
            with self.assertRaisesRegex(RuntimeError, "reproduction/requirements.txt"):
                prepare.load_dependencies()

    def test_second_download_process_is_refused_without_touching_partial(self):
        self.partial.write_bytes(self.data[:5])
        lock_path = self.raw / f".{self.path.name}.download.lock"
        with prepare.exclusive_lock(lock_path):
            result = subprocess.run([
                sys.executable, "-B", "-c", IMPORT_PREPARE +
                "p.source_file(json.loads(sys.argv[2]),Path(sys.argv[3]),True)",
                str(SCRIPT), json.dumps(self.source), str(self.raw),
            ], capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Another preparation or download holds", result.stderr)
        self.assertEqual(self.partial.read_bytes(), self.data[:5])
        self.assertFalse(self.path.exists())

    def test_download_child_keeps_lock_until_it_exits(self):
        lock_path = self.raw / ".inheritance_test.lock"
        with prepare.exclusive_lock(lock_path) as lock:
            child = subprocess.Popen([
                sys.executable, "-B", "-c", "import sys; sys.stdin.readline()",
            ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, pass_fds=(lock.fileno(),))
        try:
            with self.assertRaisesRegex(RuntimeError, "Another preparation or download holds"):
                with prepare.exclusive_lock(lock_path):
                    pass
        finally:
            try:
                child.communicate("finish\n", timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.communicate(timeout=10)
        self.assertEqual(child.returncode, 0)
        # A retained lock file is harmless after every owning process has exited.
        with prepare.exclusive_lock(lock_path):
            pass


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        try:
            self.np, self.pq, Tokenizer = prepare.load_dependencies()
            import pyarrow as pa
            from tokenizers import models, pre_tokenizers
        except (ImportError, RuntimeError):
            self.skipTest("Install reproduction/requirements.txt for the real reconstruction fixture")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "fixture_repo"
        self.root.mkdir()
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.output = self.root / "nested/new/prepared"
        (self.root / "tokenizer").mkdir()
        (self.root / "reproduction/selections").mkdir(parents=True)
        self.tokenizer_path = self.root / "tokenizer/tokenizer.json"
        vocab = {"<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3,
                 "alpha": 4, "beta": 5, "gamma": 6, "δ": 7}
        vocab.update({f"unused_{i}": i for i in range(8, 8192)})
        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer.save(str(self.tokenizer_path))
        self.texts = ["alpha beta", "gamma", "alpha δ", "beta gamma alpha"]
        source_path = self.raw / "tiny.parquet"
        self.pq.write_table(pa.table({"text": self.texts}), source_path)
        components = []
        for split, selected in [("train", [2, 0]), ("dev", [3]), ("holdout", [1, 2])]:
            folder = "small_holdout" if split == "holdout" else "small_corpus"
            records, flat, boundaries, metadata = [], [], [], []
            for index, row in enumerate(selected):
                text = self.texts[row]
                ids = tokenizer.encode(text, add_special_tokens=False).ids + [3]
                record = {"source": "tiny", "row": row, "length": len(ids)}
                boundaries.append([len(flat), len(flat) + len(ids)])
                flat.extend(ids)
                if split == "holdout":
                    record["metadata"] = {
                        "doc_id": f"doc-{row}", "domain_group": f"domain-{row}",
                        "text": None, "original_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    }
                    metadata.append(json.dumps(dict(record["metadata"], text=text), ensure_ascii=False) + "\n")
                records.append(record)
            selection = Path("reproduction/selections") / f"small_{split}.json.gz"
            with gzip.open(self.root / selection, "wt", encoding="utf-8") as stream:
                json.dump(records, stream, ensure_ascii=False)
            expected = self.root / "expected" / split
            expected.mkdir(parents=True)
            self.np.save(expected / f"tokens_{split}.npy", self.np.array(flat, dtype="<u2"), allow_pickle=False)
            self.np.save(expected / f"docs_{split}.npy", self.np.array(boundaries, dtype="<i8"), allow_pickle=False)
            if split == "holdout":
                (expected / "documents_holdout.jsonl").write_text("".join(metadata), encoding="utf-8", newline="\n")
                self.np.save(expected / "eval_offsets.npy", self.np.array(boundaries, dtype="<i8")[:, 0], allow_pickle=False)
                self.np.save(expected / "eval_document_ids.npy", self.np.arange(len(records), dtype="<i8"), allow_pickle=False)
            component = {
                "name": f"small_{split}", "study": "small", "split": split, "folder": folder,
                "documents": len(records), "tokens": len(flat), "sources": ["tiny"],
                "selection": str(selection), "selection_sha256": prepare.sha(self.root / selection),
                "files_sha256": {p.name: prepare.sha(p) for p in expected.iterdir()},
            }
            if split == "holdout":
                component["metadata"] = "documents_holdout.jsonl"
            components.append(component)
        self.plan = {
            "tokenizer_sha256": prepare.sha(self.tokenizer_path), "components": components,
            "sources": {"tiny": {"filename": source_path.name, "sha256": prepare.sha(source_path),
                                  "bytes": source_path.stat().st_size, "url": "https://example.invalid/tiny.parquet"}},
        }
        self.plan_path = self.root / "reproduction/data_plan.json"
        self.save_plan()
        self.root_patch = patch.object(prepare, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def save_plan(self):
        self.plan_path.write_text(json.dumps(self.plan), encoding="utf-8")

    def run_main(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()), patch.object(prepare.subprocess, "run") as download:
            prepare.main(["--study", "small", "--raw", str(self.raw), "--output", str(self.output), *extra])
        download.assert_not_called()

    def test_reconstructs_all_payloads_and_order_from_arbitrary_cwd(self):
        original_cwd = Path.cwd()
        try:
            os.chdir(self.raw)
            self.run_main()
        finally:
            os.chdir(original_cwd)
        self.assertFalse(self.output.with_name(self.output.name + ".partial").exists())
        self.assertEqual(json.loads((self.output / "reconstruction.json").read_text())["status"], "complete")
        for component in self.plan["components"]:
            for name, digest in component["files_sha256"].items():
                self.assertEqual(prepare.sha(self.output / component["folder"] / name), digest)
        tokens = self.np.load(self.output / "small_corpus/tokens_train.npy", allow_pickle=False)
        self.assertEqual(tokens.tolist(), [4, 7, 3, 4, 5, 3])

    def test_verify_only_does_not_import_data_dependencies(self):
        self.run_main()
        with patch.object(prepare, "load_dependencies", side_effect=AssertionError("must use stdlib only")), \
                patch.object(prepare, "exclusive_lock", side_effect=AssertionError("verification is read-only")):
            self.run_main("--verify-only")

    def test_existing_output_is_never_overwritten(self):
        self.run_main()
        before = {str(p.relative_to(self.output)): prepare.sha(p) for p in self.output.rglob("*") if p.is_file()}
        with self.assertRaisesRegex(ValueError, "Output already exists"):
            self.run_main()
        after = {str(p.relative_to(self.output)): prepare.sha(p) for p in self.output.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_output_hash_failure_preserves_partial_and_does_not_commit(self):
        self.plan["components"][0]["files_sha256"]["tokens_train.npy"] = "0" * 64
        self.save_plan()
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.run_main()
        partial = self.output.with_name(self.output.name + ".partial")
        self.assertFalse(self.output.exists())
        self.assertTrue((partial / "small_corpus/tokens_train.npy").is_file())
        with self.assertRaisesRegex(ValueError, "Partial output exists"):
            self.run_main()

    def test_changed_selection_is_rejected_before_encoding(self):
        self.plan["components"][0]["selection_sha256"] = "0" * 64
        self.save_plan()
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.run_main()
        self.assertFalse(self.output.exists())

    def test_verify_only_rejects_changed_prepared_bytes(self):
        self.run_main()
        path = self.output / "small_corpus/tokens_train.npy"
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.run_main("--verify-only")

    def test_second_preparation_process_is_refused_before_sources_or_output(self):
        lock_path = self.output.parent / f".{self.output.name}.prepare.lock"
        with prepare.exclusive_lock(lock_path):
            result = subprocess.run([
                sys.executable, "-B", "-c", IMPORT_PREPARE +
                "p.ROOT=Path(sys.argv[2]);p.main(sys.argv[3:])", str(SCRIPT), str(self.root),
                "--study", "small", "--raw", str(self.raw), "--output", str(self.output),
            ], capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Another preparation or download holds", result.stderr)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_name(self.output.name + ".partial").exists())
        # Retrying after the first process exits can reuse the persistent lock file.
        self.run_main()


if __name__ == "__main__":
    unittest.main()
