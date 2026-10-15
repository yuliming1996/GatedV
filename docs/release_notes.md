# Release snapshot: 15 October 2026

This is a standalone publication copy of the GatedV project. It contains the
current manuscript, the main numerical implementation, the GLU comparison
implementation, experiment recipes and compact result evidence. The research
workspace and the running experiment queue were not changed by packaging.

## Included and omitted

The two scientific runtimes are preserved byte for byte. The GLU shape-inspection
script is adapted to read the actual selection recipes. The result verifier is
extended to check the newly exported GLU snapshot. These changes are recorded in
the [code manifest](../code/source_manifest.json) and
[result provenance](../results/packaging_provenance.json).

Duplicate drafts, superseded figures, temporary test programs, private-machine
launchers, process monitoring and recovery logs, caches, and LaTeX intermediates
are omitted. The numerical evidence retains both positive and negative outcomes.
The original tokenizer was subsequently added byte for byte with a verified historical hash; see [tokenizer](../tokenizer/README.md). Raw corpora and checkpoints are not included; see [data](data.md).

A subsequent local update adds `tools/prepare_data.py` and `reproduction/`: pinned source URLs, accepted-row selections, and historical output hashes for the main small/medium data. The tool reconstructs accepted payloads; it does not rerun historical exclusion discovery. See the preparation guide for tested scope.

The paper is the author's 13 October draft. The GLU/lower-LR results are a
separate **5/16 development-set snapshot** observed at 08:40 Beijing time on
15 October; the fifth endpoint's audit finished at 08:41:15. The remaining
trajectories are not represented as final results. Review the manuscript's
learning-rate and GLU discussion when the new comparison is complete.

## Portable reproduction entrypoint

`tools/reproduce.py` now drives data reconstruction, the complete small or medium
paired confirmation cohort, holdout evaluation and statistical analysis. The
scientific runtime and published settings are unchanged. `--study smoke` is a
tiny synthetic CPU check; full studies keep all original seeds and updates.
`requirements-reproduce.txt` installs the dependencies for the whole workflow.

Historical source paths in three provenance JSON files have explicit
`historical://` identifiers, while runtime tokenizer references point inside the
repository. Numerical values and original source hashes were preserved; current
packaged hashes have been updated and labelled as adaptations rather than byte
copies. [path_portability.json](path_portability.json) records the exact scope.
Source identifiers elsewhere in evidence ledgers remain provenance, not files
read by training or reconstruction. No author's workspace directory is required.

The public tools have bounded tests for data integrity, output protection,
cohort completeness and interruption behavior. These tests are retained because
they check the public reproduction interface, rather than temporary research
experiments. The active Spark queue and its frozen sources were not changed.

## Validation

[release_validation.json](release_validation.json) records the original packaging checks.
[portability_validation.json](portability_validation.json) records the subsequent
clean-environment reproduction checks and their scope. The code passed bounded CPU execution checks, the numerical results were
recalculated, the manuscript compiled in a clean temporary directory, and the
figures were reproduced from the packaged curve data. These checks do not rerun
the scientific training or certify a new CUDA environment.

[`release_manifest.json`](../release_manifest.json) records the size and SHA-256
of each packaged file, excluding the manifest itself and Git internals. Run:

```sh
PYTHONDONTWRITEBYTECODE=1 python -B tools/verify_release.py
```

This inventory describes this snapshot. Deliberate edits will change its hashes;
review those changes and record a new inventory before treating a later version
as another verified release. Regenerating graphics with a different Matplotlib
version can also change file metadata while preserving the plotted values.

## Public repository

The public repository is [yuliming1996/GatedV](https://github.com/yuliming1996/GatedV),
with `main` as its default branch. Clone it with:

```sh
git clone https://github.com/yuliming1996/GatedV.git
cd GatedV
```

The manuscript and `CITATION.cff` link to this repository. Adding this link does
not change the manuscript's scientific content. A fresh temporary build after
the link addition produced 13 pages with no missing citations or cross-references;
the existing underfull bibliography line remains. The code uses the MIT license
selected by the author. No DOI or arXiv identifier is assigned to this snapshot.
