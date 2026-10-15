# Third-party sources

## Model and attention methods

The implementation derives from this project's Transformer experiments; file
provenance is retained in [code/source_manifest.json](code/source_manifest.json).
The [paper bibliography](paper/references.bib) cites Transformer, RMSNorm, RoPE,
SwiGLU, GAU, Mega and related attention methods.

The GLU operator follows Zehao Wang's
[GLU Attention Improve Transformer](https://arxiv.org/abs/2507.00022v2).
The authors' [implementation](https://github.com/WangZehaoAI/GLU-Attention) was
inspected at commit `e6986495b96d484894b1f780ac22b6fc2f67cebd`,
`wzh/transformer.py`. This repository independently implements the operator;
the upstream source is not vendored.

## Data and tokenizer

Training data comes from Hugging Face FineWeb-Edu, `sample/10BT`, revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`.
See its [dataset card and source terms](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/README.md).
The [data plan](reproduction/data_plan.json) records the pinned downloads and
accepted rows; raw source text is not bundled.

The original 8,192-token BPE was trained on deduplicated WikiText-2 raw training
text using Hugging Face Tokenizers 0.23.2. Validation and test text were excluded
from fitting. Source: [Salesforce/WikiText](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/README.md),
by Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher,
*Pointer Sentinel Mixture Models*. Hashes and source notes are retained in
[tokenizer/provenance.json](tokenizer/provenance.json).

Historical WikiText metadata lists CC BY-SA 3.0/GFDL, while its dataset card
links to CC BY-SA 4.0. The provenance record preserves that discrepancy without
resolving it. See [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/),
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) and
[GFDL](https://www.gnu.org/licenses/fdl-1.3.html).
Dataset and dependency terms are not replaced by this repository's MIT code license.
