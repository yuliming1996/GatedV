# References and implementation provenance

The reference runtime derives from this project's earlier Transformer
experiments. File-level provenance is recorded beside the code and in the
release manifest. The `code/` runtime and the GLU follow-up runtime use distinct
checkpoint identities; neither silently adopts historical checkpoints.

The GLU value operator follows Zehao Wang's
[GLU Attention Improve Transformer](https://arxiv.org/abs/2507.00022v2).
The reference implementation inspected during development was
[WangZehaoAI/GLU-Attention](https://github.com/WangZehaoAI/GLU-Attention), commit
`e6986495b96d484894b1f780ac22b6fc2f67cebd`, file `wzh/transformer.py`.
The operator was implemented independently in this project's common backbone;
the upstream source is not vendored. Retaining six attention heads here requires
a nearby width of 258 for the reduced GLU control, rather than exact matching
at 256.

Value-side nonlinearities and attention gates also appear in earlier work.
The manuscript discusses these relationships and cites Transformer, RMSNorm,
RoPE, SwiGLU, GAU, Mega and other relevant architectures in
[references.bib](paper/references.bib).

Experiments used prepared FineWeb-Edu corpora. This repository contains metadata
and numerical results, plus the original [WikiText-trained tokenizer](tokenizer/README.md) with source attribution and historical license notes; source web text and tokenized training corpora are not included. Dataset provenance and preparation limits are described in
[the data documentation](docs/data.md) and [results/datasets.json](results/datasets.json).
The [reconstruction guide](reproduction/README.md) attributes the fixed FineWeb-Edu source revision and provides upstream links; frozen row selections contain no source document text. Dataset and dependency licenses are not replaced by this repository's MIT license.
