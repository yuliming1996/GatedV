# Original experiment tokenizer

`tokenizer.json` is the original, byte-for-byte ByteLevel BPE tokenizer used by the GatedV experiments. Vocabulary: 8,192. Special IDs: `<pad>`=0, `<unk>`=1, `<bos>`=2, `<eos>`=3.

SHA-256: `2857e9c9880f86b2c87475db3431344398c7caaa3ee018b25b81e352869aa217` (550,412 bytes).

From the repository root:

```sh
python -m pip install -r tokenizer/requirements.txt
```

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
ids = tokenizer.encode("Hello, world!").ids
assert tokenizer.decode(ids) == "Hello, world!"
# For corpus preparation, append EOS at each document boundary explicitly.
document_ids = ids + [tokenizer.token_to_id("<eos>")]
```

The tokenizer does not automatically insert BOS/EOS, pad or truncate. Preserve document boundaries during corpus preparation. The training runtime consumes pretokenized NumPy arrays; it does not call this tokenizer. Copy this exact JSON into your prepared corpus directory as `tokenizer.json` to include it in the training identity.

## Provenance and source attribution

Trained once with Hugging Face Tokenizers 0.23.2 on the historical deduplicated WikiText-2 raw training text, with `min_frequency=2`; validation and test text were excluded from tokenizer fitting. The historical preprocessing used cross-split deduplication, so simply downloading WikiText train and retraining is not guaranteed to recreate these bytes. The original serialized artifact is provided to avoid retraining.

Source: [Salesforce/WikiText, fixed dataset revision](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/README.md); WikiText by Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher, *Pointer Sentinel Mixture Models*. Machine-readable hashes and training-input provenance are in `provenance.json`.

Historical source notes record CC BY-SA 3.0/GFDL in dataset metadata and a CC BY-SA 4.0 link in the dataset card. This discrepancy is retained, not resolved here. Upstream source terms and attribution are not replaced by the repository MIT code license. See [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), and [GFDL](https://www.gnu.org/licenses/fdl-1.3.html).

Bundling the tokenizer does not supply or reconstruct the original FineWeb-Edu train/dev/holdout payloads.
