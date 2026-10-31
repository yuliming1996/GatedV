"""Interleaved endpoint-copy forward/training-step timing on synthetic tokens.

The final checkpoints are read-only. Optimizer updates affect disposable copies.
Timing is observational unless you independently control host interference.
"""
import argparse
import math
from pathlib import Path
import statistics
import time
import torch
from common import MODEL_KEYS, configure, digest, environment, sync, write_json
from evaluate import executable_sources, verify_cohort
from model import ARMS, Config, LanguageModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--rounds', type=int, default=7)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repetitions', type=int, default=10)
    parser.add_argument('--reference', choices=ARMS, default='v_silu')
    parser.add_argument('--candidate', choices=ARMS, default='v_glu_full')
    args = parser.parse_args()
    if args.reference == args.candidate:
        raise ValueError('Reference and candidate must differ')
    if args.rounds < 1 or args.warmup < 0 or args.repetitions < 1:
        raise ValueError('Invalid timing budget')
    cohort_sha = digest(args.cohort)
    rows = verify_cohort(args.cohort.resolve())
    if digest(args.cohort) != cohort_sha:
        raise ValueError('Cohort changed during the endpoint gate')
    arms = (args.reference, args.candidate)
    jobs = {(cfg['seed'], cfg['arm']): (run, cfg, result) for run, cfg, result in rows if cfg['arm'] in arms}
    seeds = sorted({s for s, a in jobs})
    if not seeds or set(jobs) != {(s, a) for s in seeds for a in arms}:
        raise ValueError('Interleaved benchmark requires the complete declared reference/candidate pairs')
    device = configure(args.device); torch.set_num_threads(rows[0][1]['threads'])
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('Benchmark requires an empty output directory')
    binding = {'cohort_sha256': cohort_sha, 'source_sha256': digest(__file__),
               'executable_source_sha256': executable_sources(),
               'environment': environment(device), 'rounds': args.rounds, 'warmup': args.warmup,
               'repetitions': args.repetitions, 'synthetic_seed': 2026092902,
               'reference': args.reference, 'candidate': args.candidate}
    write_json(args.output / 'binding.json', binding)
    slots = []
    for round_id in range(args.rounds):
        for index in range(len(seeds)):
            seed = seeds[(index + round_id) % len(seeds)]
            for arm in (arms if (round_id + index) % 2 == 0 else arms[::-1]):
                run, cfg, result = jobs[seed, arm]; path = run / 'final.pt'; sha = digest(path)
                if sha != result['artifacts']['final.pt']:
                    raise ValueError('Endpoint changed after gate')
                state = torch.load(path, map_location='cpu', weights_only=False)
                model = LanguageModel(Config(**{k: cfg[k] for k in MODEL_KEYS})).to(device)
                model.load_state_dict(state['model']); optimizer = model.optimizer(cfg['lr'])
                optimizer.load_state_dict(state['optimizer']); del state
                tokens = torch.randint(0, cfg['vocab'], (cfg['batch'], cfg['context'] + 1),
                                       generator=torch.Generator().manual_seed(2026092902)).to(device)
                x, y = tokens[:, :-1], tokens[:, 1:]
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                row = {'round': round_id, 'seed': seed, 'arm': arm, 'checkpoint_sha256': sha}
                for mode in ('forward', 'train_step'):
                    timings = []
                    for repetition in range(args.warmup + args.repetitions):
                        sync(device); begun = time.perf_counter()
                        if mode == 'forward':
                            model.eval()
                            with torch.no_grad():
                                logits = model(x)
                        else:
                            model.train(); optimizer.zero_grad(set_to_none=True); logits = model(x)
                            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
                            if not torch.isfinite(loss):
                                raise FloatingPointError('Nonfinite benchmark loss')
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                            optimizer.step()
                        sync(device); duration = time.perf_counter() - begun
                        if not torch.isfinite(logits).all() or not math.isfinite(duration) or duration <= 0:
                            raise FloatingPointError('Nonfinite output or timing')
                        if repetition >= args.warmup:
                            timings.append(duration)
                    row[mode] = {'seconds': timings, 'median_ms': statistics.median(timings) * 1000}
                row['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
                if digest(path) != sha:
                    raise ValueError('Read-only endpoint changed')
                name = f'slot_{len(slots):03d}.json'; write_json(args.output / name, row)
                slots.append({'path': name, 'sha256': digest(args.output / name)})
                del model, optimizer, tokens, x, y, logits
                if 'loss' in locals():
                    del loss
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    if digest(args.cohort) != cohort_sha or executable_sources() != binding['executable_source_sha256']:
        raise ValueError('Cohort or executable source changed during benchmark')
    write_json(args.output / 'done.json', {'status': 'complete', 'binding': binding, 'slots': slots,
                                         'scope': 'Endpoint copies; synthetic fixed input; shared-host timing may include interference.'})
    print(f'Completed {len(slots)} interleaved timing slots.')


if __name__ == '__main__':
    main()
