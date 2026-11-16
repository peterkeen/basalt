# Diagnostics and measurement flags

Flags for measuring, debugging and testing the engine. All are off by default and are not settings.
Binary: **CLI** = `bin/basalt` one-shot (`--tokens IDS`), **serve** = `--serve` engine, **bench** = `basalt-bench`.

## One-shot generation

| Flag | Binary | Default | What it does |
|---|---|---|---|
| `--tokens`, `--tokens-file` | CLI | required | The prompt as token ids. |
| `--max-new` | CLI | 16 | Tokens to generate. |
| `--greedy` | CLI | on | Argmax decoding. |
| `--seed` | CLI | | Sample with this seed under `--top-k` 20, `--top-p` 0.95, `--temperature` 1.0. |
| `--stop-eos` | CLI | off | Stop at an end-of-turn token. |
| `--eos-ids` | engine | `248044,248046` | The end-of-turn tokens. |
| `--gpu-plan` | engine | | Print the `--gpu` choice as JSON and exit. |

## Timing and profiles

| Flag | Binary | What it does |
|---|---|---|
| `--stats` | CLI | Per-token wall time, host phases and the pool's breakdown. |
| `--no-pool` | CLI | Skip the CPU expert pool: the GPU-only floor. |
| `--no-capture` | CLI | Run layers directly, not as graphs. Needs `--no-pool`. |
| `--stage-timing` | CLI | Per-stage shares by kernel count. Needs `--no-capture`. |
| `--graph-only` | CLI | Time the per-layer `pre` graphs only. |
| `--gpu-only-full` | CLI | Replay every graph and the head, no pool: the per-token GPU floor. Not with a layer split. |
| `--gpu-stages` | CLI | Time each layer's three graphs, print, exit. Not with a layer split. |
| `--profile` | serve | GPU stage times per request. Not for A/B timing. |
| `--wait-stats` | serve | Host wait times in the decode loop. |
| `--decode-timing` | serve | One line per request on where decode time went. |
| `--draft-log FILE` | serve | One line per verify window (width, accepted, time). |

### HTTP request timings

The API's `timings` and `/v1/status` sum completed engine runs across solo/batch transitions and
reasoning or tool continuations. `cache_n` counts only reuse present when the request began, not
its own intermediate tokens. Prompt and decode times are engine intervals, not client wall time;
queueing, transport and other untimed work can make first output slower.

`predicted_n` counts engine-generated tokens. API `usage.completion_tokens` also counts any
reasoning-close or tool-opening tokens inserted by the server, so it can differ. Decode rates
use `predicted_n` and the summed decode time. Optional counters are omitted if any run did not
report them, rather than presenting a partial sum as the whole request.

## Dumps and speculation tests

| Flag | Binary | What it does |
|---|---|---|
| `--dump-residual`, `--dump-mixed`, `--dump-final-r` | CLI | Write residual or head inputs as f32, for bisecting. |
| `--dump-routing` | CLI | Routed expert ids and weights per layer and position. |
| `--spec-oracle FILE` | CLI | With `--spec N` on a pack without a drafter: draft from an expected continuation. The verify exactness test. |
| `--prefill-until P` | CLI | Batch the first P prompt positions; run the rest teacher-forced. |
| `--spec-corrupt N` | CLI | Replace every Nth draft with a wrong token, to test rejection. |
| `--diag-teacher` | CLI | DeepSeek with `--spec-oracle`: teacher-forces the file's tokens and counts differing picks. |
| `--diag-tier-frozen` | CLI, serve | DeepSeek: fixed placement (no copies or promotion). |
| `--diag-drafter-gpu DEV` | CLI, serve | DeepSeek: the card the drafter runs on. |
| `--force-zero-doorbell` | serve | Scheduling test on a card that cannot hold every expert. The output is wrong by design. |

## Experts and n-gram table

| Flag | Binary | Default | What it does |
|---|---|---|---|
| `--ple-io direct`, `ram` | engine | `direct` | Read the n-gram table unbuffered (`direct`) or lock it in RAM (`ram`). |
| `--ple-file PATH` | engine | unset (the model file) | Flash-Next: read `per_layer_token_embd.weight` from this GGUF instead of the model file's `gguf` section; the six `blk.1.ple_*` weights, dense weights and experts still come from the pack. IQ4_NL, Q5_0, Q8_0 or BF16, shape `[160, rows]`, other tensors may share the file, and the rows must reach every index the n-gram hash produces (320,001,446), so a table of another model is refused at start. A different table is a different model: the start-up line names the file (`PLE on, table N rows of FILE (TYPE)`) and the file joins the identity parked conversations are bound to. Give an absolute path in a config file. |
| `--ple-row-cache`, `--ple-inflight`, `--ple-delay-us` | engine | 1048576 rows, 256, 0 | Row cache size, outstanding reads, and injected read delay for tests. |

## Quality gate and bench

| Flag or key | Binary | Default | What it does |
|---|---|---|---|
| `--quality` | bench | 20 chunks | Run the PPL/KLD/top-1 gate through decode. [method](../precision/quality.md) |
| `--kld-base FILE` | bench | | The llama.cpp KL-divergence base. |
| `--quality-t N` | bench | 3 | Tokens per verify window in the gate. |
| `--ppl-base`, `--ppl-out` | serve | | Base file for `ppl=` requests, and where they write rows. |
| `ppl=K`, `ppl_t=T` | request keys | | Score chunk K; window width T. |

| Bench flag | Default | What it does |
|---|---|---|
| `-t conv,struct,prose` | | Decode fixtures, run in that order. |
| `-p DEPTHS` | `-p 16384` with no `-t` or `-p` | Prefill depths, e.g. `32k`. |
| `-r N` | 1 | Repetitions; above 1 the table shows mean and deviation. |
| `--seed N` | | Seed every decode request. |
| `--set "FLAG VALUE"` | | Replace or add engine arguments (repeatable). |
| `--engine PATH`, `--fixtures DIR` | `basalt` beside the bench, `data/bench` | Binary and fixtures. |
| `-o md` or `json` | md | Output format. |
| `--power-label W` | read from the GPU | The power limit to print. |
| `--logs DIR`, `--no-windows`, `--print-command` | | Logs, window stats, command printing. |
| `--hardware` | | Measure CPU, RAM and each GPU's link, then exit. |
| `--community`, `--submitter NAME`, `--community-file PATH` | bench | | The community preset and its row, which names the model ([community runs](../community.md)). |

`basalt-device` (in the build folder): `--list-devices`, `--link-rates`, `--selftest`.

## Experiment keys (`--exp`)

`--exp NAME=VALUE[,...]` sets experiment switches. They carry no stability promise and go away with their experiment.

- `concur9_logit_hash=1`: print a hash of each accepted row's head logits on stderr.
- `concur5_digest=1`: batch slots print a state digest per window.
- `spec_log=1`: with `--draft-log`, also write each row's target distribution.
- `host_ev=1`: print GPU time per window's verify, commit and draft, and the idle gap.
