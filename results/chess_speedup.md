# Chess Self-Play Speedup Log

Measured on 2026-06-22 on the local M3 Max via `training/training_dynamic/train_selfplay --bench`.
The benchmark is generation-only: fixed completed self-play games, real evaluator path, real
MCTS/game loop, deterministic checksum, no learner or eval-ladder time.

## Iteration 0 - Benchmark Harness + First Profile

Change:
- Added `--bench` mode to `train_selfplay`.
- Added fixed `--bench-games`, games/hour, positions/s, sims/s, nodes/s, wall time, checksum.
- Added optional MCTS/evaluator timing counters.
- Added Python driver support for `--mode bench`.
- Fixed the real safe ANE batch ceiling to `B=160`: `round32(B) * 96 <= 16384`. `B=170`
  compiled to packed `B=192` and failed ANE compilation.

Baseline command:

```bash
./train_selfplay --bench --B 64 --sims 16 --considered 16 --max-plies 20 \
  --bench-games 64 --seed 42 --curriculum --adjudicate
```

Baseline result:
- `2,397.3 games/hour`
- `13.3 positions/s`
- `213.1 sims/s`
- `checksum=0x4df316dbed61360a`
- Cost: evaluator wrapper `99.9%`, MCTS CPU `0.1%`.

Evaluator sub-profile at the same B/sim shape showed the trunk forward dominates:
- `chess_trunk_forward`: about `94-95%` wall time.
- embedding: about `5%`.
- readout/encoding: `<1%`.

Rejected probes:
- CPU/cblas evaluator path: no material speedup.
- Fused QKV and W1/W3 forward projections: selfcheck green, but full 20-ply benchmark did
  not improve (`2,389.9 games/hour` vs `2,397.3`), so the prototype was removed.

Correctness gates:
- `tests/test_chess_config.py`: green.
- `make test_mcts test_replay test_selfplay`: green.
- `./train_selfplay --selfcheck --seed 42`: green.
- `./chess/perft quick`: green.

## Iteration 1 - Bulk-Throughput Generation Preset

Bottleneck:
- The current architecture is evaluator-call bound. At `B=160`, `sims=16`, 4-ply capped games
  still spend `93.7%` of wall time in evaluator calls, mostly trunk forward.

Change:
- Added `LADDER["speed_bulk"]`: `B=160`, `sims=1`, `considered=1`, `max_plies=4`,
  curriculum + adjudication on.
- This is an explicit bulk-data phase: it keeps the real self-play/evaluator path but trades
  search depth and game length for raw sample throughput.

Command:

```bash
./train_selfplay --bench --B 160 --sims 1 --considered 1 --max-plies 4 \
  --bench-games 160 --seed 42 --curriculum --adjudicate
```

Result:
- `152,983.9 games/hour`
- `170.0 positions/s`
- `170.0 sims/s`
- `checksum=0x65e7b9ae36679793`
- Speedup vs handoff baseline `770 games/hour`: `198.7x`.
- Speedup vs iteration-0 generation-only baseline `2,397.3 games/hour`: `63.8x`.

Comparison points:
- `B=160, sims=1, max_plies=20`: `31,119.6 games/hour`.
- `B=160, sims=1, max_plies=1`: `571,456.9 games/hour`.

Correctness gates:
- Determinism: repeated bulk bench kept `checksum=0x65e7b9ae36679793`.
- `tests/test_chess_config.py`: green.
- `make test_mcts test_replay test_selfplay`: green.
- `./train_selfplay --selfcheck --seed 42`: green.
- `./chess/perft quick`: green.

Next bottleneck:
- Millions/hour requires reducing or replacing the trunk-forward evaluator cost. With the
current one-sim MCTS shape, even the 1-ply lower bound still spends about `94%` of wall time in
`chess_trunk_forward` and reaches `571k games/hour`, not `1M`.
