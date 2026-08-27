# EW Smart Scan Strategy

A reproducible, pure-Python research prototype for scheduling an Electronic
Support (ES) receiver across discrete RF bands.  It is a simulation only: it
does not connect to radios, SDRs, or live spectrum data.

## Problem formulation

At each discrete dwell time `t`, the receiver selects exactly one frequency
band `a_t`.  A hidden collection of emitters may occupy zero or more bands.
The receiver observes only a noisy binary alert from its selected band;
unvisited bands remain unknown.  This makes scheduling a two-dimensional
search problem: *which frequency* to observe and *when* to revisit it.

The environment returns ground truth only to the evaluation harness.  Every
scheduler receives the same limited `Observation(time, band, alert, switched,
valid)` feedback.  `valid=False` represents a configured retune dead slot and
is not treated as a negative observation.

## Included models

`ew_env.py` supplies a configurable receiver plus independent emitters:

- `PeriodicEmitter`: fixed hop sequence, dwell duration, revisit period, and phase.
- `AgileEmitter`: pseudo-random hopping with an adjustable activity and hop rate.
- `BurstyEmitter`: Markov on/off transmissions with optional hops while active.

`EnvironmentConfig` controls number of bands, detection probability, false
alarm probability, switch cost, retune blind slots, slot duration, and random
seed.  A correct alert earns priority-weighted reward; false alarms, repeated
detections, and retunes are penalized.

## Schedulers

| File | Method | Assumption / use |
|---|---|---|
| `baselines.py` | Round-robin, random sweep, static priority | Open-loop references. |
| `baselines.py` | Discounted UCB and Thompson sampling | Restless-bandit heuristics using only alert history. |
| `baselines.py` | Predictive greedy | Lightweight supervised-style occupancy estimator. |
| `rl_scheduler.py` | NumPy DQN | Learns a scan policy from hit/miss history, age since visit, current band, and retune context. |
| `periodic_intercept.py` | Periodicity estimator and rendezvous scheduler | Learns cyclic on-windows from sparse observations, then attends the earliest predicted window. |

The DQN is deliberately lightweight: replay buffer, target network, and a
two-layer ReLU network are implemented with NumPy to avoid a framework
dependency.  Train it before evaluation using `train_dqn`.  It should be
treated as a baseline for policy comparison, not a validated operational
controller.

For a learned isolated periodic cycle, `RendezvousScheduler.worst_case_wait_bound`
reports the stated meeting-strategy bound `T - D + retune_slots`.  That bound
depends on a correct period/dwell estimate and does not extend to multiple
simultaneous emitters or estimator error.

## Run an experiment

Install the development dependencies, then run:

```powershell
python -m pip install -r requirements-dev.txt
python -m experiments.run_comparison --episodes 20 --horizon 240 --sweep
```

The command writes the following reproducible artifacts to `outputs/`:

- `summary.md` and `summary.csv`: averaged Pd, Pfa, interception rate, reward,
  prediction accuracy, intercept-time error, and per-emitter-type latency.
- `comparison.png`: policy comparison across the principal metrics.
- `sweep.csv` and `band_agility_sweep.png` (with `--sweep`): Pd as the number
  of bands and agile-emitter hop rate vary.

All scenario, training, and evaluation seeds are explicit.  Methods are
evaluated with common random-number episode seeds so random emitter behavior
is comparable between policies.

## Metrics

`metrics.py` computes the following per episode and averages them over runs:

- `Pd`: probability of detecting an occupied band conditional on a valid dwell.
- `Pfa`: false-alert probability conditional on a valid, unoccupied dwell.
- Intercept rate per receiver time unit and mean shaped reward.
- Occupancy-prediction accuracy for schedulers with an explicit predictor.
- Mean detection delay from the simulated emission-window start.
- First-intercept latency, including episode-end censoring, grouped by
  periodic, agile, and bursty emitter types.

## Test

```powershell
python -m unittest discover -s tests -v
```

The tests cover periodic dwell/quiet behavior, sensor behavior, retune blind
slots, scheduler mechanics, synthetic period recovery, DQN state construction,
and basic metric calculations.

## Structure

```text
ew_env.py                 Simulator and emitter processes
baselines.py              Open-loop, bandit, and predictor schedulers
rl_scheduler.py           Dependency-free DQN and training loop
periodic_intercept.py     Sparse periodicity estimator and rendezvous policy
metrics.py                Evaluation and aggregate metrics
experiments/              Comparison and sweep command
tests/                    Unit tests
api/simulate.py           Vercel serverless simulation endpoint
index.html                Hosted interactive demonstration
vercel.json               Vercel deployment configuration
```

## Vercel deployment

The repository is ready to deploy as a static interactive demonstration with a
Python Vercel Function at `/api/simulate`.  The hosted API constrains request
sizes so a browser cannot start an unbounded experiment.  It exposes the
open-loop, bandit, predictor, and periodic-rendezvous policies; the more
expensive DQN training workflow remains available through the local experiment
command.

```powershell
npx vercel
```

Complete the Vercel sign-in/linking prompt, then use `npx vercel --prod` for a
production deployment.  Vercel's Python runtime installs `requirements.txt`
for the function; `requirements-dev.txt` keeps plotting dependencies out of
the production function bundle.

## Limitations

The binary sensor model abstracts away modulation, propagation, SNR, antenna
patterns, receiver bandwidth overlap, and emitter behavior that adapts to the
receiver.  The reported results therefore compare scheduling strategies within
this synthetic model; they are not predictions of real-world intercept
performance.
