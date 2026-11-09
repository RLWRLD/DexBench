# Leaderboard submission guidelines

Submit a reproducible policy evaluation with the original episode results, **complete
rollout trajectories for successes and failures**, and camera videos. Use the official
[evaluation protocol](Eval_Protocol.md) and frozen rows. This guide defines the submission
contract. The release owner must announce the accepted release, suite, dataset revisions
and submission review process before accepting ranked entries. Submissions open only
when the release announcement confirms public access to the evaluator, data and
leaderboard repository and provides reviewed evaluation reference pins.

## Submission eligibility

Only **standard-data generalist** submissions are accepted.

- Additional DexBench training uses only the release's designated official training
  root(s) from [RLWRLD/DexBench-datasets](https://huggingface.co/datasets/RLWRLD/DexBench-datasets).
  The release lists the allowed repository revision and per-task or multitask roots;
  the leaderboard checks declared dataset URIs and revisions against that list.
  Ordinary resizing, normalization and online augmentation are allowed.
  No additional demonstrations, synthetic trajectories or simulator interaction may
  be used for DexBench training.
- Use one immutable checkpoint and declared inference system across all submitted
  tasks. Task instructions may change; task-specific checkpoints or adapters are
  not eligible. The validator requires the same checkpoint hash on every task.

Submissions may start from pretrained models. Disclose their immutable revisions,
known pretraining sources and any known benchmark overlap. Standard-data does not mean
identical pretraining. Name all planners, routers, adapters and ensembles; a server
endpoint alone is not a model identity. Hash a manifest of all weight files when a
checkpoint consists of several files, and retain that manifest for artifact review.
Disclose code and weight availability in the method card.

A short method card is required; a paper is optional. Method revisions and checkpoints
of the same model remain linked entries with retained history. Novelty or beating an
existing entry is not required. The operator may consolidate redundant revisions.

## Scores and coverage

- The primary table uses **v1**, the fixed 60 rows per cell. **v1-plus** uses all 120;
  report it separately. A v1-plus execution can also be scored on its pinned v1 subset.
- Each task uses its declared cells, including physical cells where present. Its score
  is their unweighted mean. `Train` means evaluation starts in the training distribution,
  not evaluating the demonstrations used to train the policy.
- Overall is the unweighted mean of complete task scores in the accepted suite. All tasks
  in the accepted release are required. Never pool episodes across tasks.
- Every submission must complete every required cell and row of every release task.
  Keep `k/n` and the official equal-tailed 95% Beta(k+1, n-k+1) interval per cell.
  Sensitivity and challenge scores remain separate; do not average interval endpoints
  to invent an Overall interval.
- Compare only the same accepted release, frozen sets and evaluation settings. v1 and
  v1-plus specify evaluation size, not the simulator/asset release. Record the code
  commit, evaluator image digest, manifest and frozen hashes. The parent Git commit and
  its assets tree identify assets; do not create another asset lock file.

The validator uses the tasks in the operator-selected benchmark manifest as the suite.
It recalculates the submitted model alone; missing baseline runs do not affect a submission.
Pilots, trial models and settings with deviations remain outside the ranked submission
path. Outside models run as `community`; a maintainer-reviewed registry entry is required
before publication as a submitted model in the ranked table.

## Evaluation conduct

Use only the approved policy observation contract: rendered camera inputs, robot
proprioception and task instruction. Declare the cameras actually consumed. Scene state,
object ground-truth poses, ground-truth success and evaluation row IDs/seeds are not
policy inputs. State recorded for auditing is not permission to read it during inference.

Do not change the embodiment, physics, cameras, controller, success rule, cap, or the
task's execution schedule. The GA schedule is selected by the runner: policy targets
at 30 Hz with linear interpolation on the 60 Hz control step. The execution horizon
comes from the protocol. Do not implement another interpolation layer in the server.
Declare action normalization, denoising, seeds, temporal ensembling and planning settings;
hold them fixed across cells. Distinguish simulated control frequency from wall-clock
model inference latency. Report GPU/count/VRAM, inference latency and evaluation time;
hardware eligibility and resource ceilings belong to the announced release, not to a
number guessed from another benchmark.

Reset policy state, cached chunks, recurrent memory and server sessions at every episode.
Cross-episode adaptation or memory requires a separately reviewed experimental setting.
No human intervention, row-specific trajectories, checkpoint selection on scored rows,
or best-of-N selection. Preserve every prescribed rollout, including failures and policy
errors. Infrastructure retries require operator approval and retain the failed attempt,
the affected row IDs and the reason. The current automated submission validator accepts
completed, non-overlapping runs only; salvaged attempts require manual review, not deleting
records to pass validation. An inference crash without a complete trajectory cannot enter
the automatic ranked path. Record its failure and resolve it through the review process.

Evaluation conditions are released through
[RLWRLD/DexBench-eval](https://huggingface.co/datasets/RLWRLD/DexBench-eval).
Access opens with the benchmark release; a repository URL alone does not imply public
access. Use the release's development conditions for tuning and its frozen rows for scoring.

## Required trajectory evidence

Run `scripts/evaluation/run_benchmark_cell.py` with **`--submission`**. This requires
dataset recording, camera videos and a checkpoint label and enables Arena's existing
`environment_builder.record_trajectories` terms. These terms record episode identity,
including when parallel environments finish out of order. They use the existing HDF5
format; no historical file is rewritten. Files lacking episode identity must be rerun; do not invent identity
from `demo_N` export order or fabricate missing states from actions.

Each HDF5 contains `data/demo_N/` with the following native recorder fields:

| Field | Meaning and alignment |
| --- | --- |
| `episode_id/env_id`, `episode_id/episode_in_env` | One integer each. Join to the result using Run name and rebuild index as well. `demo_N` is only an export name. |
| `initial_state` | One scene state immediately after reset, using the simulator's native state tree. |
| `obs` | T pre-step policy-group observations, including non-image proprioception. Preserve the original array/tree. |
| `actions` | T actions passed to the environment at control steps, after the scheduler's interpolation. Not the model's entire unexecuted predicted chunk. |
| `processed_actions` | T processed commands/robot targets in the recorded joint contract. Preserve additional recorded action fields too. |
| `states` | T post-step scene states: articulation root pose/velocity and joint position/velocity, rigid-object states, and other native state groups supported by the scene. |
| `success` | The recorded terminal verdict, matching the episode JSON. Preserve all other metric/termination fields already recorded. |
| `num_samples` attribute | T, equal to the episode result's `episode_length`. |

Keep the original HDF5 metadata, including `env_args` and joint-column names where the
embodiment writes them. Positions are metres, velocities metres/second, joint angles
radians, angular velocities radians/second, and quaternions **xyzw**. Scene root positions
are relative to the environment origin (`get_state(is_relative=True)`). Preserve native
specialized state representations; this is not a promise of bitwise simulator replay or
a complete serialization of hidden solver/particle state.

The control period is `evaluation_settings.control.sim_dt * decimation` in each episode
result. For index i, pre-step observation/action time is i·dt, and post-step state time
is (i+1)·dt; the initial state is at zero. The validation report includes this period and
the mapping to the original frozen row. Simulator substeps and policy inference calls
must not be confused with these control-step samples.

Camera frames remain in the original MP4 files by default; they are not included in
`obs`. Include the designated left head/ego view, **`robot_pov_left_cam_rgb`**
(the policy input **`camera_ego_left`**), for **every episode, successes and failures**.
Keep its original path relative to the run directory:
`arena/<Run>/robot-cam-rebuild<r>-env<e>-robot_pov_left_cam_rgb-episode-<n>.mp4`.
The Run, rebuild index, environment ID and episode-in-environment number must match
that episode's result and trajectory identity. List every required video and its SHA-256
in `runs[].videos`; missing, unlisted or empty required videos are refused.
Other cameras are optional and need not cover every episode. Confirm required coverage
and declare any additional camera/episode coverage in the method card. Complete
state/action trajectories remain required for every rollout. MP4 is visual audit evidence, not a lossless reconstruction
of network inputs. If a method needs pixel-exact input auditing, additionally retain
raw camera observations through the existing `record_dataset_images` config and document
that choice. Additional model-predicted chunks or network request traces are welcome;
label their clock/alignment separately from executed actions.

The validator checks trajectory coverage, unique episode joins, step counts, finite
numeric arrays, matching state layouts and success verdicts. Every submitted row's
evaluation signature must match the release's operator-reviewed reference, including
physics/solver settings, termination parameters, reset events and success criteria.
Missing or stale references and missing success criteria are refused. It checks video hashes,
non-empty files and designated-camera coverage by episode identity. Video content and
actual frame/episode alignment remain operator review. File hashes
and consistent metadata detect corruption, not fabricated evaluations. Published scores
remain the submitter's reported results after evidence review.

## Run, package and validate

Use a clean accepted Git checkout with the package installed and a running model server.
The validator requires recorded `code.dirty: false` for the run and every episode; dirty
or unknown checkout state is refused. Keep generated files outside tracked source paths.
Download the release's evaluation conditions after access is available:

```bash
./scripts/download_datasets.sh --eval-conditions --repo-id RLWRLD/DexBench-eval
```

Without a matching repository entry in `reproducibility/datasets.lock.json`, this command
uses the repository's default branch. The frozen hashes still check the downloaded rows.
A repository-revision lock is a separate release change; do not invent a pin locally.
For example, the registered GR00T adapter on one Manufacturing - Task 1 cell:

```bash
/isaac-sim/python.sh scripts/evaluation/run_benchmark_cell.py \
    --task m1 --cell train --model gr00t --version v1 --submission \
    --checkpoint checkpoint-sha256:YOUR_ACTUAL_DIGEST \
    --output outputs/submission/runs/m1/train -- --viz none
```

Run the other required cells into separate directories. `--checkpoint` is an identity
label; it does not load or hash server weights. Supply the actual immutable checkpoint
URI and SHA-256 in the submission and document which weights the evaluated server loaded.
For outside models pass `--checkpoint-sha256` as well; keep that digest and the recorded
checkpoint label identical on every task.
Start and configure servers using the model's `note` in `benchmark/models.py`.

Keep each run directory intact (`run_manifest.json`, `composed_experiment.yaml`, `arena/`
and any generated row subsets). Use the [JSON template](leaderboard/submission.example.json)
and [method card](leaderboard/method_card.template.md). Include `model_entry.json` and its
SHA-256 via `submission.model_entry`: use `{"key": "gr00t"}` for a release baseline, or
the exact config object passed through `--model-entry` for an outside model. The report
also records that object. The example is a fill-in template,
not a valid evaluation: its zero hashes and one-cell result will be rejected.
Run-local artifact paths are relative to the run directory; run directories and the method
card are relative to the submission JSON's directory. All paths must stay inside the bundle.

Install the offline tools' dependencies with
`python3 -m pip install numpy h5py scipy PyYAML 'jsonschema>=4'`.
They do not launch Kit. Generate each run's inventory, then put the printed object
in the submission's `runs` list and make its `directory` relative to the bundle root:

```bash
python3 scripts/evaluation/validate_leaderboard_submission.py inventory \
    --run outputs/submission/runs/m1/train
python3 scripts/evaluation/validate_leaderboard_submission.py validate \
    outputs/submission/submission.json --conditions-root . \
    --json outputs/submission/validation.json
```

Inventory hashes files and identifies the standard rebuild filenames; validation joins
episodes by recorded identity. For older filenames, provide an explicit reviewed file-to-Run
and rebuild inventory. No archive is extracted or URL fetched and no participant code runs.
The operator selects the trusted manifest with `--manifest` when needed; participants must
not narrow the task suite or change evaluation settings by supplying their own manifest.
`benchmark.evaluation_sha256` lists the release's per-task contract hashes. The recorded
manifest SHA-256 remains provenance and must agree between the bundle and its runs;
model registration or formatting changes in the trusted manifest do not invalidate it.

### Release references

The release owner publishes reviewed per-task reference pins before opening submissions.
Validation checks every observed row against those pins. A v1-only reference contains no
signatures for unobserved v1-plus rows; a release using those pins opens v1 submissions
only. Release preparation is managed separately from the participant submission workflow.

## Connecting a new policy

Use [Connect your policy](../scripts/evaluation/eval_launch/CONNECT_YOUR_POLICY.md)
and the existing `--model-entry` path. A config model supplies `key`, `label`, optional
`note`, `group: community`, and its client's `policy` block. Keep the same entry and
checkpoint across every task. This does not require editing `models.py` or regenerating
the official benchmark manifest. Policy clients use the existing transport/codec
extension points; keep custom client packages in the declared runtime image.

After reviewing the submission, maintainers add `models/<key>.json` in
`RLWRLD/DexBench-leaderboard`, binding the config entry to its checkpoint hash and label.
That registry grants `registered` status; participants cannot choose it through
`--model-entry`. Keys are immutable; a new checkpoint or entry uses a new key.
Registered models use the same scorer and ranked table as baselines, marked as submitted.
They remain optional for the official baseline completeness calculation. An incomplete
registered result has no complete score and cannot enter the submission leaderboard.
The official scorer accepts the reviewed registry with `--model-registry PATH_TO_MODELS`.

### Evaluating through the launcher

Use the launcher config's `submission: true`, one `model.checkpoint` and
`model.checkpoint_sha256`. The plan preserves these fields for every unit and all
backends forward `--submission` and `--checkpoint-sha256` to the cell runner. Use the
launcher's normal checkpoint path for the policy server; its hash is a separate field.
Include all release tasks and cells. Packaging combines the completed unit directories;
each directory keeps its own results, HDF5, videos and inventory.

Only completed, non-overlapping unit outputs are currently submittable automatically.
The launcher's `score_<task>.accepted.json` can select salvaged or retried rows for the
ordinary scorer, but is not an admission mechanism for leaderboard evidence. Launcher
outputs that depend on that selection are not yet submittable; preserve the failed
attempts and request manual review rather than deleting them or selecting a better run.

## Submission and review

Once submissions open, create a PR in the leaderboard repository,
[RLWRLD/DexBench-leaderboard](https://github.com/RLWRLD/DexBench-leaderboard), with metadata and the method card.
The leaderboard repository owns submission records independently of the Arena code
and its exported mirrors. When submissions open, fork `RLWRLD/DexBench-leaderboard`,
create a submission branch, and add `entry.json`, `submission.json`, `model_entry.json`, `method_card.md` and the local
validation report using that repository's announced directory layout. Open a pull request
from the fork to `RLWRLD/DexBench-leaderboard`, using the submission template and linking
the evidence bundle. Reviewers validate the evidence with the trusted evaluator checkout;
maintainers merge accepted records. Revisions use new entries and retain the previous
entry and review history. Source-code export jobs must not target this repository.
Put large trajectory/video bundles in durable artifact storage with access for reviewers,
not in Git or `assets/`. Include a bundle URL
and checksum in the PR. The local schema describes the downloaded bundle, not a storage
provider. Never put credentials in the PR. Public weights are encouraged; private access
to code/weights or an immutable serving image is acceptable for artifact review.

1. Automated validation checks metadata, hashes, original results, trajectories, official
   frozen rows and each row's pinned release evaluation signature; it recalculates the
   score and reports `validated`, with the reference-file hashes.
2. Reviewers check data eligibility, runtime settings, videos and reproducibility. Accepted
   self-reported entries receive **Reported** status.
3. Keep release, model revision, coverage and review status
   visible. Corrections retain their old entry and reason. Rules changes affecting scores
   require a new release or an explicitly documented correction applied consistently.

Announce the submission review turnaround target, per-team resubmission allowance and
appeal contact before opening submissions. Reviews audit the submitted artifacts without
running the submitted policy. The CLI does not upload, merge, publish or approve entries.
