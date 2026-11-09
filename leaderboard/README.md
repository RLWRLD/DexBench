# Leaderboard submissions

The [submission guide](../docs/Leaderboard_Submission.md) defines eligibility, evidence
and review. When submissions open, send PRs to
[RLWRLD/DexBench-leaderboard](https://github.com/RLWRLD/DexBench-leaderboard). This directory contains reference
instructions for that workflow; it does not accept entries or configure a hosted board.

Once the release owner announces an accepted release and opens submissions in
`RLWRLD/DexBench-leaderboard`, fork it, create a submission branch, and follow that
repository's directory layout. Open a PR back to it with:

- `submission.json`, following the [template](../docs/leaderboard/submission.example.json).
- `model_entry.json`, the baseline key or exact config entry, with its hash in submission metadata.
- `method_card.md`, following the [method card](../docs/leaderboard/method_card.template.md).
- The local validation report and the artifact bundle URL with its archive SHA-256.

Use stable team/model/revision identifiers. Include the artifact bundle URL,
archive SHA-256 and validation report in the PR in `RLWRLD/DexBench-leaderboard`.
Use the submission PR template in that repository.
The bundle must contain byte-identical copies of the PR's metadata and method card,
plus the original run directories at the relative paths in `submission.json`.
Large HDF5, video, checkpoint and log files belong in artifact storage, not this tree.

Reviewers download the bundle, compare its metadata to the PR and run the validator
against their trusted release checkout, official frozen conditions and pinned evaluation
signatures. A local `validated` report is evidence for review, not an authorization to merge.
Accepted entries publish the submitter's scores with Reported status.
Publication and status changes are maintainer decisions. Keep historical entries when
adding corrections or a new model revision.
