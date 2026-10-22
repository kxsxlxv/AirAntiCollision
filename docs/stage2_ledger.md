# Stage 2 experiment ledger

2026-10-22: User approved the Stage 2 handoff. Work starts on
`stage2-collision-research`, from current main. The handoff contains a pre-existing
user edit; preserve it. Weekly account meter at start: 45% used, 10080-minute
window, reset Unix 1793083556. Stop at 75% used; reserve capacity for tests and PR.
This account-wide delta includes any concurrent usage.

Initial hypotheses, before experiments:

- Full Bauer geometry may resolve off-axis failures of image-plane intercepts
  under its calibrated, fixed-attitude/constant-shape assumptions. Unknown focal
  length and ownship orientation remain missing measurements on NPS.
- Published image-region and looming cues can be reproduced as observables, but
  their apparent-risk semantics need explicit separation from physical CPA.
- Better persistent association should improve evidence continuity; predicted
  tracker boxes must not create new scale observations.
- Detector B may improve tiny fixed-wing recall, with higher input/runtime cost.
  Its training independence is unverified. No held-out detector-B claim allowed.

First checkpoint: verify primary equations and annotation/model contracts, freeze
video-level splits and an executable plan, then commit before estimator redesign.
