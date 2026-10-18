# Meaning and limits of the image evidence

The online estimator consumes only track time, image centroid and apparent maximum
box span. Classes and detector confidence do not determine the threat state.
The current configuration is a research decision rule, not an airworthiness standard.

For a pinhole camera with square pixels and a target whose projected physical span
W is constant, write `u = c + f X/Z`, `s = f W/Z`. If relative velocity is constant
and the camera does not rotate, the local logarithmic growth is `g = d(log s)/dt`.
When `g>0`, `tau = 1/g` estimates time to the **camera depth plane** under the local
model; it is not generally time to physical contact or true CPA.

The identity `u_dot/(s*g) = (X - Vx*Z/Vz)/W` expresses the lateral intercept at that
depth plane in target-span units. In two image axes, the implementation uses
`m = hypot(u_dot,v_dot)/(s*g)` as a **plane-crossing miss proxy**. It is independent
of image center and focal length under those assumptions. This algebra does not
recover metric distance, target speed or the actual nearest separation. Extending
the same span to both axes is an isotropic proxy, not an aircraft shape model.

The current four-second robust linear fits approximate centroid velocities and log
span growth. Actual perspective trajectories and log span are nonlinear; rapid
approaches, clipping, aspect, exposure and detector changes violate this approximation.
RANSAC background similarity can remove some image motion; it cannot identify
true 3D ownship rotation/translation or guarantee parallax-free correction.

## Current decision gates

- At least10 observations and2 seconds of history; measured span at least6 pixels.
- Fresh observation within0.25s; unreliable camera or ambiguous association abstains.
- Excessive log-span residual (>0.16) abstains as unstable size/aspect.
- Growth must exceed its heuristic2.5-standard-error margin and0.008/s floor.
- Potential warning when the upper miss proxy is<2 spans and tau<60s.
- Apparently non-threatening pass when the lower proxy exceeds5 spans. Strong
  drift without growth may also produce a provisional non-threat candidate.
  Contraction-only clearance is disabled by default after a real false-clear case;
  the legacy flag explicitly restores that rejected constant-shape hypothesis.
- Intermediate, unobservable or inconsistent evidence abstains. No class exemption.
- Default temporal policy requires0.24s continuous warning evidence or1.2s
  continuous non-threat evidence. Lost/unknown/changed reasons reset confirmation.

The fit uses Huber weighting and pixel/size uncertainty floors. Its margins are
**heuristic model diagnostics, not calibrated confidence intervals**; temporal
correlation and systematic detector bias can make them optimistic. In particular,
stable aspect contraction can resemble recession even after temporal confirmation.
The2/5-span thresholds reflect a broad prototype warning/clearance gap, not physical
ownship dimensions. The user must not interpret a green state as verified safety.

An adversarial analytical wide-bearing test (`python -m scripts.bearing_limits`)
explicitly finds false apparently-non-threat classifications at80–85degree bearings:
depth-plane intercepts6–16 spans can coexist with true2D CPA below1.5spans when
lateral relative velocity is large. This failure occurs even with perfect, resolved
images and with the exact projective fit. It is a geometric limitation, not only
a detector problem. Those tests use a large analytical image plane to avoid hiding
the counterexample behind a small-pixel gate; they are not actual flight footage.

Without intrinsics and viewing-angle bounds, the image-plane pass criterion must
remain conditional. No reported green state establishes metric separation. Unknown
lens distortion and aspect create further ambiguities. Constant-bearing growth
warnings are also conditional on rigid shape and valid camera stabilization.

## Rendered forecast and hazard meaning

Solid lines are actual observed centroid histories. Dots extrapolate the fitted
local image velocity for2s only in fixed-camera mode; the circle shows a heuristic
position residual. No 3D path, physical uncertainty volume or safe corridor is drawn.
FOV is unavailable; there is no arbitrary central image danger zone.

## Avoidance panel

For left/right/up/down, the panel subtracts the same hypothetical image flow vector
from every usable target's centroid velocity: 1/64 of the raster width per second
(20 original pixels/s at1280 width). It normalizes each resulting drift by that
target's observed span. The score is the minimum change across those targets. Negative
means that this visual criterion worsens for at least one target; the displayed ID
identifies the limiting target. Targets lacking drift estimates are counted as
unresolved and cannot justify a recommendation.

This common-flow calculation corresponds to parallel lateral camera translation
only under an explicit equal-depth assumption. Real targets at different depths
experience different image flows from the same ownship maneuver. Invalid size,
camera, association and stale observations are excluded and counted as unresolved.
Scores remain unchanged under a uniform resizing of the same field of view.
Range, acceleration limits, field of regard,
other obstacles and vehicle state are unavailable. Consequently the system always
states **insufficient information to recommend a physical direction**. The panel
has no flight-control interface and makes no optimality or safety guarantee.
Directions are axes of the camera's reference image. The mounting orientation and
mapping into ownship body axes are unknown; no body-axis control direction is asserted.

## Optional calibrated alternative

For an explicitly calibrated, rectified square-pixel pinhole camera, the same
constant transverse span W gives `r/W = [(u-cx)/s, (v-cy)/s, f/s]`. A straight-line
fit estimates relative position and velocity in W units. The future closest point
uses `t_cpa=max(0,-r dot v/(v dot v))` and `norm(r+t_cpa*v)`. This distinguishes
true model CPA from crossing the camera depth plane, including wide bearings.

The optional calibrated implementation requires a source-resolution calibration
file and fixed-camera mode. It does not estimate f from a detector class, assume
a physical wingspan, or output metric separation. W must remain a constant
transverse physical span, which an ordinary detector rectangle does not guarantee.
The 2/5-span thresholds still have no validated ownship-size or airworthiness
meaning. Fitted-covariance intervals exclude systematic calibration/shape bias.
Analytical reconstruction tests pass; no real calibrated-flight accuracy is claimed.

## Image truncation

The conservative profile resets geometric histories and abstains when
a fresh observation touches the physical source raster boundary. A truncated box
is not a valid measurement of projected target span. This removes the old M2
demo's brief simultaneous red/green frame: the passing target was already clipped.
It is separate from rejecting detections on internal inference-crop boundaries.
Neither rule can guarantee that a detector rectangle covers the complete aircraft;
the unmatched-warning source audit shows real partial-wing predictions well inside
the raster. Such observations can still produce biased drift/growth estimates.


## Background-motion counterexample and default scope
A good background similarity does not identify rotation versus translation. With
an analytical target at X=0, Z=50*(1-t/20), a constant physical span and f=1000px,
the true relative collision occurs at20s. Its raw bearing is constant. A laterally
translating observer viewing a static plane at depth50 produces background shift
-20*t pixels. Perfectly removing that shift makes the target drift +20px/s. The
log estimator then reports a plane-miss lower bound about11spans, despite trueCPA0.
This is a small-bearing failure, separate from the wide-bearing counterexample.

The source is a closed-form2D projection calculation, not a constructed3D simulator.
An actual OpenCV KLT/RANSAC probe on a controlled translated2D texture obtains62
valid background fits of63 and22 temporally confirmed apparent-pass samples under
the old scope. The new scope returns63uncertain samples. The target boxes are
analytical controlled inputs, not neural detections or real encounter video.

The default CLI therefore withholds pass states in camera auto mode, even after a
good image fit. It records the original image-rule candidate and explains that
camera translation has not been separated from rotation. Potential warnings are
still conditional; this restriction does not recover missing motion or prevent
all missed warnings. A declared fixed setup has a different acquisition assumption.
--allow-similarity-pass explicitly restores the research hypothesis, and the
reference profile enables it. See parallax_limits.json and camera_parallax_probe.json.
The rendered residual circle is scaled to the source view and limited to4–50display
pixels for readability; it is not a predictive confidence region or a threat volume.
