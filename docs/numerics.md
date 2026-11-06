# Numerical behavior

## Dynamic programming

Each Bernoulli factor updates the polynomial with
`new[k] = old[k]*(1-p) + old[k-1]*p`, starting from `[1]`. There is one Python
loop over trials and tensor operations across counts and batches. The recurrence
is never clamped or renormalized. Constructing each new state separately keeps
earlier states safe for backward.

For 100 fair coins with float64 probabilities, PMF entry zero, CDF at zero, and SF
at 99 retain `2**-100`, approximately `7.89e-31`. DP can still underflow for
sufficiently small events; use the log-domain functions below for extended range.

Inside support, CDF takes O(B n min(n,k+1)) time and O(B min(n,k+1)) forward
workspace. SF uses failures and takes O(B n (n-k)) time and O(B (n-k)) workspace.
Outside support, exact constants remain connected to autograd with zero derivatives.

Subtraction in `1-p` is also subject to rounding: a small representable success
probability does not guarantee its complement is distinguishable from one. SF
uses original probabilities directly, avoiding a second complement `1-(1-p)`
that would otherwise erase tiny successes.

## Log-domain DP

`logpmf`, `logcdf`, and `logsf` compute in the log semiring with `logaddexp`;
they do not take logs of an underflowed ordinary PMF. For 2,000 fair coins the
log probability of zero or 2,000 successes remains `-2000*log(2)` even though
the corresponding float64 probability is zero. Cumulative queries truncate the
recurrence; SF uses original log-p and log-one-minus-p with the roles swapped.
Log recurrences use float64 internally and cast back to the promoted input dtype.
This avoids accumulating float32 rounding error across many trials, but the final
float32 logarithm still has float32 quantization error. Use float64 for precise logs.

Inference workspace has the same count dimensions as ordinary DP. The log-PMF
backward retains prefix states and propagates positive/negative coefficient
adjoints in signed log arithmetic. This preserves paths through zero coefficients
at deterministic inputs, rather than differentiating unreachable log states.

Cumulative-tail derivatives use the exact identities
`dCDF(k)/dp_i = -P(X without i = k)` and `dSF(k)/dp_i = P(X without i = k)`.
Prefix/suffix log convolutions compute these nonnegative densities, choosing the
smaller success/failure count. Dividing in log space by the event probability
gives the log-tail derivative without subtracting nearly equal adjoints. This
retains tiny derivatives even for near-one events. Forward need not save its
prefixes; backward builds the required density prefixes. Only the log-PMF's
general weighted-output backward needs signed arithmetic. Forward log tails are
capped at zero to remove positive roundoff; accumulation error near probability
one can still dominate a very small log. Shifted FFT's logarithmic complementation
is useful in that regime.

Impossible log events return `-inf`; their derivatives are undefined, and this
implementation uses zero gradients for those entries to make masked losses safe.
Consequently exponentiating a log-PMF and differentiating through impossible
entries is not a substitute for the ordinary PMF at deterministic boundaries.
Only first derivatives are supported by the dedicated log adjoint. `auto` stays
on log DP; it never logs an ordinary FFT result.

## FFT methods

Characteristic-function inversion uses direct complex products and an inverse
DFT, in frequency chunks of at most 128. Both FFT backends use float64/complex128
internals and cast the cleaned result back to the public dtype. The broader
accuracy sweep found float32 phase errors and accumulated off-support mass on
deterministic inputs; double precision keeps these cases within the acceptance
thresholds. This costs memory and can reduce speed on GPUs with slow double
arithmetic. Forward workspace is a constant number of O(B n) tensors; backward
can retain all characteristic-function factor chunks.

Tree convolution repeatedly multiplies adjacent `[1-p, p]` polynomials, carrying
odd nodes to the next level. Linear convolution is padded to a power-of-two
transform length. Intermediate coefficients are not clipped.

Both methods clamp negative roundoff in the final PMF and divide by its total
mass. Nonfinite results or nonpositive mass raise `RuntimeError`. There is no
epsilon injection or silent fallback. Cleanup enforces nonnegativity and
normalization within floating-point precision; it cannot recover relative
accuracy in rare events. Derivatives at clipped bins may differ from exact
probability derivatives. Use DP at deterministic boundaries.

Reference tests through n=512 target maximum absolute FFT error of `1e-5` for
float32 and `1e-10` for float64. These are tested acceptance thresholds, not
universal error bounds. Auto profiles must pass these thresholds on their exact
measured workload across five distribution regimes.

## Shifted FFT tails

The explicit `shift_fft` backend implements exponential tilting as described in
[Peres, Lee, and Keich (2020), Algorithm 1](https://arxiv.org/abs/2004.07429).
It uses the real-FFT convolution tree already provided by this package, rather
than their separate frugal transform optimization. It is an exact-distribution
algorithm subject to floating-point error, not a saddlepoint approximation.

For a tilt theta, each factor has log normalizer
`logaddexp(log(1-p), log(p)+theta)`. Its tilted success and failure coefficients
are computed separately; a rounded sigmoid complement cannot erase one. Bracket
expansion and bisection solve for a tilted mean near the requested count. Targets
at the effective support endpoints are moved half a count inward so the tilt
remains finite. Zero/one probabilities define the effective support and remain
exact deterministic factors. Guaranteed-success shifts are canceled analytically
in the normalizer before reconstruction.

Each batch row computes a favorable lower or upper tail. Unshifted logarithmic
weights decay away from its boundary, so FFT noise across the opposite tail is
not amplified. If the requested event is the complement, `log1p`/`expm1` give
stable logarithmic complementation, including log probabilities near zero.
The tilt solver is bracketed and bounded. Nonconvergence, nonfinite coefficients,
negative coefficients below -1e-10, mass error above 1e-8, or invalid reconstructed
tails raise RuntimeError. Bounded negative roundoff is clipped and normalized;
no epsilon floor or algorithm fallback is used.

The solver runs without autograd. The exact unshifted result is independent of
the tilt, but differentiating numerical FFT clipping can introduce errors. Our
explicit first-order backward uses nonnegative leave-one-out log densities,
including defined deterministic boundary derivatives. Prefix/suffix DP has
O(B n (1+min(k,n-1-k))) work/storage inside support; this is not an FFT-complexity
backward. The forward log tail supplies its normalization, so no cumulative
reference probability is recomputed in backward. It shares the
log-DP convention for impossible events and does not support second derivatives.
Threshold, dtype/device, and batch semantics match the existing count-tail API.
Ordinary cdf/sf wrappers exponentiate the returned log probability; that output
and its exponential derivative can underflow. The log APIs preserve the extended
range and are the intended interface for those events.

## Hybrid DP–FFT convolution

`method="fft_hybrid"` builds independent PMFs for blocks of `block_size` trials
using DP, then combines them with the balanced real-FFT tree. Partial final
blocks are padded with guaranteed failures; output is trimmed to n+1 counts.
Float64 internals, final cleanup, and output dtype semantics match `fft_tree`.
This changes constant factors and the initial work/memory balance. It does not
remove FFT rare-tail limitations or imply that every block size is faster.
The method is explicit; existing auto profiles are not broadened by these
experiments without fresh accuracy and speed evidence.

## Proportional events

The two subsets are independent. Compute their PMFs once each, then combine one
with the other's cumulative probabilities. No O(n²) joint probability table is
formed. Masks may differ per row; zero-probability dummy trials keep shapes uniform.

For `0 < beta < 1`, thresholds use the exact rational value of the supplied float,
avoiding floating-point `ceil`/`floor` errors. Right cumulative probabilities
are accumulated in reverse to avoid cancellation. The no-success outcome is
excluded. At beta zero only positive interesting counts enter the right event;
at beta one only positive uninteresting counts enter the left event.

At beta 0.5, equal proportions and empty outcomes enter neither event. Left and
right probabilities therefore need not sum to one.

## Supported execution

The supported target is eager PyTorch on Linux CPU/CUDA with float32/float64
computation. Half inputs promote to float32. Native half arithmetic, custom
CUDA/Triton kernels, graph capture, compilation guarantees, `vmap`, other
accelerators are outside this release. Probability
validation can synchronize CUDA and may be disabled for already-validated inputs.
