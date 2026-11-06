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

Inference workspace has the same count dimensions as ordinary DP. Backward
retains the prefix states and propagates positive/negative coefficient adjoints
in signed log arithmetic. Differentiating coefficients, rather than log states,
preserves paths through zero coefficients at deterministic inputs. Cumulative
events seed all included coefficients, including zeros, so finite-event first
derivatives remain correct at probabilities zero and one.

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
