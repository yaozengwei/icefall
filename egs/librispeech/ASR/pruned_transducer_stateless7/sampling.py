import os
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.utils.cpp_extension import load

VERBOSE = True


def _resolve(name):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), name)


try:
    import balanced_sampling_cpu
except ImportError:
    if VERBOSE:
        print("Falling back to JIT compiling balanced_sampling_cpu")
    balanced_sampling_cpu = load(
        name="balanced_sampling_cpu",
        sources=[
            _resolve("balanced_sampling_cpu.cpp"),
        ],
        verbose=VERBOSE,
    )

try:
    import balanced_sampling_cuda
except ImportError:
    if VERBOSE:
        print("Falling back to JIT compiling balanced_sampling_cuda")
    balanced_sampling_cuda = None
    if torch.cuda.is_available():
        balanced_sampling_cuda = load(
            name="balanced_sampling_cuda",
            sources=[
                _resolve("balanced_sampling_cuda.cpp"),
                _resolve("balanced_sampling_cuda_kernel.cu"),
            ],
            verbose=VERBOSE,
        )


def compute_count_indexes(
    counts_cumsum: torch.Tensor, max_count: int
) -> Tensor:
    """
     Returns a Tensor of torch.int32 containing the indexes (e.g. class indexes)
     that have nonzero counts, or -1 for padding.  The indexes will be indexes into
     the last dim of `counts`.
     Args:
         counts_cumsum: a Tensor with dtype torch.int32, of shape
             (size0, size1), containing the cumulative sum, over dim=1, of
              original counts counts >= 0, of type torch.int32;
             usually these will be zero or 1, but in any case
             for efficiency they should be fairly small numbers (the implementation
             has a loop on GPU)
      max_count: the largest number of counts that might appear in any given
             "row" of counts, i.e. the user asserts that:
             torch.all(counts.sum(dim=-1) <= max_count)
    Returns:
       a Tensor of shape (*, max_count), with the size of the final dim of `counts`
       replaced with `max_count`.  For as many counts as exist for each row,
       the final index into `counts` will be present; then -1's will be present.
    """
    assert counts_cumsum.ndim == 2
    assert counts_cumsum.dtype == torch.int32

    ans = torch.full(
        (counts_cumsum.shape[0], max_count),
        -1,
        dtype=torch.int32,
        device=counts_cumsum.device,
    )

    if counts_cumsum.is_cuda:
        if balanced_sampling_cuda is None:
            raise EnvironmentError("Failed to load native CUDA module")
        balanced_sampling_cuda.compute_count_indexes(counts_cumsum, ans)
    else:
        balanced_sampling_cpu.compute_count_indexes(counts_cumsum, ans)

    return ans


def zero_excess_class_counts(
    class_counts: torch.Tensor,
    class_tot_counts: torch.Tensor,
    frames_for_class: torch.Tensor,
    target_class_count: int,
):
    """
    This function modifies `class_counts` in-place, by zeroing some counts that
    were 1, to ensure that no class
    has more than `target_class_count` samples in total.  This is done pseudo-randomly,
     using the class counts themselves as the source of randomness.

    Args (all tensor arguments are of dtype=torch.int32):
       class_counts [this will be modified by this function by zeroing some elements]:
           a Tensor of shape (num_frames, num_classes),
           containing 1's where a class was sampled on that frame and 0 everywhere else.
       class_tot_counts: the total, over frames, of class_counts; gives
           the total counts for each class.  if class_tot_counts[c] > target_class_count,
           then the excess samples above target_class_count are to be removed
           by zeroing the entries in `class_counts`.
       frames_for_class: a Tensor of shape (num_classes,max_class_count), where
           frames_for_class[c][0:target_class_count[c]] contains the frame
           indexes that this class count is 1 on (the remaining elements are -1).
       target_class_count: the maximum allowed class count; we'll be zeroing some
           counts for classes that exceed this value.
    """
    assert class_counts.ndim == 2
    assert class_counts.dtype == torch.int32

    if class_counts.is_cuda:
        if balanced_sampling_cuda is None:
            raise EnvironmentError("Failed to load native CUDA module")
        balanced_sampling_cuda.zero_excess_class_counts(
            class_counts, class_tot_counts, frames_for_class, target_class_count
        )
    else:
        balanced_sampling_cpu.zero_excess_class_counts(
            class_counts, class_tot_counts, frames_for_class, target_class_count
        )


def compute_target_marginals(p: Tensor, K: int) -> Tensor:
    """
    This function, which is non-differentiable,
    returns the optimal marginal probabilities (in the sense of
    reducing variance of the sampled output) of the classes, if we are
    to sample exactly K distinct classes from a distribution over M > K
    items.  If the probabilities for one of the distributions are p_i,
    with 0 <= i < M, this will be min(1, beta p), where beta is chosen so that
    the sum of the resulting marginals is exactly K.

    Args:
       p: the input probabilities, of shape (*, M);
          these do not have to be properly normalized to sum to 1.
       K: the number of items to sample, 0 < K < M.
    Returns:
       Returns a Tensor `ans` of the same shape (*, M) and the same dtype and device
       as p.  If input_is_log == False, it will be an expression of the form:
            ans == (p * beta).clamp_(max=1.0)
       where beta is a tensor of shape (*, 1) with values >= K, that is not returned; and will satisfy:
             ans.sum(dim=1) == K
    """
    M = p.shape[-1]
    p_sorted, _ = torch.sort(p, dim=-1)

    p_sorted_cumsum = p_sorted.cumsum(dim=-1)

    # If want to compute, for each item in the batch, the number of items 0 <= N < K
    # such that N items of (p * beta) are greater than 1, with beta computed
    # such that the marginals sum to exactly K, i.e. that
    #     (p * beta).clamp_(max=1.0) == K,
    # OK, if (p * beta).clamp_(max=1.0) == K, and exactly 0 <= N < K items of (p * beta)
    # are greater than 1, then...
    #    N + beta_N * (p_sorted_cumsum[M - N - 1]) == K,
    # i.e. we add N for the items exactly equal to 1, plus the sum of the smallest (M - N)
    # items,
    # where beta_N is "the computed beta, assuming exactly N items of (p * beta) > 1."
    # So
    #    beta_N = (K - N) / (p_sorted_cumsum[M - N - 1]).        (eqn:1)
    # Now, beta_N is valid [in the sense that exactly N items of (p * beta_N)
    # are >1.0], if:
    #
    #   N == 0 or p_sorted[M - N] * beta_N > 1.0   # this condition will fail if N is too large.
    #       p_sorted[M - 1 - N] * beta_N <= 1.0    # this condition will fail if N is too small.
    #
    # ... because I am concerned about ties, I think we can compute just one condition.
    #  if we compute just the 2nd condition, i.e.:
    #       p_sorted[M - 1 - N] * beta_N <= 1.0            (eqn:2)
    # ... then the correct value of N equals the number of items in N=0,1,..K-1 for which this
    # fails (is false).
    # So if we compute:
    #   N = sum(p_sorted[M - 1 - N] * beta_N > 1.0),  [sum of boolean array]
    # the number of nonzero items should equal the correct value of N (with 0 <= N < K).
    # .. because, mathematically, this should always be false
    #
    # However, because in practice we are computing things in reverse order, the index
    # that we really want is not N but K-1-N, which we can get by summing the number
    # of items in N=0,1,2,...K-2 for which (eqn:2) succeeds, i.e.:
    #
    #  K1_minus_N = sum(p_sorted[M - 1 - N] * beta_N <= 1.0) [sum of boolean array]            (eqn:3)
    #
    # .. we exclude the value for N==K-1 because, mathematically, thsi should always be
    # true.

    # ... only including the elements for N < K,this should be the correct value of K-1-N.
    # The element of the sum for N == K should always be true.

    M = p.shape[-1]
    p_sorted_cumsum_part = p_sorted_cumsum[..., M - K :]  # shape: (..., K+1)
    p_sorted_part = p_sorted[..., M - K :]

    # the last dim p_sorted_cumsum_part corresponds to N==[K, K-1, K-2,..., 1, 0],
    # which is the number of classes excluded from the cumsum, so the value of
    # K-N along this array would equal [1, 2, ..., K].
    # torch.arange(K+1)
    K_minus_N = torch.arange(1, K + 1, device=p.device, dtype=p.dtype)

    beta_N = K_minus_N / p_sorted_cumsum_part  # (eqn:1) above

    condition = p_sorted_part * beta_N <= 1.0
    K_minus_N_chosen = torch.sum(
        condition[..., 1:].to(torch.int64), dim=-1, keepdim=True
    )  # (eqn:3)

    # print("K_minus_N_chosen = ", K_minus_N_chosen)
    # print("beta = ", beta_N)
    beta = torch.gather(beta_N, dim=-1, index=K_minus_N_chosen)

    return (p * beta).clamp_(max=1.0)


def sample_from_target_marginals(target_marginals: Tensor, K: int) -> Tensor:
    """
    Does systematic sampling to produce a sample of exactly K items from each
    distribution.
    Args:
        target_marginals: a Tensor of shape (num_frames, num_classes)
             such that 0 <= target_marginals <= 1 and
             target_marginals.sum(dim=1) == K.
            K: the number of samples to draw.
    Returns:
         a Tensor of integer (zero or one) counts, of type torch.int64
         and of shape (num_frames, num_classes), that
         will have exactly K values per frame set to 1 and the remaining
         values set to 0.
    """
    # Make sure that K is a power of 2.  This avoids certain potential errors
    # due to rounding effects, e.g. the possibility that (target_marginals * int_scale)
    # would get rounded up due to limited float precision, prior to conversion to
    # integer, leading to elements of int_marginals greater than int_scale, so potentially
    # duplicates of the same class.
    assert K & (K - 1) == 0, K
    # note: for testing and validation of the code, I used a small power here, 7 instead of
    # 30.
    int_scale = (
        2 ** 30
    ) // K  # the scale on 1.0 in the marginals, when we integerize.
    int_marginals = (target_marginals * int_scale).to(torch.int32)
    int_marginals_cumsum = int_marginals.cumsum(dim=-1)
    # excsum means exclusive cumsum
    int_marginals_excsum = int_marginals_cumsum - int_marginals

    # int_marginals_tot will be very close to int_scale * K == 2**30.
    # OK, mathematically, we are going to choose a random integer 0 <= r < int_scale,
    # and then the classes we choose are the classes m such that, for
    # some integer k,
    #     int_marginals_excsum[f,m] <= r+k*int_scale < int_marginals_cumsum[f,m].
    # If the total (over classes) of int_marginals were exactly equal to
    # K*int_scale, this procedure would give us exactly K chosen classes, with
    # no class repeated twice.
    #
    # But because the total of int_marginals may be slightly different from K*int_scale,
    # we need to be a bit careful when picking r, to make sure that
    # exactly K classes are picked.  The criterion comes down to:
    #  r >= 0,
    #  r + K*int_scale >= tot
    #  r + (K-1)*int_scale < tot
    # where `tot` is the total of the integerized marginals, i.e. int_marginals_cumsum[f,-1].
    # This means:
    #    max(0, tot - K*int_scale) <= r < tot - (K-1)*int_scale.
    # so we can use:
    # r = r_start + rand() % (1 + r_end - r_start), where
    # r_start = max(0, tot - K*int_scale)
    #   r_end = tot - (K-1)*int_scale.

    tot = int_marginals_cumsum[:, -1].unsqueeze(-1)
    r_start = (tot - (K * int_scale)).clamp_(min=0)
    r_end = tot - ((K - 1) * int_scale)

    num_frames = target_marginals.shape[0]
    r = (
        r_start
        + (
            (
                torch.randint(
                    low=0,
                    high=2 ** 63 - 1,
                    size=(num_frames, 1),
                    device=target_marginals.device,
                    dtype=torch.int64,
                )
                % (r_end - r_start)
            )
        ).to(torch.int32)
    )

    # the "+ K * int_scale" is to avoid the discontinuity in how integer division behaves when the
    # numerator becomes negative.
    # the reason for the "-1" is as follows...
    # We want a discontinuity between, say,
    #     (r+n*K - cumsum)==-1, and (r+n*K - cumsum == 0)
    # but because we are subtracting r from cumsum, the sign is flipped, so we want
    # a discontinuity between (cumsum-r) == n*K and (cumsum-r) == n*K+1, so we have
    # to subtract one because it's between n*K-1 and n*K that operator "//" gives a
    # discontinuity.
    #
    # cur_remainder == r -> (int_scale - 1) -> divide by int_scale -> 0
    # cur_remainder == r+1 -> int_scale -> divide by int_scale -> 1.
    cum_remainder = int_marginals_cumsum + (int_scale - 1) - r
    exc_remainder = int_marginals_excsum + (int_scale - 1) - r

    is_this_class = (
        (exc_remainder // int_scale) < (cum_remainder // int_scale)
    ).to(torch.int32)
    # sum = is_this_class.sum(dim=1)
    # print(f"K={K}, sum={sum}, r={r.flatten()}")
    # assert torch.all(is_this_class.sum(dim=1) == K)
    return is_this_class


def balance_target_marginals(
    log_p: Tensor,
    padding_scale: Tensor,
    K: int,
    num_iters: int,
    momentum: float = 0.0,
) -> Tensor:
    """
      Approximately balances the target marginals by means of
      a multiplicative factor on the probabilities p.  This is one iteration of an iterative update;
      you can call this in a loop. This update method is safe -- will not diverge --
      but not particularly exact or aggressive.  It is a bit like generalized
      iterative scaling (GIS).

      Args:
            log_p: the log-probabilities from which we'll be computing the target
              marginals to be balanced, of shape (num_frames, num_classes).
              It is not required to be
              normalized (i.e. sum to 1 over the M dimension)
              We want the computed target_marginals to be equal over the classes,
              within each block.
    padding_scale: a Tensor of shape (num_frames, 1), that is 1 for padding frames
              and 0 for non-padding frames.  padding_scale.sum() cannot be 0.
            K: the num-samples we'll be using to randomly approximate the distribution;
               an argument to compute_target_marginals.
        Returns:
             a modified log_p, which is log_p plus a correction factor for
             each class.  Will not be properly normalized to sum to 1 after exp.
    """

    num_non_padding_frames = padding_scale.sum()

    for i in range(num_iters):
        target_marginals = padding_scale * compute_target_marginals(
            log_p.softmax(dim=-1), K
        )
        M = log_p.shape[-1]
        # `target` is the average marginal for each class, if they have equal
        # probabilities within each block.
        target = num_non_padding_frames * K / M

        actual = target_marginals.sum(dim=0, keepdim=True).clamp_(min=1.0e-05)
        if i == 0 or momentum == 0.0:
            factor = target / actual
        else:
            new_factor = target / actual
            different_update_sign = (factor > 1.0) != (new_factor > 1.0)
            factor.masked_fill_(different_update_sign, 1.0)
            factor = (factor ** momentum) * (target / actual)
        log_p = log_p + factor.log()
    return log_p


def counts_to_row_splits(counts: Tensor) -> Tensor:
    """
    Given a tensor of arbitrary counts >= 0, of type torch.int32, return a vector of row_splits
    (as in k2, but we're not taking a dependency on k2), which is ([0] + counts).cumsum(),
    """
    assert counts.dtype == torch.int32 and counts.ndim == 1
    row_splits = torch.cat(
        (torch.zeros(1, dtype=torch.int32, device=counts.device), counts), dim=0
    ).cumsum(dim=0, dtype=torch.int32)
    return row_splits


def counts_to_row_splits_and_ids(counts: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Given a tensor of arbitrary counts >= 0, of type torch.int32, return a vector of row_splits
    (as in k2, but we're not taking a dependency on k2), which is ([0] + counts).cumsum(),
    and a vector of row-ids which we obtain from the row_splits.

    Args:
      counts: a Tensor of int32 shape (num_rows,), containing [row_length0, row_length1, ..., ]
    Returns:
      a pair (row_splits, row_ids), where:
         row_splits is a Tensor of shape (counts.numel()+1,) containing [0, counts[0], counts[0]+counts[1], ... ]
            row_ids is a Tensor containing something like [0,0,0,1,1,2,2,2,2,2,3,...], of length
                  equal to counts.sum(), where row_ids.max() == counts.numel()-1.
    """
    row_splits = counts_to_row_splits(counts)
    num_items = row_splits[-1].item()
    row_ids = torch.empty(num_items, dtype=torch.int32, device=counts.device)

    row_splits_to_row_ids(row_splits, row_ids)
    return (row_splits, row_ids)


def row_splits_to_row_ids(row_splits: Tensor, row_ids: Tensor):
    """
    Fills in the 'row_ids' tensor based on the 'row_splits' tensor.  This is like a similar
    function in k2 (we are avoiding a dependency on k2).
    E.g. if row_splits == [ 0, 2, 2, 4, 5 ],
    row_ids is expected to have 5 elements at entry, and this function will
    fill it in with the following values: [ 0, 0, 2, 2, 3 ].
    """
    if row_splits.is_cuda:
        if balanced_sampling_cuda is None:
            raise EnvironmentError("Failed to load native CUDA module")
        return balanced_sampling_cuda.row_splits_to_row_ids(row_splits, row_ids)
    else:
        return balanced_sampling_cpu.row_splits_to_row_ids(row_splits, row_ids)


def assign_missing_counts(
    frame_missing_counts_row_splits,
    class_missing_counts_row_ids,
    randperm,
    class_counts,
):
    """
    Pseudo-randomly assign missing counts, to ensure that all frames and all classes
    have exactly the same number of counts.

    Args:
       frame_missing_counts_row_ids: a row-ids vector (something
          like [0, 0, 0, 1, 1, 2, 3, 3, ... ]) of length
          (num_missing_counts,) that assigns missing counts to frames.
       class_missing_counts_row_ids: a row-ids vector of length
          (num_missing_counts,) that assigns missing counts to classes.
     randperm: a random permutation, of type torch.int32, of integers
        {0,1,...num_missing_counts-1}.
     class_counts: a Tensor of type torch.int32, which at entry will contain only 0 or 1.
        this function will increment some elements of class_counts, so that at exit
        a few elements of `class_counts` may be greater than 1.  At exit, each
        frame and each class should have exactly the same count.
    """
    if frame_missing_counts_row_splits.is_cuda:
        if balanced_sampling_cuda is None:
            raise EnvironmentError("Failed to load native CUDA module")
        return balanced_sampling_cuda.assign_missing_counts(
            frame_missing_counts_row_splits,
            class_missing_counts_row_ids,
            randperm,
            class_counts,
        )
    else:
        return balanced_sampling_cpu.assign_missing_counts(
            frame_missing_counts_row_splits,
            class_missing_counts_row_ids,
            randperm,
            class_counts,
        )


def indexes_from_cumsums(
    class_counts_Mcumsum: Tensor,
    class_counts_Fcumsum: Tensor,
    K: int,
    target_class_count: int,
) -> Tuple[Tensor, Tensor]:
    """
     This function creates two tensors of int32 indexes that are the output
     of the `balanced_sample` function.  At the point when this is called,
     we have created a tensor of class-counts of shape (num_frames, num_classes)
     containing small numbers like 0, 1 or sometimes 2 or 3, that represent
     how many times this class was sampled on this frame.  This tensor is
     'balanced', in that each frame's total count is K and each
     class's total count is `target_class_count`.

     Args:
         class_counts_Mcumsum: the result of calling class_counts.cumsum(dim=1),
                            containing values in 0,1,..K;
                            shape is (num_frames, num_classes).
         class_counts_Fcumsum: the result of calling class_counts.cumsum(dim=0),
                            containing values in 0,1,..target_class_count;
                            shape is (num_frames, num_classes).
        K, target_class_count: the total count per (frame, class)


    Returns: (indexes_in, indexes_out), where:
         indexes_in: a Tensor of torch.int64, of shape (num_classes, target_class_count),
                with elements in [0..num_frames-1]
         indexes_out: a Tensor of torch.int64, of shape (num_frames, K),
                with elements in [0..(num_classes*target_class_count)], formed
                as (class_idx*target_class_count + pos_within_class).
    """

    (num_frames, num_classes) = class_counts_Mcumsum.shape
    device = class_counts_Mcumsum.device
    debug = True
    # outputs are torch.int64 because they will eventually be used for
    # torch indexing, which, for early torch versions, only supports int64.
    if debug:
        indexes_in = torch.full(
            (num_classes, target_class_count),
            -1,
            dtype=torch.int64,
            device=device,
        )
        indexes_out = torch.full(
            (num_frames, K), -1, dtype=torch.int64, device=device
        )
    else:
        indexes_in = torch.empty(
            num_classes, target_class_count, dtype=torch.int64, device=device
        )
        indexes_out = torch.empty(
            num_frames, K, dtype=torch.int64, device=device
        )

    if class_counts_Mcumsum.is_cuda:
        if balanced_sampling_cuda is None:
            raise EnvironmentError("Failed to load native CUDA module")
        balanced_sampling_cuda.indexes_from_cumsums(
            class_counts_Mcumsum, class_counts_Fcumsum, indexes_in, indexes_out
        )
    else:
        balanced_sampling_cpu.indexes_from_cumsums(
            class_counts_Mcumsum, class_counts_Fcumsum, indexes_in, indexes_out
        )
    if debug:
        assert torch.all(indexes_in >= 0)
        assert torch.all(indexes_out >= 0)
    return (indexes_in, indexes_out)


def compute_positive_expectation(mean: Tensor, var: Tensor) -> Tensor:
    """
    For a normally distributed variable x ~ N(mean; var), compute
    the expected value of max(x, 0).  This is equal to sqrt(var)
     E[max(N(norm_mean; 1), 0)], where norm_mean = mean/sqrt(var).

    We can write E[max(N(norm_mean; 1), 0)] as the integral:
      \int_{x=0}^\infinity        x . 1/sqrt(2*pi) exp(-(x-norm_mean)^2)   dx

       = ((sqrt(pi) m (erf(m) + 1)  +  exp(-m^2)) / (2*sqrt(2 pi))

    if m == norm_mean == mean/sqrt(var)

    We approximate the error function as in:
    https://en.wikipedia.org/wiki/Error_function#Approximation_with_elementary_functions
    where the simplest formula is:

     erf(x) \simeq  1 - (1 / f(x)^4),  for x >= 0
     where f(x) = (1 + 0.278393 x + 0.230389 x^2 + 0.000972 x^3 + 0.078108 x^4)^4
     with maximum error: 5*10_4.
    For x < 0 we use antisymmetry to compute it.
    """
    if mean.dtype != torch.float:
        return compute_positive_expectation(
            mean.to(torch.float), var.to(torch.float)
        ).to(mean.dtype)

    ans = torch.empty_like(mean)
    if mean.is_cuda:
        if balanced_sampling_cuda is None:
            raise EnvironmentError("Failed to load native CUDA module")
        balanced_sampling_cuda.compute_positive_expectation(mean, var, ans)
    else:
        balanced_sampling_cpu.compute_positive_expectation(mean, var, ans)
    return ans


def compute_excess_count(mean: Tensor, var: Tensor, cutoff: int) -> Tensor:
    """
     If counts for class c are distributed with mean `mean[c]` and variance `var[c]`,
     return a tensor that tells us, for each class, the expected excess count
     that is greater than `cutoff`.  Of course, if these are integer counts,
     this will be an approximation based on a Gaussian distribution (valid
     as counts get large due to central limit theorem).

     Args:
        mean: a Tensor of shape (num_classes,), giving the mean/expected
              count per class.
        var: a Tensor of shape (num_classes,), giving the variance of the
              count for ech class.
    Returns:
        the expected value of the excess count above `cutoff`, i.e.
        the expected value of max(x - cutoff, 0) with x ~ N(mean; var),
        elementwise.
    """
    return compute_positive_expectation(mean - cutoff, var)


def compute_corrected_counts(
    marginals: Tensor, padding_scale: Tensor, K: int, target_class_count: int
) -> Tensor:
    """
    Compute corrected expected-counts/marginals for classes on individual frames;
    meaning, corrected for our procedure that we use to ensure all classes have
    exactly the smame count.  These marginals will only be correect for non-padding
    frames; we will output values for padding frames but assume they will be ignored.

    Our sampling procedure (the procedure we are computing expected class-counts
    for) is as follows.  (Note: K -- not provided -- is the number of samples we want per frame, and
    target_class_count is the number of samples w want per class; we have
    made sure via adding padding frames that target_class_count is a whole
    number).

         (i) sample exactly K distinct classes for each non-padding frame,
            with expected counts equal to those in `marginals`
        (ii) for any class that has an "excess count" (i.e. has a count
            greater than `target_class_count`, zero enough of its counts
            (with equal probability) to reduce the count to `target_class_count`
        (iii) randomly assign the unassigned counts, without regard to the
           target marginals, to ensure that all frames and all classes have
           the exact same count.  When doing this we do not exclude already-chosen
           classes, so it is possible for a class's count on a particular
           frame to be more than 1.


    Args:
          marginals:  the marginal probabilities with which we sampled the
                  classes in stage (i); this is only valid for non-padding frames.
                  Shape: (num_frames, num_classes).  0 <= marginals <= 1.
                  Will satisfy marginals.sum(dim=1) == K (K integer but not
                  provided).
         padding_scale: a Tensor fo shape (num_frames, 1) that contains a 1
                  on non-padding frames and a 1 on padding frames.  Frames
                  may be padded either because of rounding up the number of
                  frames to ensure target_class_count is a whole number, or
                  because the user/caller told us they were padding frames.
             K: the number of samples per frame
         target_class_count: the number of samples we will have for each class;
                  will be equal to num_frames * K / num_classes.
    """
    # keep just the non-padding-frame marginals.
    marginals = marginals * padding_scale
    # mean count per class.  shape=(num_classes,)
    mean = marginals.sum(dim=0)
    # var is the variance of the count around its mean, per class.
    # shape=(num_classes,)
    var = (marginals * (1.0 - marginals)).sum(dim=0)
    # excess is the expected excess count above `target_class_count`
    excess = compute_excess_count(mean, var, target_class_count)

    # The expected aount after stage (ii) is (mean - excess), since
    # we randomly remove the excess count.  So the expected *unassigned*
    # count for each class after stage (ii) is as below.
    # shape is (num_classes,)
    unassigned = target_class_count - (mean - excess)

    # zero_prob is the probability that we zero a count in stage (ii),
    # given that we selected it in stage (i).  This is a slight approximation,
    # because the fact that we chose the count in stage (i) gives us a small
    # amount of information about the total-count for that class, but
    # if `var` is large enough this is a small effect.
    # shape is (num_frames, num_classes,)
    zero_prob = excess / mean

    # OK, if a count is zeroed, what is it replaced with?
    # our first approximation would be `unassigned / unassigned.sum()`,
    # because those are the expected value of the counts that
    # we randomly distribute.  But we have a little more information than
    # this.  If we zero a class 'c', then we know that we are not
    # replacing it with class 'c', because on this batch, 'c'
    # had nonzero 'excess' count so its unassigned count is zero.
    # So if count c is zeroed, the expected replacement count is:
    #  ((unassigned, but with count c zeroed)/(unassigned.sum()-unassigned[c])
    # For a particular frame, with marginals 'm', we can compute the expected
    # reassigned-count (i.e. the class distributed from the 'unassigned' category)
    # to be as follows:
    #  \sum_c marginal(c) * zero_prob(c) *  ((unassigned, but with count c zeroed)/(unassigned.sum()-unassigned[c])
    #
    # So ignoring the "with count c zeroed" part, the redistributed count would be:
    #   unassigned * \sum_c marginal(c) * zero_prob(c) / (unassigned.sum() - unassigned[c])
    # .. and the correction term to implement the "with count c zeroed" part of the above, is:
    #  - marginal(c) * zero_prob(c) * unassigned(c) / (unassigned.sum() - unassigned[c])
    # .. and the count change due to zeroing the excess counts in phase (ii) is:
    # - marginal(c) * zero_prob(c)
    # So if we let "inv_den" = 1.0 / (unassigned.sum()-unassigned),
    # then the final counts will be:
    #  marginal   # original count
    # - marginal * zero_prob  # count removed in phase (ii)
    # + unassigned * (marginal * zero_prob / (unassigned.sum() - unassigned)).sum()  # count added in phase (iii)
    # - unassigned * (marginal * zero_prob / (unassigned.sum() - unassigned)) # correction because excess count
    #                                                                         # can't be replaced by same class
    # The above is the correct formula for the expecgted count on the non-padding frames; for padding
    # frames it is simply K * unassigned / unassigned.sum()

    mz = marginals * zero_prob
    mzdiv = mz / (unassigned.sum() - unassigned)
    corrected_marginals = (
        marginals
        - mz
        + unassigned * mzdiv.sum(dim=1, keepdim=True)
        - unassigned * mzdiv
    )
    # the above corrected marginal is only
    padding_marginals = unassigned * (K / unassigned.sum())
    return (
        padding_scale * corrected_marginals
        + (1 - padding_scale) * padding_marginals
    )


def balanced_sample(
    logprobs: Tensor,
    K: int,
    padding: Optional[Tensor] = None,
    min_pad_items: int = 0,
) -> Tuple[int, Tensor, Tensor, Tensor]:
    """
       Method of sampling multiple from categorical distributions that ensures each class
       is sampled exactly the same number of times (so that we can use batched matrix
       multiply in a computation that depends on these sampled class identities.

      Args:
       logprobs: A tensor of shape (B, M) where B will be treated as a batch dimension,
               and M is the number of classes, containing the logprobs of the classes to
               in a weighted sum that we'll be approximating via sampling.
             K: Number of samples that we'll be approximating each weighted sum with.
                Must be >= 1 (but probably, in practice, >= 2).
                K must exactly divide M.
        padding: a Tensor of bool, of shape (B,) containing True in positions that correspond
                to padding frames.  The user asserts that the indexes chosen for these
                frame do not matter (i.e. that they are not going to affect the final
                result of the computation.
    min_pad_items: an integer saying the minimum number of padding items that must be
                added to the B items in `logprobs` (items with elements set to True
                in `padding` count towards this total).  You can just leave this at 0;
                larger values will tend to make the approximation more exact, at the
                expense of extra computation.

       Returns (F, indexes_in, indexes_out, weights_out), where:
              F: (number of frames after padding): an integer >= the input batch size B,
                 which may not exactly equal B
                 because of padding to a multiple of M//K, plus possibly extra padding
                 depending on the value of min_pad_items.

           indexes_in: a Tensor of torch.int64, of shape (M, samples_per_class),
                 where samples_per_class == (F*K) // M
                 containing indexes in [0..F-1]; you can use (indexes_in % B) for indexing
                 if you want, in your computation, as the padding values do not matter.
                 M is the number of classes; this is the leading tensor dimension for
                 convenience of batched matrix multiply in whatever weighted computation
                 you will be doing.
           weights_out: a Tensor of shape (M, samples_per_class), giving the weight corresponding to
                 each item in the weighted sum (these should sum over the last dimension
                 to a value fairly close to 1).  The weights have the property that
                 the *expected* values of the weights, viewed as a sparse tensor with
                 indexes_in, equals logprobs.exp() for non-padding frames.
                 For padding frames, all weights are set to zero.
           indexes_out: a Tensor of shape (B, K), containing indexes in 0..M*samples_per_class - 1,
                 which can be used to gather the weighted outputs of the computation,
                 to be summed.  I.e. if the weighted output of the computation is a
                 tensor of shape (M, N, K, *), you can reshape it to (M*N*K, *) and index
                 it with this tensor to produce something of shape (B, K, *) that
                 can be summed over the K dimension.
    """
    # Implementation notes:
    # (1) compute F >= B; pad logprobs to shape (F, M).

    (B, num_classes) = logprobs.shape

    if True:
        # This block computes the rounded-up and padded number of frames.
        assert num_classes % K == 0
        # the padded num-frames, F, must be an exact multiple of n so that the
        # average count of each class can be a whole number.
        N = num_classes // K

        # work out F >= B, the padded and rounded number of frames.
        F = N * ((B + min_pad_items + (N - 1)) // N)

    # pad `p` to F frames.  doesn't need to be normalized.  Anyway, we'll eventually be ignoring
    # the values of `probs` on padding frames.
    logprobs = torch.cat(
        (
            logprobs,
            torch.zeros(
                F - B, num_classes, device=logprobs.device, dtype=logprobs.dtype
            ),
        ),
        dim=0,
    )

    if True:
        # This block computes "padding_scale", which is a tensor of shape (F, 1) containing
        # 1 for non-padding frames and 0 for padding frames.
        if padding is None:
            padding_scale = torch.ones(
                B, 1, device=logprobs.device, dtype=torch.int32
            )
        else:
            padding_scale = (
                torch.logical_not(padding).to(logprobs.dtype).unsqueeze(-1)
            )
        padding_scale = torch.cat(
            (
                padding_scale,
                torch.zeros(
                    F - B, 1, device=logprobs.device, dtype=logprobs.dtype
                ),
            ),
            dim=0,
        )

    # Approximately rebalance logprobs so target marginals are approx. equal; compute
    # target marginals.
    balanced_logprobs = balance_target_marginals(
        logprobs.detach(), padding_scale, K, num_iters=2
    )
    p = balanced_logprobs.softmax(dim=-1)
    target_marginals = compute_target_marginals(p, K)

    # class_counts: zero or one counts of shape (F, M)
    # i.e. (num_frames_padded, num_classes)
    class_counts = sample_from_target_marginals(target_marginals, K)
    # zero class counts that are on padding frames.
    class_counts *= padding_scale.to(class_counts.dtype)

    class_counts_Mcumsum = class_counts.cumsum(dim=1, dtype=torch.int32)

    class_counts_Fcumsum = (
        class_counts.transpose(0, 1)
        .contiguous()
        .cumsum(dim=1, dtype=torch.int32)
    )

    # class_tot_counts: total counts per class, of shape (M,)
    class_tot_counts = class_counts_Fcumsum[:, -1].contiguous()

    largest_class_count = class_tot_counts.max().item()

    # frames_for_class is a tensor of shape (num_classes, largest_class_count)
    # containing the frame-indexes that this class was sampled on.
    frames_for_class = compute_count_indexes(
        class_counts_Fcumsum, largest_class_count
    )

    target_class_count = (F * K) // num_classes
    assert (F * K) == (num_classes * target_class_count)

    if largest_class_count > target_class_count:
        zero_excess_class_counts(
            class_counts, class_tot_counts, frames_for_class, target_class_count
        )

        class_tot_counts.clamp_(max=target_class_count)

        assert torch.all(
            class_tot_counts == class_counts.sum(dim=0, dtype=torch.int32)
        )

    frame_tot_counts = class_counts.sum(dim=1, dtype=torch.int32)

    frame_missing_counts = K - frame_tot_counts
    class_missing_counts = target_class_count - class_tot_counts

    frame_missing_counts_row_splits = counts_to_row_splits(frame_missing_counts)
    (
        class_missing_counts_row_splits,
        class_missing_counts_row_ids,
    ) = counts_to_row_splits_and_ids(class_missing_counts)

    assert (
        frame_missing_counts_row_splits[-1].item()
        == class_missing_counts_row_ids.numel()
    )  # TEMP.. may be slow
    randperm = torch.randperm(
        class_missing_counts_row_ids.numel(),
        dtype=torch.int32,
        device=class_counts.device,
    )

    assign_missing_counts(
        frame_missing_counts_row_splits,
        class_missing_counts_row_ids,
        randperm,
        class_counts,
    )

    assert torch.all(class_counts.sum(dim=1, dtype=torch.int32) == K)
    assert torch.all(
        class_counts.sum(dim=0, dtype=torch.int32) == target_class_count
    )

    # [:B] discards the extra padding frames that we added.
    class_counts_Mcumsum = class_counts.cumsum(dim=1, dtype=torch.int32)
    class_counts_Fcumsum = class_counts.cumsum(dim=0, dtype=torch.int32)

    # indexes_in shape is (num_classes, target_class_count); it contains values
    # in [0..F-1],.
    # indexes_out shape is (F, K); it
    # contains elements in [0..(num_classes*target_class_count)-1].
    (indexes_in, indexes_out) = indexes_from_cumsums(
        class_counts_Mcumsum, class_counts_Fcumsum, K, target_class_count
    )

    # corrected_marginals shape is (F, num_classes)
    corrected_marginals = compute_corrected_counts(
        target_marginals, padding_scale, K, target_class_count
    )

    # `orig_target_probs` is the original probabilities that the user requested
    # (the values in padding frames don't really matter, and in fact they may be
    # quite arbitrary).
    # Note: this may have requires_grad == True.
    orig_target_probs = logprobs.softmax(dim=-1)

    # print(f"max err vs. K is {(corrected_marginals.sum(dim=1) - K).abs().max()}")
    # print(f"max err vs. target_class_count is {(corrected_marginals.sum(dim=0) - target_class_count).abs().max()}")

    # `weights` are the importance-sampling weights that we need if we are going
    # to view the sampled classes as an importance-sampling approximation of
    # `orig_target_probs`
    # Note: weights may have requires_grad == True, from orig_target_probs.
    weights = orig_target_probs / corrected_marginals.clamp_(min=1.0e-10)

    # weights_as_indexes_in contains the importance-sampling weights
    # with the same shape as indexes_in, which is (num_classes, target_class_count)
    weights_as_indexes_in = torch.gather(
        weights.transpose(0, 1), dim=1, index=indexes_in
    )
    # weights_as_indexes_out contains the importance-sampling weights with
    # the same shape as indexes_out, i.e. (F, K)
    weights_as_indexes_out = torch.gather(
        weights_as_indexes_in.flatten(), dim=0, index=indexes_out.flatten()
    ).reshape(F, K)

    return (F, indexes_in, indexes_out[:B], weights_as_indexes_out[:B])


def _test_sample_from_target_marginals():
    num_frames = 100
    num_classes = 20
    probs = torch.randn(num_frames, num_classes).softmax(dim=1)
    K = 4
    target_marginals = compute_target_marginals(probs, K)

    samples = sample_from_target_marginals(target_marginals, K)

    assert torch.all(samples.sum(dim=1) == K)


def _test_compute_count_indexes():
    for device in ["cpu", "cuda"]:
        counts = torch.tensor(
            [[0, 1, 0, 0], [0, 0, 1, 1], [2, 0, 0, 0]],
            dtype=torch.int32,
            device=device,
        )
        counts_cumsum = counts.cumsum(dim=1, dtype=torch.int32)
        max_count = 2
        ans = compute_count_indexes(counts_cumsum, max_count)
        ans_ref = torch.tensor(
            [[1, -1], [2, 3], [0, 0]], dtype=torch.int32, device=device
        )
        print("ans = ", ans)
        assert torch.all(ans == ans_ref)


def _test_balanced_sample():
    for device in ["cpu", "cuda"]:
        num_frames = 100
        num_classes = 32
        logprobs = torch.randn(num_frames, num_classes, device=device) * 3.0
        K = 4
        balanced_sample(logprobs, K, padding=None, min_pad_items=0)


def _test_counts_to_row_splits_and_ids():
    for device in ["cpu", "cuda"]:
        row_counts = torch.tensor(
            [2, 0, 3, 1], dtype=torch.int32, device=device
        )
        (row_splits, row_ids) = counts_to_row_splits_and_ids(row_counts)
        assert row_splits.tolist() == [0, 2, 2, 5, 6]
        assert row_ids.tolist() == [0, 0, 2, 2, 2, 3]


def _test_compute_positive_expectation():
    N = 100
    for device in ["cpu", "cuda"]:
        mean = torch.randn(N, device=device)
        var = (5.0 * torch.randn(N, device=device)).exp()

        # compute E[max(x, 0)]
        pos_exp = compute_positive_expectation(mean, var)
        neg_exp = -compute_positive_expectation(-mean, var)
        reconstruct_mean = pos_exp + neg_exp
        print(
            f"mean={mean}, var={var}, pos_exp={pos_exp}, neg_exp={neg_exp}, reconstruct_mean={reconstruct_mean}, err={mean-reconstruct_mean}"
        )
        stddev = var.sqrt()
        assert torch.allclose(
            mean / stddev, reconstruct_mean / stddev, atol=1.0e-05
        )


def _test_balanced_sample_expectation():
    for device in ["cpu", "cuda"]:
        for num_frames in [100, 200]:
            num_classes = 32
            logprobs = torch.randn(num_frames, num_classes, device=device)
            K = 8

            p = logprobs.softmax(dim=-1)
            sampled_p = torch.zeros_like(p)

            num_samples = 50

            for _ in range(num_samples):
                (F, indexes_in, indexes_out, weights_out) = balanced_sample(
                    logprobs, K, padding=None, min_pad_items=0
                )

                samples_per_class = indexes_in.shape[1]
                classes_out = indexes_out // samples_per_class
                approx_p = torch.zeros_like(logprobs)
                approx_p.scatter_(dim=1, index=classes_out, src=weights_out)

                sampled_p.add_(approx_p, alpha=1.0 / num_samples)
            print(
                f"num_frames={num_frames}, num_samples={num_samples}, p={p}, sampled_p={sampled_p}, rel-diff={(sampled_p-p)/p}, abs-diff={(sampled_p-p)}"
            )


def _test_balanced_sample_variance():
    # compare the variance of the approximation with
    # compute_target_marginals+sample_from_target_marginals.

    # CPU: and CUDA have similar metrics, algorithm is equivalent, so not comparing.
    # With balanced_sample(), num_frames=100, num_samples=10, mean error is:  abs-diff=0.0109
    # With sample_from_target_marginals(), num_frames=100, num_samples=10, mean error is:  abs-diff=0.0102
    # With balanced_sample(), num_frames=100, num_samples=50, mean error is:  abs-diff=0.0046
    # With sample_from_target_marginals(), num_frames=100, num_samples=50, mean error is:  abs-diff=0.0045
    # With balanced_sample(), num_frames=100, num_samples=100, mean error is:  abs-diff=0.0032
    # With sample_from_target_marginals(), num_frames=100, num_samples=100, mean error is:  abs-diff=0.0031
    # With balanced_sample(), num_frames=100, num_samples=200, mean error is:  abs-diff=0.0023
    # With sample_from_target_marginals(), num_frames=100, num_samples=200, mean error is:  abs-diff=0.0023

    for device in ["cpu", "cuda"]:
        for num_frames in [32, 50, 256]:
            for num_samples in [10, 50, 100, 200]:
                num_classes = 32
                logprobs = torch.randn(num_frames, num_classes, device=device)
                K = 8

                p = logprobs.softmax(dim=-1)

                for min_pad_items in [0, 20 + num_samples // 10]:
                    sampled_p = torch.zeros_like(p)

                    # (1) compute difference of p vs. sampled_p, with balanced_sample().
                    for _ in range(num_samples):
                        (
                            F,
                            indexes_in,
                            indexes_out,
                            weights_out,
                        ) = balanced_sample(
                            logprobs, K, padding=None, min_pad_items=0
                        )

                        samples_per_class = indexes_in.shape[1]
                        classes_out = indexes_out // samples_per_class
                        approx_p = torch.zeros_like(logprobs)
                        approx_p.scatter_(
                            dim=1, index=classes_out, src=weights_out
                        )

                        sampled_p.add_(approx_p, alpha=1.0 / num_samples)
                    print(
                        f"With balanced_sample(), min_pad_items={min_pad_items}, num_frames={num_frames}, num_samples={num_samples}, "
                        f"mean error is:  abs-diff={(sampled_p-p).abs().mean().item():.4f}"
                    )

                sampled_p = torch.zeros_like(p)
                target_marginals = compute_target_marginals(p, K)
                for _ in range(num_samples):
                    # class_indexes is a matrix of zeros and ones, of shape (num_frames,
                    # num_classes)
                    class_counts = sample_from_target_marginals(
                        target_marginals, K
                    )
                    approx_p = class_counts * p / target_marginals

                    sampled_p.add_(approx_p, alpha=1.0 / num_samples)

                print(
                    f"With sample_from_target_marginals(), num_frames={num_frames}, num_samples={num_samples}, mean error is:  abs-diff={(sampled_p-p).abs().mean().item():.4f}"
                )


def _test_balanced_sample_grad():
    for device in ["cpu", "cuda"]:
        for num_frames in [100, 200]:
            num_classes = 32
            logprobs = torch.randn(num_frames, num_classes, device=device)
            logprobs.requires_grad = True
            loss_scale = torch.randn_like(logprobs)
            K = 8

            p = logprobs.softmax(dim=-1)

            exact_loss = (loss_scale * p).sum()
            exact_loss.backward()
            exact_grad = logprobs.grad

            logprobs.grad = None
            p = logprobs.softmax(dim=-1)  # reconstruct for backprop reasons.
            sampled_p = torch.zeros_like(p)

            # Caution: the following loop will require a fair amount of memory, proportional
            # to num_samples, as it keeps some things around for the gradient.
            num_samples = 50
            for _ in range(num_samples):
                (F, indexes_in, indexes_out, weights_out) = balanced_sample(
                    logprobs, K, padding=None, min_pad_items=0
                )

                samples_per_class = indexes_in.shape[1]
                classes_out = indexes_out // samples_per_class
                approx_p = torch.zeros_like(logprobs)
                approx_p.scatter_(dim=1, index=classes_out, src=weights_out)

                sampled_p.add_(approx_p, alpha=1.0 / num_samples)
            print(
                f"test_balanced_sample_grad: num_frames={num_frames}, num_samples={num_samples}, p={p}, "
                f"sampled_p={sampled_p}, rel-diff={(sampled_p-p)/p}, abs-diff={(sampled_p-p)}"
            )

            approx_loss = (loss_scale * sampled_p).sum()
            print(
                f"test_balanced_sample_grad: exact_loss={exact_loss.item()}, approx_loss={approx_loss.item()}"
            )
            approx_loss.backward()
            approx_grad = logprobs.grad
            cos_angle = (exact_grad * approx_grad).sum() / (
                exact_grad.norm() * approx_grad.norm()
            )
            print(
                f"test_balanced_sample_grad: exact_grad norm={exact_grad.norm().item()}, "
                f"approx_grad norm={approx_grad.norm().item()}, cos-angle={cos_angle.item()}"
            )

    # Output is:
    # test_balanced_sample_grad: exact_loss=2.680172920227051, approx_loss=1.496469259262085
    # test_balanced_sample_grad: exact_grad norm=3.4283077716827393, approx_grad norm=3.484287738800049, cos-angle=0.9906828999519348
    # .. don't worry about the relatively big loss difference, there is a lot of cancellation
    # going on there.  we care more about the gradient, and that is in almost exactly the desired
    # direction.


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _test_sample_from_target_marginals()
    _test_compute_count_indexes()
    _test_counts_to_row_splits_and_ids()
    _test_compute_positive_expectation()
    _test_balanced_sample()
    _test_balanced_sample_expectation()
    _test_balanced_sample_variance()
    _test_balanced_sample_grad()
