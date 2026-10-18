# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run torch.svd_lowrank on CPU when its input lives on MPS.

Why: with torch 2.14.1 on Apple Silicon, torch.linalg.qr on tall MPS matrices is pathologically
slow (a 3840x10 QR takes > 60 s; a 64x10 one takes 0.01 s). torch.svd_lowrank(niter=6) runs
13 such QRs per matrix, so Heretic's default row_normalization="full" (which calls
torch.svd_lowrank once per abliterated o_proj/down_proj module) stalled for 20+ minutes on the
first trial. The same call on CPU takes ~0.04-0.11 s for Gemma 4 12B's matrix shapes.

What: replaces torch.svd_lowrank with a wrapper that, for MPS inputs only, moves the input to
CPU, calls the original function unchanged, and moves U, S, V back to the input's device.
Everything else (other devices, Heretic's math, settings, and seeding) is untouched. Heretic
calls torch.manual_seed(seed) right before each svd_lowrank call, which also seeds the CPU
generator, so results stay deterministic.
"""

import torch

_original_svd_lowrank = torch.svd_lowrank


def _svd_lowrank_cpu_for_mps(A, q=6, niter=2, M=None):
    if A.device.type != "mps":
        return _original_svd_lowrank(A, q=q, niter=niter, M=M)

    U, S, V = _original_svd_lowrank(
        A.cpu(),
        q=q,
        niter=niter,
        M=None if M is None else M.cpu(),
    )
    return U.to(A.device), S.to(A.device), V.to(A.device)


torch.svd_lowrank = _svd_lowrank_cpu_for_mps
