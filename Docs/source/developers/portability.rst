.. _developers-portability:

Portability
===========

WarpX runs on CPUs (serial or OpenMP) and GPUs (CUDA, HIP, SYCL) from a single source.
Compute kernels are written as C++ lambdas and launched through AMReX's portable constructs, which compile to loops on CPU and kernel launches on GPU (see the `AMReX GPU documentation <https://amrex-codes.github.io/amrex/docs_html/GPU.html>`__).

Choosing a kernel launch construct
----------------------------------

The launch constructs differ in the promises they make to the compiler.
Picking the wrong one compiles and runs, but can produce silently wrong results, so review this choice carefully in every new kernel:

* ``amrex::ParallelFor``: on CPU, the innermost loop is marked with ``AMREX_PRAGMA_SIMD`` (e.g., ``#pragma GCC ivdep``), which **promises the compiler that loop iterations are independent** and safe to vectorize.
  Use it only when no two iterations can touch the same memory location.
  Standard field stencils (each iteration writes only its own ``(i,j,k)`` and reads separate input arrays) and per-particle updates (each iteration writes only element ``ip``) qualify.

* ``amrex::For``: identical to ``ParallelFor`` on GPU, but omits the SIMD pragma on CPU.
  Use it whenever different iterations may access the same memory location with at least one write.
  Typical cases in WarpX: **charge/current deposition** and any other particle-to-grid scatter, histogram binning, and updates of shared counters.

* ``amrex::ParallelForRNG``: carries no SIMD pragma on CPU; used when random numbers are needed and also safe for non-independent iterations.

* Whole-loop reductions (sums, maxima) should use the ``amrex::ReduceSIMD`` (or ``amrex::Reduce``) functions instead of accumulating into a shared scalar from a kernel.
  When compiled for GPU, the ``ReduceSIMD`` code path is inactive and the standard ``amrex::Reduce`` device reduction is used.

.. warning::

   ``amrex::Gpu::Atomic`` operations (``AddNoRet``, ``Add``, ``Max``, ...) are **plain, non-atomic updates on CPU**.
   They make scatter kernels safe between GPU threads, but they do *not* make a ``ParallelFor`` loop safe on CPU: under the SIMD pragma the compiler may still vectorize the loop, and vector lanes that hit the same address lose all but one update.
   A kernel that needs ``Gpu::Atomic`` because iterations collide almost always needs ``amrex::For`` (or ``ParallelForRNG``) rather than ``ParallelFor``.

   ``amrex::HostDevice::Atomic`` (``Add``, ``FetchAdd``) is atomic on GPU *and* across OpenMP threads on CPU, so prefer it over ``Gpu::Atomic`` in host-device code.
   It still does not make a ``ParallelFor`` safe: the SIMD pragma's independence promise remains violated, and in non-OpenMP builds the update is a plain ``+=``.

Getting ``amrex::ParallelFor`` and atomics wrong is silent and compiler-dependent: the miscompilation only appears when a vectorizer decides to act on the pragma.
For example, GCC 15 vectorized the charge deposition loop for cubic shape factors in 1D, which dropped three of every four contributions and produced a charge density four times too small, while GCC 13 left the same invalid code intact.
See `issue #7097 <https://github.com/BLAST-WarpX/warpx/issues/7097>`__ for the full analysis.
