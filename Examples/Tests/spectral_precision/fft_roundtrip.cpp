/**
 * fft_roundtrip: Standalone test for the AnyFFT abstraction layer.
 *
 * Validates R2C -> C2R round-trip accuracy and spectral correctness
 * across all FFT backends (FFTW, cuFFT, rocFFT, MKL DFT).
 *
 * Test 1: Round-trip (forward R2C + backward C2R) preserves the input
 *         array to machine precision (modulo the N normalization factor).
 *
 * Test 2: A single-mode sinusoid sin(2*pi*k0*i/N) produces spectral
 *         peaks only at the expected bin k0 and is zero elsewhere.
 *
 * Returns 0 (PASS) or 1 (FAIL) for CTest integration.
 */

#include <ablastr/math/fft/AnyFFT.H>

#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuComplex.H>
#include <AMReX_Vector.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <vector>

using namespace amrex;
using Complex = ablastr::math::anyfft::Complex;

namespace {

// Extract real and imaginary parts from vendor-specific Complex type.
AMREX_FORCE_INLINE
Real cabs2 (Complex const& c)
{
#if defined(AMREX_USE_CUDA)
    return c.x * c.x + c.y * c.y;
#elif defined(AMREX_USE_HIP)
    return c.x * c.x + c.y * c.y;
#elif defined(AMREX_USE_SYCL)
    return c.real() * c.real() + c.imag() * c.imag();
#else
    // FFTW: Complex is double[2]
    return c[0] * c[0] + c[1] * c[1];
#endif
}

AMREX_FORCE_INLINE
Real creal_part (Complex const& c)
{
#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
    return c.x;
#elif defined(AMREX_USE_SYCL)
    return c.real();
#else
    return c[0];
#endif
}

AMREX_FORCE_INLINE
Real cimag_part (Complex const& c)
{
#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
    return c.y;
#elif defined(AMREX_USE_SYCL)
    return c.imag();
#else
    return c[1];
#endif
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Test 1: 1D round-trip — R2C then C2R should recover N * original
// ---------------------------------------------------------------------------
bool test_roundtrip_1d ()
{
    amrex::Print() << "\n[Test 1] 1D FFT round-trip (N=64)\n";

    const int N = 64;
    const int Nc = N / 2 + 1;  // R2C complex output size
    const Real tol = std::is_same_v<Real, float> ? Real(1.0e-5) : Real(1.0e-12);

    // Allocate on host
    std::vector<Real> h_real(N);
    const int k0 = 5;
    for (int i = 0; i < N; ++i) {
        h_real[i] = std::sin(2.0 * M_PI * k0 * i / N)
                  + 0.3 * std::cos(2.0 * M_PI * 11 * i / N);
    }
    std::vector<Real> h_orig(h_real);

    // Allocate on device
    Gpu::DeviceVector<Real> d_real(N);
    Gpu::DeviceVector<Complex> d_complex(Nc);
    Gpu::htod_memcpy(d_real.data(), h_real.data(), N * sizeof(Real));

    // Create plans
    IntVect real_size(AMREX_D_DECL(N, 1, 1));
    auto fwd = ablastr::math::anyfft::CreatePlan(
        real_size, d_real.data(), d_complex.data(),
        ablastr::math::anyfft::direction::R2C, 1);
    auto bwd = ablastr::math::anyfft::CreatePlan(
        real_size, d_real.data(), d_complex.data(),
        ablastr::math::anyfft::direction::C2R, 1);

    // Forward then backward
    ablastr::math::anyfft::Execute(fwd);
    ablastr::math::anyfft::Execute(bwd);

    // Copy result back
    Gpu::dtoh_memcpy(h_real.data(), d_real.data(), N * sizeof(Real));

    // Check: result should be N * original
    Real max_err = 0.0;
    for (int i = 0; i < N; ++i) {
        Real expected = h_orig[i] * N;
        Real err = std::abs(h_real[i] - expected);
        Real scale = std::max(std::abs(expected), Real(1.0));
        max_err = std::max(max_err, err / scale);
    }

    ablastr::math::anyfft::DestroyPlan(fwd);
    ablastr::math::anyfft::DestroyPlan(bwd);

    bool pass = (max_err < tol);
    amrex::Print() << "  max relative error = " << max_err
                   << " (tol = " << tol << ") ... "
                   << (pass ? "PASS" : "FAIL") << "\n";
    return pass;
}

// ---------------------------------------------------------------------------
// Test 2: Spectral correctness — single-mode sinusoid should produce
//         peaks only at the expected frequency bin.
// ---------------------------------------------------------------------------
bool test_spectral_peaks_1d ()
{
    amrex::Print() << "\n[Test 2] 1D spectral peak verification (N=64, k0=7)\n";

    const int N = 64;
    const int Nc = N / 2 + 1;
    const int k0 = 7;
    const Real amplitude = 3.14;
    const Real tol = std::is_same_v<Real, float> ? Real(1.0e-4) : Real(1.0e-10);

    // sin(2*pi*k0*i/N) -> DFT: imaginary peak at bin k0 with magnitude -N/2
    std::vector<Real> h_real(N);
    for (int i = 0; i < N; ++i) {
        h_real[i] = amplitude * std::sin(2.0 * M_PI * k0 * i / N);
    }

    Gpu::DeviceVector<Real> d_real(N);
    Gpu::DeviceVector<Complex> d_complex(Nc);
    Gpu::htod_memcpy(d_real.data(), h_real.data(), N * sizeof(Real));

    IntVect real_size(AMREX_D_DECL(N, 1, 1));
    auto fwd = ablastr::math::anyfft::CreatePlan(
        real_size, d_real.data(), d_complex.data(),
        ablastr::math::anyfft::direction::R2C, 1);

    ablastr::math::anyfft::Execute(fwd);

    // Copy spectral data back
    std::vector<Complex> h_complex(Nc);
    Gpu::dtoh_memcpy(h_complex.data(), d_complex.data(), Nc * sizeof(Complex));

    ablastr::math::anyfft::DestroyPlan(fwd);

    // Expected: bin k0 should have imaginary part = -amplitude*N/2
    //           all other bins should be zero
    Real expected_imag = -amplitude * N / 2.0;
    Real peak_mag2 = cabs2(h_complex[k0]);
    Real expected_mag2 = expected_imag * expected_imag;

    Real peak_err = std::abs(std::sqrt(peak_mag2) - std::abs(expected_imag))
                  / std::abs(expected_imag);

    // Check that the real part at bin k0 is zero (sin is purely imaginary in DFT)
    Real real_at_peak = std::abs(creal_part(h_complex[k0]));
    Real peak_real_err = real_at_peak / (amplitude * N / 2.0);

    // Check that all non-peak bins are near zero
    Real max_noise = 0.0;
    for (int k = 0; k < Nc; ++k) {
        if (k == k0) { continue; }
        Real mag = std::sqrt(cabs2(h_complex[k]));
        max_noise = std::max(max_noise, mag);
    }
    Real noise_ratio = max_noise / (amplitude * N / 2.0);

    bool pass = (peak_err < tol) && (peak_real_err < tol) && (noise_ratio < tol);
    amrex::Print() << "  peak magnitude error = " << peak_err << "\n"
                   << "  real part at peak    = " << peak_real_err << "\n"
                   << "  max noise ratio      = " << noise_ratio << "\n"
                   << "  (tol = " << tol << ") ... "
                   << (pass ? "PASS" : "FAIL") << "\n";
    return pass;
}

// ---------------------------------------------------------------------------
// Test 3: 2D round-trip (if AMREX_SPACEDIM >= 2)
// ---------------------------------------------------------------------------
bool test_roundtrip_2d ()
{
#if AMREX_SPACEDIM < 2
    amrex::Print() << "\n[Test 3] 2D FFT round-trip — SKIPPED (1D build)\n";
    return true;
#else
    amrex::Print() << "\n[Test 3] 2D FFT round-trip (32x32)\n";

    const int Nx = 32, Ny = 32;
    const int Nc = Nx / 2 + 1;  // R2C: first dim is halved
    const int N_real = Nx * Ny;
    const int N_complex = Nc * Ny;
    const Real tol = std::is_same_v<Real, float> ? Real(1.0e-4) : Real(1.0e-11);

    std::vector<Real> h_real(N_real);
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            h_real[j * Nx + i] =
                std::sin(2.0 * M_PI * 3 * i / Nx) *
                std::cos(2.0 * M_PI * 2 * j / Ny);
        }
    }
    std::vector<Real> h_orig(h_real);

    Gpu::DeviceVector<Real> d_real(N_real);
    Gpu::DeviceVector<Complex> d_complex(N_complex);
    Gpu::htod_memcpy(d_real.data(), h_real.data(), N_real * sizeof(Real));

    IntVect real_size(AMREX_D_DECL(Nx, Ny, 1));
    auto fwd = ablastr::math::anyfft::CreatePlan(
        real_size, d_real.data(), d_complex.data(),
        ablastr::math::anyfft::direction::R2C, 2);
    auto bwd = ablastr::math::anyfft::CreatePlan(
        real_size, d_real.data(), d_complex.data(),
        ablastr::math::anyfft::direction::C2R, 2);

    ablastr::math::anyfft::Execute(fwd);
    ablastr::math::anyfft::Execute(bwd);

    Gpu::dtoh_memcpy(h_real.data(), d_real.data(), N_real * sizeof(Real));

    Real max_err = 0.0;
    Real N_total = Real(Nx * Ny);
    for (int i = 0; i < N_real; ++i) {
        Real expected = h_orig[i] * N_total;
        Real err = std::abs(h_real[i] - expected);
        Real scale = std::max(std::abs(expected), Real(1.0));
        max_err = std::max(max_err, err / scale);
    }

    ablastr::math::anyfft::DestroyPlan(fwd);
    ablastr::math::anyfft::DestroyPlan(bwd);

    bool pass = (max_err < tol);
    amrex::Print() << "  max relative error = " << max_err
                   << " (tol = " << tol << ") ... "
                   << (pass ? "PASS" : "FAIL") << "\n";
    return pass;
#endif
}

// ---------------------------------------------------------------------------
// Test 4: 3D round-trip (if AMREX_SPACEDIM >= 3)
// ---------------------------------------------------------------------------
bool test_roundtrip_3d ()
{
#if AMREX_SPACEDIM < 3
    amrex::Print() << "\n[Test 4] 3D FFT round-trip — SKIPPED ("
                   << AMREX_SPACEDIM << "D build)\n";
    return true;
#else
    amrex::Print() << "\n[Test 4] 3D FFT round-trip (16x16x16)\n";

    const int Nx = 16, Ny = 16, Nz = 16;
    const int Nc = Nx / 2 + 1;
    const int N_real = Nx * Ny * Nz;
    const int N_complex = Nc * Ny * Nz;
    const Real tol = std::is_same_v<Real, float> ? Real(1.0e-4) : Real(1.0e-11);

    std::vector<Real> h_real(N_real);
    for (int k = 0; k < Nz; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                h_real[(k * Ny + j) * Nx + i] =
                    std::sin(2.0 * M_PI * 2 * i / Nx) *
                    std::cos(2.0 * M_PI * 3 * j / Ny) *
                    std::sin(2.0 * M_PI * 1 * k / Nz);
            }
        }
    }
    std::vector<Real> h_orig(h_real);

    Gpu::DeviceVector<Real> d_real(N_real);
    Gpu::DeviceVector<Complex> d_complex(N_complex);
    Gpu::htod_memcpy(d_real.data(), h_real.data(), N_real * sizeof(Real));

    IntVect real_size(Nx, Ny, Nz);
    auto fwd = ablastr::math::anyfft::CreatePlan(
        real_size, d_real.data(), d_complex.data(),
        ablastr::math::anyfft::direction::R2C, 3);
    auto bwd = ablastr::math::anyfft::CreatePlan(
        real_size, d_real.data(), d_complex.data(),
        ablastr::math::anyfft::direction::C2R, 3);

    ablastr::math::anyfft::Execute(fwd);
    ablastr::math::anyfft::Execute(bwd);

    Gpu::dtoh_memcpy(h_real.data(), d_real.data(), N_real * sizeof(Real));

    Real max_err = 0.0;
    Real N_total = Real(Nx * Ny * Nz);
    for (int i = 0; i < N_real; ++i) {
        Real expected = h_orig[i] * N_total;
        Real err = std::abs(h_real[i] - expected);
        Real scale = std::max(std::abs(expected), Real(1.0));
        max_err = std::max(max_err, err / scale);
    }

    ablastr::math::anyfft::DestroyPlan(fwd);
    ablastr::math::anyfft::DestroyPlan(bwd);

    bool pass = (max_err < tol);
    amrex::Print() << "  max relative error = " << max_err
                   << " (tol = " << tol << ") ... "
                   << (pass ? "PASS" : "FAIL") << "\n";
    return pass;
#endif
}

// ---------------------------------------------------------------------------
int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    bool ok = true;
    {
        amrex::Print() << "\n========================================\n";
        amrex::Print() << "  fft_roundtrip: AnyFFT precision test\n";
        amrex::Print() << "========================================\n";

#ifdef AMREX_USE_FLOAT
        amrex::Print() << "  Precision: SINGLE\n";
#else
        amrex::Print() << "  Precision: DOUBLE\n";
#endif

#if defined(AMREX_USE_CUDA)
        amrex::Print() << "  Backend: cuFFT\n";
#elif defined(AMREX_USE_HIP)
        amrex::Print() << "  Backend: rocFFT\n";
#elif defined(AMREX_USE_SYCL)
        amrex::Print() << "  Backend: MKL DFT (oneAPI)\n";
#else
        amrex::Print() << "  Backend: FFTW\n";
#endif
        amrex::Print() << "  AMREX_SPACEDIM = " << AMREX_SPACEDIM << "\n";

        ablastr::math::anyfft::setup();

        ok &= test_roundtrip_1d();
        ok &= test_spectral_peaks_1d();
        ok &= test_roundtrip_2d();
        ok &= test_roundtrip_3d();

        ablastr::math::anyfft::cleanup();

        amrex::Print() << "\n----------------------------------------\n";
        amrex::Print() << "  Overall: " << (ok ? "PASS" : "FAIL") << "\n";
        amrex::Print() << "----------------------------------------\n\n";
    }

    amrex::Finalize();
    return ok ? 0 : 1;
}
