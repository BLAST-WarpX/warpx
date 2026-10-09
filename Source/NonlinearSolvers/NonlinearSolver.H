#ifndef WARPX_NONLINEAR_SOLVER_H_
#define WARPX_NONLINEAR_SOLVER_H_

#include "PreconditionerLibrary.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>

#include <AMReX_BaseFwd.H>

#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>

/**
 * \brief Input parameters common to all nonlinear solvers, read from the
 * ParmParse prefix of the solver (e.g., "newton" or "picard").
 */
struct NonlinearSolverParams
{
    //! Verbosity flag.
    bool verbose = true;
    //! Relative tolerance.
    amrex::Real rtol = 1.0e-6;
    //! Absolute tolerance.
    amrex::Real atol = 0.;
    //! Maximum iterations.
    int maxits = 100;
    //! Flag to determine whether convergence is required.
    bool require_convergence = true;
    //! Diagnostic file and the time-step interval at which it is written.
    std::string diagnostic_file;
    int diagnostic_interval = 1;

    void Parse (const std::string& a_prefix)
    {
        const amrex::ParmParse pp(a_prefix);
        pp.query("verbose",             verbose);
        pp.query("absolute_tolerance",  atol);
        pp.query("relative_tolerance",  rtol);
        pp.query("max_iterations",      maxits);
        pp.query("require_convergence", require_convergence);
        pp.query("diagnostic_file",     diagnostic_file);
        pp.query("diagnostic_interval", diagnostic_interval);
    }

    void Print (const std::string& a_name) const
    {
        amrex::Print() << a_name << " verbose:             " << (verbose?"true":"false") << "\n";
        amrex::Print() << a_name << " max iterations:      " << maxits << "\n";
        amrex::Print() << a_name << " relative tolerance:  " << rtol << "\n";
        amrex::Print() << a_name << " absolute tolerance:  " << atol << "\n";
        amrex::Print() << a_name << " require convergence: " << (require_convergence?"true":"false") << "\n";
    }
};

/**
 * \brief Top-level class for the nonlinear solver
 *
 *    This class is templated on a vector class Vec, and an operator class Ops.
 *
 *    The Ops class must have the following function:
 *    ComputeRHS( R_vec, U_vec, time, nl_iter, from_jacobian ),
 *    where U_vec and R_vec are of type Vec.
 *
 *    The Vec class must have basic math operators, such as Copy, +=, -=,
 *    increment(), linComb(), scale(), etc.. See WarpXSolverVec.H for an example.
 */

template<class Vec, class Ops>
class NonlinearSolver
{
public:

    NonlinearSolver() = default;

    virtual ~NonlinearSolver() = default;

    // Prohibit Move and Copy operations
    NonlinearSolver(const NonlinearSolver&) = delete;
    NonlinearSolver& operator=(const NonlinearSolver&) = delete;
    NonlinearSolver(NonlinearSolver&&) noexcept = delete;
    NonlinearSolver& operator=(NonlinearSolver&&) noexcept = delete;

    /**
     * \brief Read user-provided parameters that control the nonlinear solver.
     * Allocate intermediate data containers needed by the solver. For Newton,
     * setup the linear solver for computing the Newton step.
     */
    virtual void Define ( const Vec&,
                          Ops* ) = 0;

    /**
     * \brief Check if the nonlinear solver has been defined.
     */
    [[nodiscard]] bool IsDefined () const { return m_is_defined; }

    /**
     * \brief Solve the specified nonlinear equation for U.
     *  Picard: U = b + R(U).
     *  Newton: F(U) = U - b - R(U) = 0.
     */
    virtual void Solve (Vec&,
                  const Vec&,
                        amrex::Real,
                        amrex::Real,
                        int,
                        bool) const = 0;

    /**
     * \brief Return the nonlinear solver exit status.
     */
    int GetExitStatus() const { return m_exit_status; }

    /**
     * \brief Print parameters used by the nonlinear solver.
     */
    virtual void PrintParams () const = 0;

    /**
     * \brief Return the convergence parameters used by the nonlinear solver.
     */
    void GetSolverParams (amrex::Real&  a_rtol,
                          amrex::Real&  a_atol,
                          int&          a_maxits) const
    {
        a_rtol = m_params.rtol;
        a_atol = m_params.atol;
        a_maxits = m_params.maxits;
    }

    /**
     * \brief Return preconditioner type.
     */
    virtual PreconditionerType GetPreconditionerType () const { return PreconditionerType::none; }

    /**
     * \brief Allow caller to dynamically change the verbosity flag. For
     * example, one may want to only print solver information every 100 steps.
     */
    virtual void Verbose ( bool  a_verbose ) { m_params.verbose = a_verbose; }

protected:

    /**
     * \brief Create the diagnostic file and write its header. The columns
     * common to all solvers are followed by a_extra_columns.
     */
    void InitDiagnosticFile (const std::vector<std::string>& a_extra_columns = {}) const
    {
        if (!amrex::ParallelDescriptor::IOProcessor()
            || m_params.diagnostic_file.empty()
            || amrex::FileExists(m_params.diagnostic_file)) { return; }

        std::filesystem::path const diagnostic_path(m_params.diagnostic_file);
        std::filesystem::path const diagnostic_dir = diagnostic_path.parent_path();
        if (!diagnostic_dir.empty()) {
            std::filesystem::create_directories(diagnostic_dir);
        }

        std::ofstream diagnostic_file{m_params.diagnostic_file, std::ofstream::out | std::ofstream::trunc};
        std::vector<std::string> columns =
            {"step()", "time(s)", "iters", "total_iters", "norm_abs", "norm_rel"};
        columns.insert(columns.end(), a_extra_columns.begin(), a_extra_columns.end());
        diagnostic_file << "#";
        for (std::size_t c = 0; c < columns.size(); ++c) {
            diagnostic_file << (c > 0 ? " " : "") << "[" << c << "]" << columns[c];
        }
        diagnostic_file << "\n";
        diagnostic_file.close();
    }

    /**
     * \brief Append a row to the diagnostic file at the diagnostic interval:
     * the step index, the time, and a_values.
     */
    template <typename... Args>
    void WriteDiagnostics (int a_step, amrex::Real a_time, const Args&... a_values) const
    {
        if (m_params.diagnostic_file.empty() || !amrex::ParallelDescriptor::IOProcessor() ||
            !((a_step+1)%m_params.diagnostic_interval==0 || a_step==0)) { return; }

        std::ofstream diagnostic_file{m_params.diagnostic_file, std::ofstream::out | std::ofstream::app};
        diagnostic_file << std::setprecision(14);
        diagnostic_file << a_step+1 << " " << a_time;
        ((diagnostic_file << " " << a_values), ...);
        diagnostic_file << "\n";
        diagnostic_file.close();
    }

    bool m_is_defined = false;
    NonlinearSolverParams m_params;
    bool m_usePC = false;

    mutable int m_exit_status = 0;

};

#endif
