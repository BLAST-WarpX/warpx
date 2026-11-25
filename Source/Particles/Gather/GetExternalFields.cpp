#include "Particles/Gather/GetExternalFields.H"

#include "AcceleratorLattice/AcceleratorLattice.H"

#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <AMReX_Vector.H>

#include <string>

using namespace amrex::literals;

GetExternalEBField::GetExternalEBField (const WarpXParIter& a_pti, long a_offset,
                                      std::optional<amrex::Real> a_field_time) noexcept
{
    auto& warpx = WarpX::GetInstance();
    auto& mypc = warpx.GetPartContainer();

    const int lev = a_pti.GetLevel();

    const auto& dts = warpx.getdt();

    AcceleratorLattice const & accelerator_lattice = warpx.get_accelerator_lattice(lev);
    if (accelerator_lattice.m_lattice_defined) {
        d_lattice_element_finder = accelerator_lattice.GetFinderDeviceInstance(a_pti, static_cast<int>(a_offset), dts);
    }

    m_gamma_boost = WarpX::gamma_boost;
    m_uz_boost = std::sqrt(WarpX::gamma_boost*WarpX::gamma_boost - 1._prt)*PhysConst::c;

    m_Etype = Unknown;
    m_Btype = Unknown;

    if (mypc.m_E_ext_particle_s == "none") { m_Etype = None; }
    if (mypc.m_B_ext_particle_s == "none") { m_Btype = None; }

    // These lines will be removed once the user interface is redefined and the CI tests updated
    if (mypc.m_E_ext_particle_s == "constant") { m_Etype = None; }
    if (mypc.m_B_ext_particle_s == "constant") { m_Btype = None; }

    if (mypc.m_E_ext_particle_s == "parse_e_ext_particle_function" ||
        mypc.m_B_ext_particle_s == "parse_b_ext_particle_function" ||
        mypc.m_E_ext_particle_s == "repeated_plasma_lens" ||
        mypc.m_B_ext_particle_s == "repeated_plasma_lens")
    {
        // Parsed fields use the implicit stage time when supplied. Keep a separate
        // m_time to preserve the existing interval timing of repeated plasma lenses;
        // this does not establish that plasma lenses support implicit stepping.
        // m_time and m_parser_time are the same when called from the explicit path.
        // TODO: Determine whether repeated plasma lenses support implicit stepping.
        // Either implement consistent stage/substep timing or reject that combination,
        // then reconsider whether separate times are needed.
        m_time = warpx.gett_new(a_pti.GetLevel());
        m_parser_time = a_field_time.value_or(m_time);
        m_get_position = GetParticlePosition<PIdx>(a_pti, a_offset);
    }

    if (mypc.m_E_ext_particle_s == "parse_e_ext_particle_function")
    {
        m_Etype = ExternalFieldInitType::Parser;
        m_Exfield_partparser = mypc.m_Ex_particle_parser->compile<4>();
        m_Eyfield_partparser = mypc.m_Ey_particle_parser->compile<4>();
        m_Ezfield_partparser = mypc.m_Ez_particle_parser->compile<4>();
    }

    if (mypc.m_B_ext_particle_s == "parse_b_ext_particle_function")
    {
        m_Btype = ExternalFieldInitType::Parser;
        m_Bxfield_partparser = mypc.m_Bx_particle_parser->compile<4>();
        m_Byfield_partparser = mypc.m_By_particle_parser->compile<4>();
        m_Bzfield_partparser = mypc.m_Bz_particle_parser->compile<4>();
    }

    if (mypc.m_E_ext_particle_s == "repeated_plasma_lens" ||
        mypc.m_B_ext_particle_s == "repeated_plasma_lens")
    {
        if (mypc.m_E_ext_particle_s == "repeated_plasma_lens") { m_Etype = RepeatedPlasmaLens; }
        if (mypc.m_B_ext_particle_s == "repeated_plasma_lens") { m_Btype = RepeatedPlasmaLens; }
        m_dt = warpx.getdt(a_pti.GetLevel());
        const auto& attribs = a_pti.GetAttribs();
        m_ux = attribs[PIdx::ux].dataPtr() + a_offset;
        m_uy = attribs[PIdx::uy].dataPtr() + a_offset;
        m_uz = attribs[PIdx::uz].dataPtr() + a_offset;
        m_repeated_plasma_lens_period = mypc.m_repeated_plasma_lens_period;
        m_n_lenses = static_cast<int>(mypc.h_repeated_plasma_lens_starts.size());
        m_repeated_plasma_lens_starts = mypc.d_repeated_plasma_lens_starts.data();
        m_repeated_plasma_lens_lengths = mypc.d_repeated_plasma_lens_lengths.data();
        m_repeated_plasma_lens_strengths_E = mypc.d_repeated_plasma_lens_strengths_E.data();
        m_repeated_plasma_lens_strengths_B = mypc.d_repeated_plasma_lens_strengths_B.data();
    }

    // When the external particle fields are read from file,
    // the external fields are not added directly inside the gather kernel.
    // (Hence of `None`, which ensures that the gather kernel is compiled without support
    // for external fields.) Instead, the external fields are added to the MultiFab
    // Efield_aux and Bfield_aux before the particles gather from these MultiFab.
    if (mypc.m_E_ext_particle_s == "read_from_file") {
        m_Etype = ExternalFieldInitType::None;
    }
    if (mypc.m_B_ext_particle_s == "read_from_file") {
        m_Btype = ExternalFieldInitType::None;
    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_Etype != Unknown, "Unknown E_ext_particle_init_style");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_Btype != Unknown, "Unknown B_ext_particle_init_style");

}
