/* Copyright 2021 Luca Fedeli Maxence Thevenet
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "SpeciesPhysicalProperties.H"

#include "Utils/WarpXConst.H"

#include <AMReX_AmrCore.H>

#include <limits>
#include <map>
#include <string>

using namespace species;

namespace {
    struct Properties
    {
        amrex::Real mass;
        amrex::Real charge;
    };

    const auto string_to_species = std::map<std::string, PhysicalSpecies>{
        {"unspecified", PhysicalSpecies::unspecified},
        {"electron"   , PhysicalSpecies::electron},
        {"positron"   , PhysicalSpecies::positron},
        {"muon"       , PhysicalSpecies::muon},
        {"antimuon"   , PhysicalSpecies::antimuon},
        {"photon"     , PhysicalSpecies::photon},
        {"neutron"    , PhysicalSpecies::neutron},
        {"hydrogen"   , PhysicalSpecies::hydrogen},
        {"hydrogen1"  , PhysicalSpecies::hydrogen1},
        {"protium"    , PhysicalSpecies::hydrogen1},
        {"hydrogen2"  , PhysicalSpecies::hydrogen2},
        {"deuterium"  , PhysicalSpecies::hydrogen2},
        {"hydrogen3"  , PhysicalSpecies::hydrogen3},
        {"tritium"    , PhysicalSpecies::hydrogen3},
        {"proton"     , PhysicalSpecies::proton},
        {"helium"     , PhysicalSpecies::helium},
        {"helium3"    , PhysicalSpecies::helium3},
        {"helium4"    , PhysicalSpecies::helium4},
        {"alpha"      , PhysicalSpecies::alpha},
        {"lithium"    , PhysicalSpecies::lithium},
        {"lithium6"   , PhysicalSpecies::lithium6},
        {"lithium7"   , PhysicalSpecies::lithium7},
        {"beryllium"  , PhysicalSpecies::beryllium},
        {"beryllium9"  , PhysicalSpecies::beryllium9},
        {"boron"      , PhysicalSpecies::boron},
        {"boron10"    , PhysicalSpecies::boron10},
        {"boron11"    , PhysicalSpecies::boron11},
        {"carbon"     , PhysicalSpecies::carbon},
        {"carbon12"   , PhysicalSpecies::carbon12},
        {"carbon13"   , PhysicalSpecies::carbon13},
        {"carbon14"   , PhysicalSpecies::carbon14},
        {"nitrogen"   , PhysicalSpecies::nitrogen},
        {"nitrogen14" , PhysicalSpecies::nitrogen14},
        {"nitrogen15" , PhysicalSpecies::nitrogen15},
        {"oxygen"     , PhysicalSpecies::oxygen},
        {"oxygen16"   , PhysicalSpecies::oxygen16},
        {"oxygen17"   , PhysicalSpecies::oxygen17},
        {"oxygen18"   , PhysicalSpecies::oxygen18},
        {"fluorine"   , PhysicalSpecies::fluorine},
        {"fluorine19" , PhysicalSpecies::fluorine19},
        {"neon"       , PhysicalSpecies::neon},
        {"neon20"     , PhysicalSpecies::neon20},
        {"neon21"     , PhysicalSpecies::neon21},
        {"neon22"     , PhysicalSpecies::neon22},
        {"aluminium"  , PhysicalSpecies::aluminium},
        {"argon"      , PhysicalSpecies::argon},
        {"copper"     , PhysicalSpecies::copper},
        {"xenon"      , PhysicalSpecies::xenon},
        {"gold"       , PhysicalSpecies::gold},
    };

    const auto species_to_string = std::map<PhysicalSpecies, std::string>{
        {PhysicalSpecies::unspecified, "unspecified"},
        {PhysicalSpecies::electron   , "electron"},
        {PhysicalSpecies::positron   , "positron"},
        {PhysicalSpecies::muon       , "muon"},
        {PhysicalSpecies::antimuon   , "antimuon"},
        {PhysicalSpecies::photon     , "photon"},
        {PhysicalSpecies::neutron    , "neutron"},
        {PhysicalSpecies::hydrogen   , "hydrogen"},
        {PhysicalSpecies::hydrogen1  , "hydrogen1"},
        {PhysicalSpecies::hydrogen2  , "hydrogen2"},
        {PhysicalSpecies::hydrogen3  , "hydrogen3"},
        {PhysicalSpecies::proton     , "proton"},
        {PhysicalSpecies::helium     , "helium"},
        {PhysicalSpecies::helium3    , "helium3"},
        {PhysicalSpecies::helium4    , "helium4"},
        {PhysicalSpecies::alpha      , "alpha"},
        {PhysicalSpecies::lithium    , "lithium"},
        {PhysicalSpecies::lithium6   , "lithium6"},
        {PhysicalSpecies::lithium7   , "lithium7"},
        {PhysicalSpecies::beryllium  , "beryllium"},
        {PhysicalSpecies::beryllium9 , "beryllium9"},
        {PhysicalSpecies::boron      , "boron"},
        {PhysicalSpecies::boron10    , "boron10"},
        {PhysicalSpecies::boron11    , "boron11"},
        {PhysicalSpecies::carbon     , "carbon"},
        {PhysicalSpecies::carbon12   , "carbon12"},
        {PhysicalSpecies::carbon13   , "carbon13"},
        {PhysicalSpecies::carbon14   , "carbon14"},
        {PhysicalSpecies::nitrogen   , "nitrogen"},
        {PhysicalSpecies::nitrogen14 , "nitrogen14"},
        {PhysicalSpecies::nitrogen15 , "nitrogen15"},
        {PhysicalSpecies::oxygen     , "oxygen"},
        {PhysicalSpecies::oxygen16   , "oxygen16"},
        {PhysicalSpecies::oxygen17   , "oxygen17"},
        {PhysicalSpecies::oxygen18   , "oxygen18"},
        {PhysicalSpecies::fluorine   , "fluorine"},
        {PhysicalSpecies::fluorine19 , "fluorine19"},
        {PhysicalSpecies::neon       , "neon"},
        {PhysicalSpecies::neon20     , "neon20"},
        {PhysicalSpecies::neon21     , "neon21"},
        {PhysicalSpecies::neon22     , "neon22"},
        {PhysicalSpecies::aluminium  , "aluminium"},
        {PhysicalSpecies::argon      , "argon"},
        {PhysicalSpecies::copper     , "copper"},
        {PhysicalSpecies::xenon      , "xenon"},
        {PhysicalSpecies::gold       , "gold"}
    };

    constexpr auto quiet_NaN = std::numeric_limits<amrex::Real>::quiet_NaN();

    // The atomic mass data below is from these NIST pages
    // https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=&ascii=ascii2&isotype=some
    // https://physics.nist.gov/cgi-bin/cuu/Value?malu
    const
    std::map<PhysicalSpecies,Properties> species_to_properties
    {
        {PhysicalSpecies::unspecified, Properties{
            quiet_NaN,
            quiet_NaN}},
        {PhysicalSpecies::electron, Properties{
            PhysConst::m_e,
            -PhysConst::q_e}},
        {PhysicalSpecies::positron, Properties{
            PhysConst::m_e,
            PhysConst::q_e}},
        {PhysicalSpecies::muon, Properties{
            amrex::Real(206.7682830) * PhysConst::m_e,
            -PhysConst::q_e}},
        {PhysicalSpecies::antimuon, Properties{
            amrex::Real(206.7682830) * PhysConst::m_e,
            PhysConst::q_e}},
        {PhysicalSpecies::photon, Properties{
            amrex::Real(0.0),
            amrex::Real(0.0)}},
        {PhysicalSpecies::neutron, Properties{
            amrex::Real(1.0013784193052508) * PhysConst::m_p,
            amrex::Real(0.0)}},
        {PhysicalSpecies::proton, Properties{
             PhysConst::m_p,
             PhysConst::q_e}},
        {PhysicalSpecies::hydrogen, Properties{
             amrex::Real(1.00797) * PhysConst::m_u,
             amrex::Real(1) * PhysConst::q_e}},
        {PhysicalSpecies::hydrogen1, Properties{
             amrex::Real(1.00782503223) * PhysConst::m_u,
             amrex::Real(1) * PhysConst::q_e}},
        {PhysicalSpecies::hydrogen2, Properties{
             amrex::Real(2.01410177812) * PhysConst::m_u,
             amrex::Real(1) * PhysConst::q_e}},
        {PhysicalSpecies::hydrogen3, Properties{
             amrex::Real(3.0160492779) * PhysConst::m_u,
             amrex::Real(1) * PhysConst::q_e}},
        {PhysicalSpecies::helium, Properties{
             amrex::Real(4.002602) * PhysConst::m_u,
             amrex::Real(2) * PhysConst::q_e}},
        {PhysicalSpecies::helium3, Properties{
             amrex::Real(3.0160293201) * PhysConst::m_u,
             amrex::Real(2) * PhysConst::q_e}},
        {PhysicalSpecies::helium4, Properties{
             amrex::Real(4.00260325413) * PhysConst::m_u,
             amrex::Real(2) * PhysConst::q_e}},
        {PhysicalSpecies::alpha, Properties{
             amrex::Real(4.001506179127) * PhysConst::m_u,
             amrex::Real(2) * PhysConst::q_e}},
        {PhysicalSpecies::lithium, Properties{
             amrex::Real(6.967) * PhysConst::m_u,
             amrex::Real(3) * PhysConst::q_e}},
        {PhysicalSpecies::lithium6, Properties{
             amrex::Real(6.0151228874) * PhysConst::m_u,
             amrex::Real(3) * PhysConst::q_e}},
        {PhysicalSpecies::lithium7, Properties{
             amrex::Real(7.0160034366) * PhysConst::m_u,
             amrex::Real(3) * PhysConst::q_e}},
        {PhysicalSpecies::beryllium, Properties{
             amrex::Real(9.0121831) * PhysConst::m_u,
             amrex::Real(4) * PhysConst::q_e}},
        {PhysicalSpecies::beryllium9, Properties{
             amrex::Real(9.012183065) * PhysConst::m_u,
             amrex::Real(4) * PhysConst::q_e}},
        {PhysicalSpecies::boron, Properties{
             amrex::Real(10.813) * PhysConst::m_u,
             amrex::Real(5) * PhysConst::q_e}},
        {PhysicalSpecies::boron10, Properties{
             amrex::Real(10.01293695) * PhysConst::m_u,
             amrex::Real(5) * PhysConst::q_e}},
        {PhysicalSpecies::boron11, Properties{
             amrex::Real(11.00930536) * PhysConst::m_u,
             amrex::Real(5) * PhysConst::q_e}},
        {PhysicalSpecies::carbon, Properties{
             amrex::Real(12.0106) * PhysConst::m_u,
             amrex::Real(6) * PhysConst::q_e}},
        {PhysicalSpecies::carbon12, Properties{
             amrex::Real(12.0000000) * PhysConst::m_u,
             amrex::Real(6) * PhysConst::q_e}},
        {PhysicalSpecies::carbon13, Properties{
             amrex::Real(13.00335483507) * PhysConst::m_u,
             amrex::Real(6) * PhysConst::q_e}},
        {PhysicalSpecies::carbon14, Properties{
             amrex::Real(14.0032419884) * PhysConst::m_u,
             amrex::Real(6) * PhysConst::q_e}},
        {PhysicalSpecies::nitrogen, Properties{
             amrex::Real(14.00685) * PhysConst::m_u,
             amrex::Real(7) * PhysConst::q_e}},
        {PhysicalSpecies::nitrogen14, Properties{
             amrex::Real(14.00307400443) * PhysConst::m_u,
             amrex::Real(7) * PhysConst::q_e}},
        {PhysicalSpecies::nitrogen15, Properties{
             amrex::Real(15.00010889888) * PhysConst::m_u,
             amrex::Real(7) * PhysConst::q_e}},
        {PhysicalSpecies::oxygen, Properties{
             amrex::Real(15.9994) * PhysConst::m_u,
             amrex::Real(8) * PhysConst::q_e}},
        {PhysicalSpecies::oxygen16, Properties{
             amrex::Real(15.99491461957) * PhysConst::m_u,
             amrex::Real(8) * PhysConst::q_e}},
        {PhysicalSpecies::oxygen17, Properties{
             amrex::Real(16.99913175650) * PhysConst::m_u,
             amrex::Real(8) * PhysConst::q_e}},
        {PhysicalSpecies::oxygen18, Properties{
             amrex::Real(17.99915961286) * PhysConst::m_u,
             amrex::Real(8) * PhysConst::q_e}},
        {PhysicalSpecies::fluorine, Properties{
             amrex::Real(18.998403163) * PhysConst::m_u,
             amrex::Real(9) * PhysConst::q_e}},
        {PhysicalSpecies::fluorine19, Properties{
             amrex::Real(18.99840316273) * PhysConst::m_u,
             amrex::Real(9) * PhysConst::q_e}},
        {PhysicalSpecies::neon, Properties{
             amrex::Real(20.1797) * PhysConst::m_u,
             amrex::Real(10) * PhysConst::q_e}},
        {PhysicalSpecies::neon20, Properties{
             amrex::Real(19.9924401762) * PhysConst::m_u,
             amrex::Real(10) * PhysConst::q_e}},
        {PhysicalSpecies::neon21, Properties{
             amrex::Real(20.993846685) * PhysConst::m_u,
             amrex::Real(10) * PhysConst::q_e}},
        {PhysicalSpecies::neon22, Properties{
             amrex::Real(21.991385114) * PhysConst::m_u,
             amrex::Real(10) * PhysConst::q_e}},
        {PhysicalSpecies::aluminium, Properties{
             amrex::Real(26.98153853) * PhysConst::m_u,
             amrex::Real(13) * PhysConst::q_e}},
        {PhysicalSpecies::argon, Properties{
             amrex::Real(39.948) * PhysConst::m_u,
             amrex::Real(18) * PhysConst::q_e}},
        {PhysicalSpecies::copper, Properties{
             amrex::Real(63.546) * PhysConst::m_u,
             amrex::Real(29) * PhysConst::q_e}},
        {PhysicalSpecies::xenon, Properties{
             amrex::Real(131.293) * PhysConst::m_u,
             amrex::Real(54) * PhysConst::q_e}},
        {PhysicalSpecies::gold, Properties{
             amrex::Real(196.966569) * PhysConst::m_u,
             amrex::Real(79) * PhysConst::q_e}},
    };
}

namespace species
{
    std::optional<PhysicalSpecies> from_string(const std::string& species)
    {
        const auto phys_spec = string_to_species.find(species);
        return (phys_spec != string_to_species.end())?
            std::make_optional(phys_spec->second) : std::nullopt;
    }

    amrex::Real get_charge (const PhysicalSpecies& ps)
    {
        return species_to_properties.at(ps).charge;
    }

    amrex::Real get_mass (const PhysicalSpecies& ps)
    {
        return species_to_properties.at(ps).mass;
    }

    std::string get_name (const PhysicalSpecies& ps)
    {
        return species_to_string.at(ps);
    }
}
