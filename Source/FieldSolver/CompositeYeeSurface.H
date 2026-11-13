/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_COMPOSITE_YEE_SURFACE_H_
#define WARPX_COMPOSITE_YEE_SURFACE_H_

#include <AMReX_REAL.H>
#include <memory>
#include <string>

class WarpX;

/** Experimental single-rank two-level conforming Yee interface on a closed
 * patch. */
class CompositeYeeSurface {
  public:
    explicit CompositeYeeSurface (WarpX& warpx);
    ~CompositeYeeSurface ();
    CompositeYeeSurface (CompositeYeeSurface const&) = delete;
    CompositeYeeSurface& operator= (CompositeYeeSurface const&) = delete;
    CompositeYeeSurface (CompositeYeeSurface&&) = delete;
    CompositeYeeSurface& operator= (CompositeYeeSurface&&) = delete;
    static void WriteCheckpointIdentity (WarpX& w,
                                         std::string const& directory);
    static void CheckRestartIdentity (WarpX& w, std::string const& directory,
                                      bool enabled);
    void RememberElectric ();
    void AdvanceElectric (amrex::Real dt);
    void RestrictMagnetic ();
    void UpdateGatherFields ();
    void AssembleParticleCurrent (amrex::Real dt);

  private:
    struct Implementation;
    std::unique_ptr<Implementation> m_impl;
};
#endif
