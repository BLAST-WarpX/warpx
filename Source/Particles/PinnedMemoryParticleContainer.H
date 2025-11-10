#ifndef WARPX_PinnedMemoryParticleContainer_H_
#define WARPX_PinnedMemoryParticleContainer_H_

#include "WarpXParticleContainer.H"

using PinnedMemoryParticleContainer = amrex::ParticleContainerPureSoA<PIdx::nattribs, 0, amrex::PolymorphicArenaAllocator>;

#endif //WARPX_PinnedMemoryParticleContainer_H_
