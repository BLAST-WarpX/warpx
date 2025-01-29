#ifndef WARPX_PinnedMemoryParticleContainer_H_
#define WARPX_PinnedMemoryParticleContainer_H_

#include "NamedComponentParticleContainer.H"

using PinnedMemoryParticleContainer = NamedComponentParticleContainer<amrex::PinnedArenaAllocator>;

#endif //WARPX_PinnedMemoryParticleContainer_H_
