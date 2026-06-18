/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_COLLISION_FUNC_BASE_H_
#define WARPX_COLLISION_FUNC_BASE_H_

class CollisionFuncBase
{
public:

    virtual ~CollisionFuncBase () = default;

    //! Optional hook to allocate additional data structures.
    virtual void AllocData () {}

};

#endif /* WARPX_COLLISION_FUNC_BASE_H_ */
