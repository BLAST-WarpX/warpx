#ifndef WARPX_HOST_RANDOM_STATE_H_
#define WARPX_HOST_RANDOM_STATE_H_

#include <string>

namespace warpx::checkpoint
{
    /** Write the AMReX host generators if requested in the input. */
    void write_host_random_state (std::string const& checkpoint);

    /** Resume the saved per-rank host streams on a compatible process layout. */
    void restore_host_random_state (std::string const& checkpoint, int step);
}

#endif
