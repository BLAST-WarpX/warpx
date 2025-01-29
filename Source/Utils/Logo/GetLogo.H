#ifndef WARPX_GET_LOGO_H_
#define WARPX_GET_LOGO_H_

#include <string>

namespace utils::logo
{
        /**
         * \brief provides an ASCII art logo for WarpX
         *
         * \return a string containing an ASCII art logo
         */
        std::string get_logo () noexcept;
} // namespace utils::logo

#endif
