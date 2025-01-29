/* Copyright 2023 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef ABLASTR_TIMER_H_
#define ABLASTR_TIMER_H_

namespace ablastr::utils::timer
{

    /**
    * This class implements a simple timer. It allows recording a start time
    * and a stop time, and it provides methods to get the total duration, either
    * local or global (i.e., the maximum duration recorder across all the MPI ranks).
    */
    class Timer
    {
        public:

        /**
        * \brief The constructor.
        */
        Timer() = default;


        /**
        * \brief This function records the start time
        */
        void record_start_time() noexcept;


        /**
        * \brief This function records the stop time
        */
        void record_stop_time() noexcept;


        /**
        * \brief This function returns the duration, calculated as the
        * difference between the stop time and the start time (in seconds).
        *
        * @return the duration
        */
        [[nodiscard]] double get_duration () const noexcept;


        /**
        * \brief This collective function returns the maximum duration
        * recorded across all the MPI ranks (in seconds).
        *
        * @return the maximum duration across all the MPI ranks
        */
        [[nodiscard]] double get_global_duration () const;

        private:

        double m_start_time /*! The start time*/;
        double m_stop_time /*! The stop time*/;
    };

}

#endif //ABLASTR_TIMER_H_
