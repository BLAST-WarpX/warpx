/* Copyright 2025 Luca Fedeli
 *
 * This file is part of WarpX.
 * Based on PICSAR QED.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef ABLASTR_QED_LOOKUP_TABLES_
#define ABLASTR_QED_LOOKUP_TABLES_

#include "ablastr/utils/Serialization.H"
#include "ablastr/utils/TextMsg.H"

#include <AMReX_Extension.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Vector.H>

#include <algorithm>

namespace ablastr::qed
{

    /**
    * This class implements a generic equispaced 1D lookup table
    * for a function f(x)
    *
    * @tparam RealType the floating point type to be used (e.g. double or float)
    */
    template <typename RealType>
    class equispaced_1d_table
    {

    public:

        /**
        * Constructor (not usable on GPUs)
        *
        * @param[in] x_min the leftmost extreme of the x coordinates
        * @param[in] x_max the rightmost extreme of the x coordinates
        * @param[in] values the values of the function that should be interpolated (passed by const ref)
        */
        equispaced_1d_table(
           RealType x_min, RealType x_max, const amrex::Gpu::DeviceVector<RealType>& values):
           m_x_min{x_min}, m_x_max{x_max}, m_values{values}
        {
            m_how_many_x = static_cast<int>(values.size());
            m_x_size     = x_max - x_min;
            m_dx         = m_x_size/(m_how_many_x-1);
        }

        /**
        * Constructor (not usable on GPUs)
        *
        * @param[in] x_min the leftmost extreme of the x coordinates
        * @param[in] x_max the rightmost extreme of the x coordinates
        * @param[in] values the values of the function that should be interpolated (passed by move)
        */
        equispaced_1d_table(
            RealType x_min, RealType x_max, amrex::Gpu::DeviceVector<RealType>&& values):
            m_x_min{x_min}, m_x_max{x_max}, m_values{values}
        {
            m_how_many_x = static_cast<int>(values.size());
            m_x_size     = x_max - x_min;
            m_dx         = m_x_size/(m_how_many_x-1);
        }


        /**
        * Empty constructor
        */
        constexpr
        equispaced_1d_table(){}

        /**
        * Constructor from byte array (not usable on GPUs)
        *
        * @param[in] raw_data a const reference to a byte vector
        */
        equispaced_1d_table(const amrex::Vector<char>& raw_data)
        {
            using namespace ablastr::utils;
            using namespace amrex;

            constexpr size_t min_size =
                sizeof(char)+ //double or float
                sizeof(m_x_min)+sizeof(m_x_max)+
                sizeof(m_how_many_x) + sizeof(m_dx);

            ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE(
                raw_data.size() >= min_size,
                "Binary data is too small to be a 1D table.");

            auto it_raw_data = raw_data.begin();

            ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE(
                serialization::get_out<char>(it_raw_data) == static_cast<char>(sizeof(RealType)),
                "Mismatch between RealType used to write and to read the 1D table");

            const auto get_out = [&it_raw_data](auto& arg){
                arg = serialization::get_out<decltype(arg)>(it_raw_data);};

            get_out(m_x_min);
            get_out(m_x_max);
            get_out(m_how_many_x);
            get_out(m_dx);

            m_x_size = m_x_max - m_x_min;

            ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE(
                m_x_size > 0 && m_how_many_x > 0,
                "raw_data contains invalid data.");

            const auto vals = serialization::get_n_out<RealType>(it_raw_data, m_how_many_x);

            m_values = Gpu::DeviceVector<RealType>(m_how_many_x);
            Gpu::copy(Gpu::hostToDevice, vals.begin(), vals.end(), m_values.begin());
            Gpu::synchronize();
        }

        /**
        * Operator ==
        *
        * @param[in] rhs a const reference to a 1D table of the same type
        * @return true if rhs and *this are equal. false otherwise
        */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        bool operator== (
            const equispaced_1d_table<RealType> &rhs) const
        {
            return
                (m_x_min == rhs.m_x_min) &&
                (m_x_max == rhs.m_x_max) &&
                (m_x_size == rhs.m_x_size) &&
                (m_how_many_x == rhs.m_how_many_x) &&
                (m_dx == rhs.m_dx) &&
                (m_values == rhs.m_values);
        }

        /**
        * Returns the i-th coodinate along x axis
        *
        * @param[in] i the index of the desired coordinate
        * @return the i-th coordinate
        */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        RealType get_x_coord(const int i) const noexcept
        {
            return i*m_dx + m_x_min;
        }

        /**
        * Returns the i-th value
        *
        * @param[in] i the index of the desired value
        * @return the i-th value
        */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        RealType get_val(const size_t i) const noexcept
        {
            return m_values[i];
        }

        /**
        * Returns the number of points along x
        *
        * @return the number of points along x
        */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        int get_how_many_x() const noexcept
        {
           return m_how_many_x;
        }

        /**
        * Returns the minimum x coordinate
        *
        * @return the minimum x coordinate
        */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        RealType get_x_min() const noexcept
        {
           return m_x_min;
        }

        /**
        * Returns the maximum x coordinate
        *
        * @return the maximum x coordinate
        */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        RealType get_x_max() const noexcept
        {
            return m_x_max;
        }

        /**
        * Returns the size along x
        *
        * @return the size along x
        */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        RealType get_x_size() const noexcept
        {
            return m_x_size;
        }

       /**
       * Returns the size of the steps along x
       *
       * @return the size of the steps along x
       */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        RealType get_dx() const noexcept
        {
            return m_dx;
        }

        /**
        * Returns an amrex::Vector containing all the coordinates (not usable on
        * GPUs).
        *
        * @return all the coordinates
        */
        amrex::Vector<RealType> get_all_coordinates()  const noexcept
        {
            auto all_coords = amrex::Vector<RealType>(m_how_many_x);
            std::generate(all_coords.being(), all_coords.end(),
                mutable [i=0](){return get_x_coord(i++);});
            return all_coords;
        }

        /**
        * Returns a const reference to the underlying Vector holding value data
        *
        * @return a const reference to the underlying Vector holding value data
        */
        const amrex::Vector<RealType>& get_values_reference() const noexcept
        {
           return m_values;
        }

        /**
        * Perfoms a linear interpolation at a given position
        * Warning: where_x must be within [x_min, x_max]. No safety check
        * is performed!
        *
        * @param[in] where_x where to perform the interpolation
        * @return the interpolated value
        */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        RealType interp(const RealType where_x) const noexcept
        {
            const auto idx_left = static_cast<int>(
                std::floor((m_how_many_x-1)*(where_x-m_x_min)/m_x_size));

            if (idx_left == (m_how_many_x-1)) AMREX_UNLIKELY {
                return  m_values[m_how_many_x-1];
            }
            const auto idx_right = idx_left + 1;

            const auto xleft  = get_x_coord(idx_left);
            const auto xright = get_x_coord(idx_right);
            const auto yleft  = m_values[idx_left];
            const auto yright = m_values[idx_right];

            return utils::linear_interp(xleft, xright, yleft, yright, where_x);
        }

       /**
       * Overwrites the i-th value (not designed for GPU usage)
       * Warning: no safety check on i is performed!
       *
       * @param[in] i index of the value to be overwritten
       * @param[in] what value to be written at position i
       */
       PXRMP_FORCE_INLINE
       void set_val(int i, RealType what)
       {
           m_values[i] = what;
           //VectorType may need a call to a user-defined method for CPU-GPU synchronization
           details::aux_sync_vec(m_values);
       }

       /**
       * Overwrites all values
       *
       * @param[in] new_values
       */
       PXRMP_FORCE_INLINE
       void set_all_vals(const std::vector<RealType>& new_values)
       {
           if (new_values.size() != m_values.size()){
               throw std::runtime_error("Mismatch new_values length \
                   and m_values length");
           }
           std::copy(new_values.begin(), new_values.end(), m_values.begin());
           //VectorType may need a call to a user-defined method for CPU-GPU synchronization
           details::aux_sync_vec(m_values);
       }


       /**
       * Returns a byte vector containing all the raw data of the
       * table (not usable on GPUs)
       *
       * @return a byte vector containing table data
       */
       std::vector<char> serialize() const
       {
           auto raw_data = std::vector<char>{};

           utils::serialization::put_in(
               static_cast<char>(sizeof(RealType)), raw_data);
           utils::serialization::put_in(m_x_min, raw_data);
           utils::serialization::put_in(m_x_max, raw_data);
           utils::serialization::put_in(m_how_many_x, raw_data);
           utils::serialization::put_in(m_dx, raw_data);
           for (auto val : m_values){
               utils::serialization::put_in(val, raw_data);
           }

           return raw_data;
       }

   private:
       auto m_x_min  = static_cast<RealType>(0.0);     /* minimum x coordinate */
       auto m_x_max  = static_cast<RealType>(0.0);     /* maximum x coordinate */
       auto m_x_size = static_cast<RealType>(0.0);     /* size along x */
       int  m_how_many_x  = 0;                         /* how many values are stored in the table */
       auto m_dx     = static_cast<RealType>(0.0);     /* size of the step along x */
       auto m_values = amrex::GpuArray<RealType, 3>{}; /* values f(x) */
   };

}


#endif //ABLASTR_QED_LOOKUP_TABLES_
