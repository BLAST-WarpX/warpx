/* Copyright 2025 Luca Fedeli
 *
 * This file is part of WarpX.
 * Based on PICSAR QED.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef ABLASTR_QED_LOOKUP_TABLES_
#define ABLASTR_QED_LOOKUP_TABLES_

#include "ablastr/math/LinearInterpolation.H"
#include "ablastr/utils/Serialization.H"
#include "ablastr/utils/TextMsg.H"

#include <AMReX_Array.H>
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
           const RealType x_min, const RealType x_max,
           const amrex::Gpu::DeviceVector<RealType>& values):
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
            const RealType x_min, const RealType x_max,
            amrex::Gpu::DeviceVector<RealType>&& values):
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
        amrex::Vector<RealType> get_all_coordinates() const noexcept
        {
            using namespace amrex;

            auto all_coords = amrex::Vector<RealType>(m_how_many_x, 0_rt);
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

            return math::linear_interp(xleft, xright, yleft, yright, where_x);
        }


        /**
        * Overwrites all values
        *
        * @param[in] new_values
        */
        void set_all_vals(const amrex::Vector<RealType>& new_values)
        {
            ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE(
                new_values.size() != m_values.size(),
                "Mismatch new_values length and m_values length");

            Gpu::copy(Gpu::hostToDevice, new_values.begin(), new_values.end(), m_values.begin());
            Gpu::synchronize();
        }


        /**
        * Returns a byte vector containing all the raw data of the
        * table (not usable on GPUs)
        *
        * @return a byte vector containing table data
        */
        amrex::Vector<char> serialize() const
        {
            auto raw_data = amrex::Vector<char>{};

            utils::serialization::put_in(
                static_cast<char>(sizeof(RealType)), raw_data);
            utils::serialization::put_in(m_x_min, raw_data);
            utils::serialization::put_in(m_x_max, raw_data);
            utils::serialization::put_in(m_how_many_x, raw_data);
            utils::serialization::put_in(m_dx, raw_data);
            for (const auto& val : m_values){
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
        auto m_values = amrex::Gpu::DeviceVector<RealType>{}; /* values f(x) */

    };

    //________________ 2D equispaced table _____________________________________

    /**
    * This class implements a generic equispaced 2D lookup table
    * for a function f(x, y)
    *
    * @tparam RealType the floating point type to be used (e.g. double or float)
    */
    template <typename RealType>
    class equispaced_2d_table
    {

    public:

        /**
        * Constructor (not usable on GPUs)
        *
        * @param[in] x_min the leftmost extreme of the x coordinates
        * @param[in] x_max the rightmost extreme of the x coordinates
        * @param[in] y_min the leftmost extreme of the y coordinates
        * @param[in] y_max the rightmost extreme of the y coordinates
        * @param[in] how_many_x number of grid points along x
        * @param[in] how_many_y number of grid points along y
        * @param[in] values the values of the function that should be interpolated (row major order, passed by const ref)
        */
        equispaced_2d_table(
            const RealType x_min, const RealType x_max,
            const RealType y_min, const RealType y_max,
            const int how_many_x, const int how_many_y,
            const amrex::Gpu::DeviceVector<RealType>& values):
            m_x_min{x_min}, m_x_max{x_max},
            m_y_min{y_min}, m_y_max{y_max},
            m_how_many_x{how_many_x}, m_how_many_y{how_many_y},
            m_values{values}
        {
            m_x_size = x_max - x_min;
            m_y_size = y_max - y_min;
            m_dx = m_x_size/(m_how_many_x-1);
            m_dy = m_y_size/(m_how_many_y-1);
        }

        /**
        * Constructor (not usable on GPUs)
        *
        * @param[in] x_min the leftmost extreme of the x coordinates
        * @param[in] x_max the rightmost extreme of the x coordinates
        * @param[in] y_min the leftmost extreme of the y coordinates
        * @param[in] y_max the rightmost extreme of the y coordinates
        * @param[in] how_many_x number of grid points along x
        * @param[in] how_many_y number of grid points along y
        * @param[in] values the values of the function that should be interpolated (row major order, passed by move)
        */
        equispaced_2d_table(
            const RealType x_min, const RealType x_max,
            const RealType y_min, const RealType y_max,
            const int how_many_x, const int how_many_y,
            amrex::Gpu::DeviceVector<RealType>&& values):
            m_x_min{x_min}, m_x_max{x_max},
            m_y_min{y_min}, m_y_max{y_max},
            m_how_many_x{how_many_x}, m_how_many_y{how_many_y},
            m_values{values}
        {
            m_x_size = x_max - x_min;
            m_y_size = y_max - y_min;
            m_dx = m_x_size/(m_how_many_x-1);
            m_dy = m_y_size/(m_how_many_y-1);
        }


        /**
        * Empty constructor
        */
        constexpr
        equispaced_2d_table(){}


        /**
        * Constructor from byte array (not usable on GPUs)
        *
        * @param[in] raw_data a const reference to a byte vector
        */
        equispaced_2d_table(const amrex::Vector<char>& raw_data)
        {
            using namespace ablastr::utils;
            using namespace amrex;


            constexpr size_t min_size =
                sizeof(char)+//double or float
                sizeof(m_x_min)+sizeof(m_x_max)+
                sizeof(m_y_min)+sizeof(m_y_max)+
                sizeof(m_how_many_x)+
                sizeof(m_how_many_y)+
                sizeof(m_dx) + sizeof(m_dy);

            ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE(
                raw_data.size() >= min_size,
                "Binary data is too small to be a 2D table.");


            auto it_raw_data = raw_data.begin();

            ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE(
                serialization::get_out<char>(it_raw_data) == static_cast<char>(sizeof(RealType)),
                "Mismatch between RealType used to write and to read the 2D table");

            const auto get_out = [&it_raw_data](auto& arg){
                arg = serialization::get_out<decltype(arg)>(it_raw_data);};

            get_out(m_xmin);
            get_out(m_x_max);
            get_out(m_y_min);
            get_out(m_y_max);
            get_out(m_how_many_x);
            get_out(m_how_many_y);
            get_out(m_dx);
            get_out(m_dy);

            m_x_size = m_x_max - m_x_min;
            m_y_size = m_y_max - m_y_min;

            ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE(
                m_x_size > 0 && m_how_many_x > 0 &&
                m_y_size > 0 && m_how_many_y > 0,
                "raw_data contains invalid data.");

            const auto vals = serialization::get_n_out<RealType>(
                it_raw_data, m_how_many_x*m_how_many_y);

            m_values = Gpu::DeviceVector<RealType>(m_how_many_x);
            Gpu::copy(Gpu::hostToDevice, vals.begin(), vals.end(), m_values.begin());
            Gpu::synchronize();
        }


        /**
        * Operator ==
        *
        * @param[in] rhs a const reference to a 2D table of the same type
        * @return true if rhs and *this are equal. false otherwise
        */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        bool operator== (
            const equispaced_2d_table<RealType, VectorType> &rhs) const
        {
            return
                (m_x_min == rhs.m_x_min) &&
                (m_x_max == rhs.m_x_max) &&
                (m_y_min == rhs.m_y_min) &&
                (m_y_max == rhs.m_y_max) &&
                (m_x_size == rhs.m_x_size) &&
                (m_y_size == rhs.m_y_size) &&
                (m_how_many_x == rhs.m_how_many_x) &&
                (m_how_many_y == rhs.m_how_many_y) &&
                (m_dx == rhs.m_dx) &&
                (m_dy == rhs.m_dy) &&
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
        * Returns the j-th coodinate along y axis
        *
        * @param[in] j the index of the desired coordinate
        * @return the j-th coordinate
        */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        RealType get_y_coord(const int j) const noexcept
        {
            return j*m_dy + m_y_min;
        }


        /**
        * Returns the function value at (i, j)
        *
        * @param[in] i the index of the desired value along x
        * @param[in] j the index of the desired value along y
        * @return the value at (i,j)
        */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        RealType get_val(const int i, const int j) const noexcept
        {
            return m_values[idx(i, j)];
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
        * Returns the number of points along y
        *
        * @return the number of points along y
        */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        int get_how_many_y() const noexcept
        {
            return m_how_many_y;
        }


        /**
        * Returns the minimum y coordinate
        *
        * @return the minimum y coordinate
        */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        RealType get_y_min() const noexcept
        {
            return m_y_min;
        }


        /**
        * Returns the maximum y coordinate
        *
        * @return the maximum y coordinate
        */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        RealType get_y_max() const noexcept
        {
            return m_y_max;
        }


        /**
        * Returns the size along y
        *
        * @return the size along y
        */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        RealType get_y_size() const noexcept
        {
            return m_y_size;
        }


        /**
        * Returns the size of the steps along y
        *
        * @return the size of the steps along y
        */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        RealType get_dy() const noexcept
        {
            return m_dy;
        }


        /**
        * Returns an amrex::Vector<amrex::Array<RealType,2>> containing all the coordinates (not usable on
        * GPUs). Row major order is used.
        *
        *    @return all the coordinates
        */
        amrex::Vector<amrex::Array<RealType,2>> get_all_coordinates() const noexcept
        {
            using namespace amrex;

            auto all_coords = std::vector<std::array<RealType,2>>(
               m_how_many_x*m_how_many_y, {0_rt,0_rt});

           int count = 0;
           for (int i = 0; i < m_how_many_x; ++i){
               for (int j = 0; j < m_how_many_y; ++j){
                   all_coords[count][0] = get_x_coord(i);
                   all_coords[count][1] = get_y_coord(j);
                   count++;
               }
           }
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
        * Performs a bilinear interpolation at given position.
        *
        * @param[in] where_x the position along x
        * @param[in] where_y the position along y
        * @return the result of the interpolation
        */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        RealType interp(const RealType where_x, const RealType where_y) const noexcept
        {
            const auto idx_x_left = std::min(
                static_cast<int>(m_floor((m_how_many_x-1)*(where_x-m_x_min)/m_x_size)),
                m_how_many_x-2);
            const auto idx_x_right = idx_x_left + 1;

            const auto idx_y_left = std::min(
                static_cast<int>(m_floor((m_how_many_y-1)*(where_y-m_y_min)/m_y_size)),
                m_how_many_y-2);
            const auto idx_y_right = idx_y_left + 1;

            const auto xleft   = get_x_coord(idx_x_left);
            const auto xright  = get_x_coord(idx_x_right);
            const auto yleft   = get_y_coord(idx_y_left);
            const auto yright  = get_y_coord(idx_y_right);
            const auto f_xl_yl = m_values[idx(idx_x_left, idx_y_left)];
            const auto f_xl_yr = m_values[idx(idx_x_left, idx_y_right)];
            const auto f_xr_yl = m_values[idx(idx_x_right, idx_y_left)];
            const auto f_xr_yr = m_values[idx(idx_x_right, idx_y_right)];

            return math::bilinear_interp(
                xleft, xright, yleft, yright,
                f_xl_yl, f_xl_yr, f_xr_yl, f_xr_yr,
                where_x, where_y);
        }


        /**
        * Performs a linear interpolation at given position,
        * in the special case in which the second coordinate is exactly a grid
        * point along y
        *
        * @param[in] where_x the position along x
        * @param[in] j the index of the position along y
        * @return the result of the interpolation
        */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        RealType interp_first_coord(const RealType where_x, const int j) const noexcept
        {
            const auto idx_left = std::min(
                static_cast<int>(m_floor((m_how_many_x-1)*(where_x-m_x_min)/m_x_size)),
                m_how_many_x-2);
            const auto idx_right = idx_left + 1;

            const auto xleft     = idx_left*m_dx + m_x_min;
            const auto xright    = idx_right*m_dx + m_x_min;
            const auto left_val  = m_values[idx(idx_left,j)];
            const auto right_val = m_values[idx(idx_right,j)];

            return math::linear_interp(xleft, xright, left_val, right_val, where_x);
        }


        /**
        * Performs a linear interpolation at given position,
        * in the special case in which the first coordinate is exactly a grid
        * point along x
        *
        * @param[in] i the position along x
        * @param[in] where_y the position along y
        * @return the result of the interpolation
        */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        RealType interp_second_coord(const int i, const RealType where_y) const noexcept
        {
            const auto idx_left = std::min(
                static_cast<int>(m_floor((m_how_many_y-1)*(where_y-m_y_min)/m_y_size)),
                m_how_many_y-2);
            const auto idx_right = idx_left + 1;

            const auto yleft     = idx_left*m_dy + m_y_min;
            const auto yright    = idx_right*m_dy + m_y_min;
            const auto left_val  = m_values[idx(i, idx_left)];
            const auto right_val = m_values[idx(i, idx_right)];

            return math::linear_interp(yleft, yright, left_val, right_val, where_y);
       }


        /**
        * Overwrites all values
        *
        * @param[in] new_values
        */
        void set_all_vals(const std::vector<RealType>& new_values)
        {
            ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE(
                new_values.size() != m_values.size(),
                "Mismatch new_values length and m_values length");

            Gpu::copy(Gpu::hostToDevice, new_values.begin(), new_values.end(), m_values.begin());
            Gpu::synchronize();
        }


        /**
        * Returns a byte vector containing all the raw data of the
        * table (not usable on GPUs)
        *
        * @return a byte vector containing table data
        */
        amrex::Vector<char> serialize() const
        {
            auto raw_data = std::vector<char>{};

            utils::serialization::put_in(
                static_cast<char>(sizeof(RealType)), raw_data);
            utils::serialization::put_in(m_x_min, raw_data);
            utils::serialization::put_in(m_x_max, raw_data);
            utils::serialization::put_in(m_y_min, raw_data);
            utils::serialization::put_in(m_y_max, raw_data);
            utils::serialization::put_in(m_how_many_x, raw_data);
            utils::serialization::put_in(m_how_many_y, raw_data);
            utils::serialization::put_in(m_dx, raw_data);
            utils::serialization::put_in(m_dy, raw_data);
            for (const auto& val : m_values){
                utils::serialization::put_in(val, raw_data);
            }

            return raw_data;
        }

    private:

        auto m_x_min  = static_cast<RealType>(0.0);  /* minimum x coordinate */
        auto m_x_max  = static_cast<RealType>(0.0);  /* maximum x coordinate */
        auto m_y_min  = static_cast<RealType>(0.0);  /* minimum y coordinate */
        auto m_y_max  = static_cast<RealType>(0.0);  /* maximum y coordinate */
        auto m_x_size = static_cast<RealType>(0.0); /* size along x */
        auto m_y_size = static_cast<RealType>(0.0); /* size along y */
        int  m_how_many_x = 0; /* how many grid points along x */
        int  m_how_many_y = 0; /* how many grid points along y */
        auto m_dx = static_cast<RealType>(0.0); /* step size along x */
        auto m_dy = static_cast<RealType>(0.0); /* step size along y */
        auto m_values = amrex::Gpu::DeviceVector<RealType>{}; /* values f(x) */

        /**
        * Function values are stored internally in a 1D vector.
        * This function calculates the index along this vector
        * from the coordinate indices i,j. Row major order is used.
        *
        * @param[in] i index along x
        * @param[in] j index along y
        * @return index along internal 1D vector
        */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        constexpr int idx(const int i, const int j) const noexcept
        {
            return i*m_how_many_y + j;
        }

   };

}

#endif //ABLASTR_QED_LOOKUP_TABLES_
