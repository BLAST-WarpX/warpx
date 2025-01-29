/* Copyright 2021 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef ABLASTR_MSG_LOGGER_SERIALIZATION_H_
#define ABLASTR_MSG_LOGGER_SERIALIZATION_H_

#include <algorithm>
#include <array>
#include <cstring>
#include <iterator>
#include <string>
#include <type_traits>
#include <vector>

namespace ablastr::utils::serialization
{
    /**
    * This function transforms a variable of type T into a vector of chars holding its
    * byte representation and it appends this vector at the end of an
    * existing vector of chars. T must be either a trivially copyable type or an std::string
    *
    * @tparam T the variable type
    * @param[in] val a variable of type T to be serialized
    * @param[in, out] vec a reference to the vector to which the byte representation of val is appended
    */
    template <typename T>
    void put_in(const T &val, std::vector<char> &vec)
    {
        if constexpr (std::is_same<T, std::string>())
        {
            const char *c_str = val.c_str();
            const auto length = static_cast<int>(val.size());

            put_in(length, vec);
            std::copy(c_str, c_str + length, std::back_inserter(vec));
        }
        else
        {
            static_assert(std::is_trivially_copyable<T>(),
                "Cannot serialize non-trivally copyable types, except std::string.");

            const auto *ptr_val = reinterpret_cast<const char *>(&val);
            std::copy(ptr_val, ptr_val + sizeof(T), std::back_inserter(vec));
        }
    }

    /**
    * This function transforms an std::vector<T> into a vector of chars holding its
    * byte representation and it appends this vector at the end of an
    * existing vector of chars. T must be either a trivially copyable type or an std::string.
    * A specialization exists in case val is a vector of chars.
    *
    * @tparam T the variable type
    * @param[in] val a variable of type T to be serialized
    * @param[in, out] vec a reference to the vector to which the byte representation of val is appended
    */
    template <typename T>
    void put_in_vec(const std::vector<T> &val, std::vector<char> &vec)
    {
        if constexpr (std::is_same<T, char>())
        {
            put_in(static_cast<int>(val.size()), vec);
            vec.insert(vec.end(), val.begin(), val.end());
        }
        else
        {
            static_assert(std::is_trivially_copyable<T>() || std::is_same<T, std::string>(),
                "Cannot serialize vectors of non-trivally copyable types"
                ", except vectors of std::string.");

            put_in(static_cast<int>(val.size()), vec);
            for (const auto &el : val) {
                put_in(el, vec);
            }
        }
    }

    /**
    * This function extracts a variable of type T from a byte vector, at the position
    * given by a std::vector<char> iterator. The iterator is then advanced according to
    * the number of bytes read from the byte vector. T must be either a trivially copyable type
    * or an std::string.
    *
    * @tparam T the variable type (must be trivially copyable)
    * @param[in, out] it the iterator to a byte vector
    * @return the variable extracted from the byte array
    */
    template <typename T>
    T get_out(std::vector<char>::const_iterator &it)
    {
        if constexpr (std::is_same<T, std::string>())
        {
            const auto length = get_out<int>(it);
            auto str = std::string{it, it + length};
            it += length;

            return str;
        }
        else
        {
            static_assert(std::is_trivially_copyable<T>(),
                "Cannot extract non-trivally copyable types from char vectors,"
                " with the exception of std::string.");

            auto temp = std::array<char, sizeof(T)>{};
            std::copy(it, it + sizeof(T), temp.begin());
            it += sizeof(T);
            T res;
            std::memcpy(&res, temp.data(), sizeof(T));

            return res;
        }
    }

    /**
    * This function extracts an std::vector<T> from a byte vector, at the position
    * given by a std::vector<char> iterator. The iterator is then advanced according to
    * the number of bytes read from the byte vector. T must be either a trivially copyable type
    * or an std::string.
    *
    * @tparam T the variable type (must be trivially copyable)
    * @param[in, out] it the iterator to a byte vector
    * @return the variable extracted from the byte array
    */
    template <typename T>
    std::vector<T> get_out_vec(std::vector<char>::const_iterator &it)
    {
        if constexpr (std::is_same<T, std::string>())
        {
            const auto length = get_out<int>(it);
            std::vector<char> res(length);
            std::copy(it, it + length, res.begin());
            it += length;

            return res;
        }
        else
        {
            static_assert(std::is_trivially_copyable<T>() || std::is_same<T, std::string>(),
                "Cannot extract non-trivally copyable types from char vectors,"
                " with the exception of std::string.");

            const auto length = get_out<int>(it);
            std::vector<T> res(length);
            for (int i = 0; i < length; ++i) {
                res[i] = get_out<T>(it);
            }

            return res;
        }
    }
}

#endif //ABLASTR_MSG_LOGGER_SERIALIZATION_H_
