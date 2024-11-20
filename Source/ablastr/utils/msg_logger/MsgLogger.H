/* Copyright 2021 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef ABLASTR_MSG_LOGGER_H_
#define ABLASTR_MSG_LOGGER_H_

#include <AMReX_Config.H>

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ablastr::utils::msg_logger
{
    /** Priority is recorded together with messages. It influences
    * the display order and the appearance of a message.
    */
    enum class Priority
    {
        /** Low priority message */
        low,
        /** Medium priority message */
        medium,
        /** High priority message */
        high
    };

    /**
    * \brief This function converts a Priority into the corresponding
    * string (e.g, Priority::low --> "low")
    *
    * @param[in] priority the priority
    * @return the corresponding string
    */
    std::string PriorityToString(const Priority& priority);

    /**
    * \brief This function converts a string into the corresponding
    * priority (e.g, "low" --> Priority::low)
    *
    * @param[in] priority_string the priority string
    * @return the corresponding priority
    */
    Priority StringToPriority(const std::string& priority_string);

    /**
    * This struct represents a message, which is composed by
    * a topic, a text and a priority. It also provides methods for
    * serialization and deserialization.
    */
    struct Msg
    {
        std::string topic  /*! The message topic*/;
        std::string text   /*! The message text*/;
        Priority priority  /*! The priority of the message*/;

        /**
        * \brief This function returns a byte representation of the struct
        *
        * @return a byte vector
        */
        [[nodiscard]] std::vector<char> serialize() const;

        /**
        * \brief This function generates a Msg struct from a byte vector
        *
        * @param[in] it iterator of a byte array
        * @return a Msg struct
        */
        static Msg deserialize(std::vector<char>::const_iterator& it);

        /**
        * \brief Same as static Msg deserialize(std::vector<char>::const_iterator& it)
        * but accepting an rvalue as an argument
        *
        * @param[in] it iterator of a byte array
        * @return a Msg struct
        */
        static Msg deserialize(std::vector<char>::const_iterator&& it);
    };

    /**
    * This struct represents a message with counter, which is composed
    * by a message and a counter. The latter is intended to store the
    * number of times a message is recorded. The struct also provides
    * methods for serialization and deserialization.
    */
    struct MsgWithCounter
    {
        Msg msg      /*! A message*/;
        std::int64_t counter  /*! The counter*/;

        /**
        * \brief This function returns a byte representation of the struct
        *
        * @return a byte vector
        */
        [[nodiscard]] std::vector<char> serialize() const;

        /**
        * \brief This function generates a MsgWithCounter struct from a byte vector
        *
        * @param[in] it iterator of a byte array
        * @return a MsgWithCounter struct
        */
        static MsgWithCounter deserialize(std::vector<char>::const_iterator& it);

        /**
        * \brief Same as static Msg MsgWithCounter(std::vector<char>::const_iterator& it)
        * but accepting an rvalue as an argument
        *
        * @param[in] it iterator of a byte array
        * @return a MsgWithCounter struct
        */
        static MsgWithCounter deserialize(std::vector<char>::const_iterator&& it);
    };

    /**
    * This struct represents a message with counter and ranks, which is
    * composed by a message with counter, a bool flag and a std::vector<int>.
    * The bool flag is used to store if a message is emitted by all the ranks.
    * The std::vector<int> is used to store the affected ranks
    * (note: when we switch to C++17, should we consider variants?).
    * The struct also provides methods for serialization and deserialization.
    */
    struct MsgWithCounterAndRanks
    {
        MsgWithCounter msg_with_counter /*! A message with counter*/;
        bool all_ranks                  /*! Flag to store if message is emitted by all ranks*/;
        std::vector<int> ranks          /*! Affected ranks*/;

        /**
        * \brief This function returns a byte representation of the struct
        *
        * @return a byte vector
        */
        [[nodiscard]] std::vector<char> serialize() const;

        /**
        * \brief This function generates a MsgWithCounterAndRanks struct from a byte vector
        *
        * @param[in] it iterator of a byte array
        * @return a MsgWithCounterAndRanks struct
        */
        static MsgWithCounterAndRanks deserialize(std::vector<char>::const_iterator& it);

        /**
        * \brief Same as static Msg MsgWithCounterAndRanks(std::vector<char>::const_iterator& it)
        * but accepting an rvalue as an argument
        *
        * @param[in] it iterator of a byte array
        * @return a MsgWithCounterAndRanks struct
        */
        static MsgWithCounterAndRanks deserialize(std::vector<char>::const_iterator&& it);
    };

    /**
    * \brief This implements the < operator for Msg.
    * Warning messages are first ordered by priority (warning: high < medium < low
    * to give precedence to higher priorities), then by topic (alphabetically),
    * and finally by text (alphabetically).
    *
    * @param[in] l a Msg
    * @param[in] r a Msg
    * @return true if l<r, false otherwise
    */
    constexpr bool operator<(const Msg& l, const Msg& r)
    {
        return
            (l.priority > r.priority) ||
            ((l.priority == r.priority) && (l.topic < r.topic)) ||
            ((l.priority == r.priority) && (l.topic == r.topic) && (l.text < r.text));
    }

    /**
    * This class is responsible for storing messages and merging messages
    * collected by different processes.
    */
    class Logger
    {
        public:

        /**
        * \brief The constructor.
        */
        Logger();

        /**
        * \brief This function records a message
        *
        * @param[in] msg a Msg struct
        */
        void record_msg(const Msg& msg);

        /**
        * \brief This function returns a vector containing the recorded messages
        *
        * @return a vector of the recorded messages
        */
        [[nodiscard]] std::vector<Msg> get_msgs() const;

        /**
        * \brief This function returns a vector containing the recorded messages
        * with the corresponding counters
        *
        * @return a vector of the recorded messages with counters
        */
        [[nodiscard]] std::vector<MsgWithCounter>
        get_msgs_with_counter() const;

        /**
        * \brief This collective function generates a vector containing the messages
        * with counters and emitting ranks by gathering data from
        * all the ranks
        *
        * @return a vector of messages with counters and ranks if I/O rank, an empty vector otherwise
        */
        [[nodiscard]] std::vector<MsgWithCounterAndRanks>
        collective_gather_msgs_with_counter_and_ranks() const;

        private:

        /**
        * \brief This function implements the trivial special case of
        * collective_gather_msgs_with_counter_and_ranks when there is only one rank.
        *
        * @return a vector of messages with counters and ranks
        */
        [[nodiscard]] std::vector<MsgWithCounterAndRanks>
        one_rank_gather_msgs_with_counter_and_ranks() const;

#ifdef AMREX_USE_MPI
        /**
        * \brief This collective function finds the rank having the
        * most messages and how many messages this rank has. The
        * rank having the most messages is designated as "gather rank".
        *
        * @param[in] how_many_msgs the number of messages that the current rank has
        * @return a pair containing the ID of the "gather rank" and its number of messages
        */
        [[nodiscard]] std::pair<int, int>
        find_gather_rank_and_its_msgs(int how_many_msgs) const;

        /**
        * \brief This function uses data gathered on the "gather rank" to generate
        * a vector of messages with global counters and emitting rank lists
        *
        * @param[in] my_msg_map messages and counters of the current rank (as a map)
        * @param[in] all_data a byte array containing all the data gathered on the gather rank
        * @param[in] displacements a vector of displacements to access data corresponding to a given rank in all_data
        * @param[in] gather_rank the ID of the "gather rank"
        * @return if gather_rank==m_rank a vector of messages with global counters and emitting rank lists, dummy data otherwise
        */
        [[nodiscard]] std::vector<MsgWithCounterAndRanks>
        compute_msgs_with_counter_and_ranks(
            const std::map<Msg,std::int64_t>& my_msg_map,
            const std::vector<char>& all_data,
            const std::vector<int>& displacements,
            int gather_rank
        ) const;


        /**
        * \brief If the gather_rank is not the I/O rank, this function sends msgs_with_counter_and_ranks
        * to the I/O rank. This function uses point-to-point communications.
        *
        * @param[in] msgs_with_counter_and_ranks a vector of messages with counters and ranks
        * @param[in] gather_rank the ID of the "gather rank"
        */
        void
        swap_with_io_rank(
        std::vector<MsgWithCounterAndRanks>& msgs_with_counter_and_ranks,
        int gather_rank) const;

#endif

        const int m_rank       /*! MPI rank of the current process*/;
        const int m_num_procs  /*! Number of MPI ranks*/;
        const int m_io_rank    /*! Rank of the I/O process*/;

        std::map<Msg, std::int64_t> m_messages /*! This stores a map to associate warning messages with the corresponding counters*/;
    };
}

#endif //ABLASTR_MSG_LOGGER_H_
