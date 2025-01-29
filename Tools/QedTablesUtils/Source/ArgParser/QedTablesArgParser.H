#ifndef QED_TABLES_ARG_PARSER_
#define QED_TABLES_ARG_PARSER_

#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace ArgParser
{
    /* Possible types of parameters of command line arguments (no argument, int, double, or std::string) */
    enum class ArgType { NoArg, Integer, Double, String };

    /* This type represents the type of a command line argument: if it exists it is either an int, a double, or a std::string */
    using ArgVal = std::optional<std::variant<int, double, std::string>>;

    /* The type of a possible command line argument */
    using Key = std::tuple<std::string, ArgType, std::string>;

    /* The type of the map of the parsed command line arguments */
    using ParsedArgs = std::unordered_map<std::string, ArgVal>;

    /**
    * \brief Gets the value out of an ArgVal (std::optional<std::variant<int, double, std::string>>) object
    *
    * \tparam T          the type to return (must be int, double, or std::string)
    * \param[in] arg_val the ArgVal object
    * \return            the value in arg_val
    */
    template <typename T>
    T GetVal(const ArgVal& arg_val)
    {
        return std::get<T>(*arg_val);
    }

    /**
    * \brief Function to parse the command line arguments
    *
    * \param[in] keys    the list of possible command line arguments
    * \param[in] argc    the number of command line arguments
    * \param[in] argv    all the command line arguments
    * \return            the parsed command line arguments
    */
    ParsedArgs ParseArgs (const std::vector<Key>& keys, const int argc, char const* const* argv);

    /**
    * \brief Prints the command line options
    *
    * \param[in] cmd_list    the list of possible command line arguments
    */
    void PrintHelp (const std::vector<Key>& cmd_list);
};


#endif //QED_TABLES_ARG_PARSER_
