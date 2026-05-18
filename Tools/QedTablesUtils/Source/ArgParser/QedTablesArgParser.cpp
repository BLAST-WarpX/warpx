#include "QedTablesArgParser.H"

#include <iostream>

using namespace ArgParser;
using namespace std;

namespace
{
void
WarnMsg(const string& msg)
{
    cout << "!!! Parse Warning: " + msg + "\n";
}

void
ErrMsg(const string& msg)
{
    cout << "### Parse Error: " + msg + "\n";
    exit(1);
}

vector<string>
ArgvToStrVec(const int argc, char const* const* argv )
{
    auto sargs = vector<string>{};
    for (int i = 0; i < argc; ++i)
        sargs.push_back(std::string{argv[i]});
    return sargs;
}

optional<pair<string,ArgType>>
GetArgType(const std::vector<Key>& keys, vector<string>::iterator& it)
{
    const auto str = *it;
    it++;
    for (const auto& kk : keys){
        if (get<0>(kk) == str)
        {
            return std::make_pair(get<0>(kk), get<1>(kk));
        }
    }
    return nullopt;
}

ArgVal
ReadArg(const ArgType arg_type, vector<string>::iterator& it)
{
    if(arg_type == ArgType::NoArg) return nullopt;
    if(arg_type == ArgType::String) return *(it++);
    if(arg_type == ArgType::Integer) return stoi(*(it++));
    if(arg_type == ArgType::Double) return stod(*(it++));

    ErrMsg("Failed to parse type!"s);

    return nullopt;
}

}

ParsedArgs
ArgParser::ParseArgs (const std::vector<Key>& keys, const int argc, char const* const* argv)
{
    auto parsed_args = ParsedArgs{};

    auto sargs = ArgvToStrVec(argc, argv);
    auto it = sargs.begin() + 1; // the first argument is always the executable name
    while (it < sargs.end()){
        const auto tt = *it;
        auto res = GetArgType(keys, it);
        if (res != nullopt){
            const auto& [key, arg_type] = *res;
            const auto arg = ReadArg(arg_type, it);
            if (parsed_args.find(key) != parsed_args.end())
                WarnMsg("Rewriting '"+ key +"' argument !");
            parsed_args[key] = arg;
        }else{
            WarnMsg("Can't parse '"+ tt +"' !");
        }
    }

    return parsed_args;
}

void
ArgParser::PrintHelp (const vector<ArgParser::Key>& cmd_list)
{
    cout << "Command line options:\n";

    for (const auto& el : cmd_list){
        const auto type = get<1>(el);
        string stype = "[??????]";
        if (type == ArgType::NoArg)
            stype = "[NO ARG]";
        else if (type == ArgType::String)
            stype = "[STRING]";
        else if (type == ArgType::Double)
            stype = "[DOUBLE]";
        else if (type == ArgType::Integer)
            stype = "[INTEGR]";

        cout << get<0>(el)
            << " " << stype
            << " " << get<2>(el)
            << "\n";
    }

}
