#include "QedTableCommons.H"
#include "ArgParser/QedTablesArgParser.H"

#include <picsar_qed/physics/breit_wheeler/breit_wheeler_engine_tables.hpp>
#include <picsar_qed/physics/breit_wheeler/breit_wheeler_engine_tables_generator.hpp>
#include <picsar_qed/physics/quantum_sync/quantum_sync_engine_tables.hpp>
#include <picsar_qed/physics/quantum_sync/quantum_sync_engine_tables_generator.hpp>
#include <picsar_qed/utils/serialization.hpp>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace ArgParser;
using namespace std;

namespace pxr_sr = picsar::multi_physics::utils::serialization;
namespace pxr_bw = picsar::multi_physics::phys::breit_wheeler;
namespace pxr_qs = picsar::multi_physics::phys::quantum_sync;

const auto line_commands = vector<ArgParser::Key>{
    {"-h"                  , ArgType::NoArg  , "Prints all command line arguments"},
    {"--table"             , ArgType::String , "Either BW (Breit-Wheeler) or QS (Quantum Synchrotron)"},
    {"--mode"              , ArgType::String,  "Precision of the calculations: either DP (double) or SP (single)"},
    {"--dndt_chi_min"      , ArgType::Double , "minimum chi parameter for the dNdt table"},
    {"--dndt_chi_max"      , ArgType::Double , "maximum chi parameter for the dNdt table"},
    {"--dndt_how_many"     , ArgType::Integer, "number of points in the dNdt table"},
    {"--pair_chi_min"      , ArgType::Double , "minimum chi for the pair production table (BW only)"},
    {"--pair_chi_max"      , ArgType::Double , "maximum chi for the pair production table (BW only)"},
    {"--pair_chi_how_many" , ArgType::Integer, "number of chi points in the pair production table (BW only)"},
    {"--pair_frac_how_many", ArgType::Integer, "number of frac points in the pair production table (BW only)"},
    {"--em_chi_min"        , ArgType::Double , "minimum chi for the photon emission table (QS only)"},
    {"--em_chi_max"        , ArgType::Double , "maximum chi for the photon emission production table (QS only)"},
    {"--em_frac_min"       , ArgType::Double , "minimum frac for the photon emission production table (QS only)"},
    {"--em_chi_how_many"   , ArgType::Integer, "number of chi points in the photon emission table (QS only)"},
    {"--em_frac_how_many"  , ArgType::Integer, "number of frac points in the photon emission table (QS only)"},
    {"-o"                  , ArgType::String , "filename to save the lookup table"}
};

void GenerateTable (const ParsedArgs& args);
template <typename RealType>
void GenerateTableBW (const ParsedArgs& args, const string& outfile_name);
template <typename RealType>
void GenerateTableQS (const ParsedArgs& args, const string& outfile_name);

int main (int argc, char** argv)
{
    cout << "### QED Table Generator ###\n";
    const auto args_map = ParseArgs(line_commands, argc, argv);

    if (args_map.empty() || Contains(args_map, "-h")){
        PrintHelp(line_commands);
    }
    else{
        GenerateTable(args_map);
    }

    SuccessExit();
}


void GenerateTable (const ParsedArgs& args)
{
    if (!Contains(args, "--table"))
        AbortWithMessage("'--table' argument must be provided");

    if (!Contains(args, "--mode"))
        AbortWithMessage("'--mode' argument must be provided");

    if (!Contains(args, "-o"))
        AbortWithMessage("'-o' argument must be provided");

    const auto which_table = GetVal<string>(args.at("--table"));
    const auto mode = GetVal<string>(args.at("--mode"));
    const auto outfile_name = GetVal<string>(args.at("-o"));

    bool use_double = false;

    if (mode == "DP"s)
        use_double = true;
    else if (mode == "SP"s)
        use_double = false;
    else
        AbortWithMessage("'--mode' must be eiter 'DP' or 'SP'");

    if (which_table == "BW"s){
        if (use_double)
            GenerateTableBW<double>(args, outfile_name);
        else
            GenerateTableBW<float>(args, outfile_name);
    }
    else if (which_table == "QS"s)
        if (use_double)
            GenerateTableQS<double>(args, outfile_name);
        else
            GenerateTableQS<float>(args, outfile_name);
    else
        AbortWithMessage("'--table' must be eiter 'QS' or 'BW'");
}

template<typename RealType>
void GenerateTableBW (const ParsedArgs& args, const string& outfile_name)
{
    cout << "    Generating BW table " <<
        (is_same_v<RealType, double> ? "(double "s : "(single "s) << " precision)\n"s;

    if (!Contains(args, "--dndt_chi_min")      || !Contains(args, "--dndt_chi_max") ||
        !Contains(args, "--dndt_how_many")     || !Contains(args, "--pair_chi_min") ||
        !Contains(args, "--pair_chi_max")      || !Contains(args, "--pair_chi_how_many") ||
        !Contains(args, "--pair_frac_how_many"))
            AbortWithMessage("All the BW table arguments must be provided (check with -h)");

    const auto dndt_table_params =
        pxr_bw::dndt_lookup_table_params<RealType>{
            static_cast<RealType>(GetVal<double>(args.at("--dndt_chi_min"))),
            static_cast<RealType>(GetVal<double>(args.at("--dndt_chi_max"))),
            GetVal<int>(args.at("--dndt_how_many"))
        };

    const auto pair_prod_table_params =
        pxr_bw::pair_prod_lookup_table_params<RealType>{
            static_cast<RealType>(GetVal<double>(args.at("--pair_chi_min"))),
            static_cast<RealType>(GetVal<double>(args.at("--pair_chi_max"))),
            GetVal<int>(args.at("--pair_chi_how_many")),
            GetVal<int>(args.at("--pair_frac_how_many"))
        };

    std::cout << "    Params: \n";
    std::cout << "    - dndt_chi_min        : " <<  dndt_table_params.chi_phot_min << "\n";
    std::cout << "    - dndt_chi_max        : " <<  dndt_table_params.chi_phot_max << "\n";
    std::cout << "    - dndt_how_many       : " <<  dndt_table_params.chi_phot_how_many << "\n";
    std::cout << "    - pair_chi_min        : " <<  pair_prod_table_params.chi_phot_min << "\n";
    std::cout << "    - pair_chi_max        : " <<  pair_prod_table_params.chi_phot_max << "\n";
    std::cout << "    - pair_chi_how_many   : " <<  pair_prod_table_params.chi_phot_how_many << "\n";
    std::cout << "    - pair_frac_how_many  : " <<  pair_prod_table_params.frac_how_many  << "\n";
    std::cout << "    ----------------------- " << "\n\n";

    auto dndt_table =
        pxr_bw::dndt_lookup_table<RealType, vector<RealType>>{
            dndt_table_params};

    auto pair_prod_table =
        pxr_bw::pair_prod_lookup_table<RealType, vector<RealType>>{
            pair_prod_table_params};

    dndt_table.generate(true); //Progress bar is displayed
    pair_prod_table.generate(true); //Progress bar is displayed

    const auto data_dndt = dndt_table.serialize();
    const auto data_pair_prod = pair_prod_table.serialize();

    const uint64_t size_first = data_dndt.size();

    vector<char> res{};
    pxr_sr::put_in(size_first, res);
    for (const auto& tmp : data_dndt)
        pxr_sr::put_in(tmp, res);
    for (const auto& tmp : data_pair_prod)
        pxr_sr::put_in(tmp, res);

    auto of = std::ofstream{outfile_name, std::ios_base::binary};
    of.write(res.data(), res.size());
    of.close();

    cout << "    Done! \n";
}

template<typename RealType>
void GenerateTableQS (const ParsedArgs& args, const string& outfile_name)
{
    cout << "    Generating QS table " <<
        (is_same_v<RealType, double> ? "(double "s : "(single "s) << " precision)\n"s;

    if (!Contains(args, "--dndt_chi_min")      || !Contains(args, "--dndt_chi_max") ||
        !Contains(args, "--dndt_how_many")     || !Contains(args, "--em_chi_min")   ||
        !Contains(args, "--em_chi_max")        || !Contains(args, "--em_frac_min")  ||
        !Contains(args, "--em_chi_how_many")   || !Contains(args, "--em_frac_how_many"))
            AbortWithMessage("All the QS table arguments must be provided (check with -h)");

    const auto dndt_table_params =
        pxr_qs::dndt_lookup_table_params<RealType>{
            static_cast<RealType>(GetVal<double>(args.at("--dndt_chi_min"))),
            static_cast<RealType>(GetVal<double>(args.at("--dndt_chi_max"))),
            GetVal<int>(args.at("--dndt_how_many"))
        };

    const auto phot_em_table_params =
        pxr_qs::photon_emission_lookup_table_params<RealType>{
            static_cast<RealType>(GetVal<double>(args.at("--em_chi_min"))),
            static_cast<RealType>(GetVal<double>(args.at("--em_chi_max"))),
            static_cast<RealType>(GetVal<double>(args.at("--em_frac_min"))),
            GetVal<int>(args.at("--em_chi_how_many")),
            GetVal<int>(args.at("--em_frac_how_many"))
        };

    std::cout << "    Params: \n";
    std::cout << "    - dndt_chi_min           : " <<  dndt_table_params.chi_part_min << "\n";
    std::cout << "    - dndt_chi_max           : " <<  dndt_table_params.chi_part_max << "\n";
    std::cout << "    - dndt_how_many          : " <<  dndt_table_params.chi_part_how_many << "\n";
    std::cout << "    - phot_em_chi_min        : " <<  phot_em_table_params.chi_part_min << "\n";
    std::cout << "    - phot_em_chi_max        : " <<  phot_em_table_params.chi_part_max << "\n";
    std::cout << "    - phot_em_frac_min       : " <<  phot_em_table_params.frac_min << "\n";
    std::cout << "    - phot_em_chi_how_many   : " <<  phot_em_table_params.chi_part_how_many << "\n";
    std::cout << "    - phot_em_frac_how_many  : " <<  phot_em_table_params.frac_how_many  << "\n";
    std::cout << "    ----------------------- " << "\n\n";

    auto dndt_table =
        pxr_qs::dndt_lookup_table<RealType, vector<RealType>>{
            dndt_table_params};

    auto phot_em_table =
        pxr_qs::photon_emission_lookup_table<RealType, vector<RealType>>{
            phot_em_table_params};

    dndt_table.generate(true); //Progress bar is displayed
    phot_em_table.generate(true); //Progress bar is displayed

    const auto data_dndt = dndt_table.serialize();
    const auto data_phot_em = phot_em_table.serialize();

    const uint64_t size_first = data_dndt.size();

    vector<char> res{};
    pxr_sr::put_in(size_first, res);
    for (const auto& tmp : data_dndt)
        pxr_sr::put_in(tmp, res);
    for (const auto& tmp : data_phot_em)
        pxr_sr::put_in(tmp, res);

    auto of = std::ofstream{outfile_name, std::ios_base::binary};
    of.write(res.data(), res.size());
    of.close();

    cout << "    Done! \n";
}
