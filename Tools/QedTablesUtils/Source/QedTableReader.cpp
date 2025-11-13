#include "QedTableCommons.H"
#include "ArgParser/QedTablesArgParser.H"

#include <picsar_qed/physics/breit_wheeler/breit_wheeler_engine_tables.hpp>
#include <picsar_qed/physics/quantum_sync/quantum_sync_engine_tables.hpp>
#include <picsar_qed/utils/serialization.hpp>

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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
    {"-i"                  , ArgType::String , "Name of the file to open"},
    {"--table"             , ArgType::String , "Either BW (Breit-Wheeler) or QS (Quantum Synchrotron)"},
    {"--mode"              , ArgType::String , "Precision of the calculations: either DP (double) or SP (single)"},
    {"-o"                  , ArgType::String , "filename to save the lookup table in human-readable format"}
};

void ReadTable (const ParsedArgs& args);
template <typename RealType>
void ReadTableBW (const string& input_file, const string& outfile_name);
template <typename RealType>
void ReadTableQS (
    const string& input_file, const string& outfile_name);

/*Wrapper class to access protected data*/
template <typename RealType>
class bw_pair_production_table_wrapper :
    public pxr_bw::pair_prod_lookup_table<RealType, std::vector<RealType>>
{
    public:
    void write_table_data(std::ofstream& of) const;
};

/*Wrapper class to access protected data*/
template <typename RealType>
class qs_photon_emission_table_wrapper :
    public pxr_qs::photon_emission_lookup_table<RealType, std::vector<RealType>>
{
    public:
    void write_table_data(std::ofstream& of) const;
};

int main (int argc, char** argv)
{
    cout << "### QED Table Reader ###\n";
    const auto args_map = ParseArgs(line_commands, argc, argv);

    if (args_map.empty() || Contains(args_map, "-h")){
        PrintHelp(line_commands);
    }
    else{
        ReadTable(args_map);
    }

    SuccessExit();
}

void ReadTable (const ParsedArgs& args)
{
    if (!Contains(args, "--table"))
        AbortWithMessage("'--table' argument must be provided");

    if (!Contains(args, "--mode"))
        AbortWithMessage("'--mode' argument must be provided");

    if (!Contains(args, "-o"))
        AbortWithMessage("'-o' argument must be provided");

    if (!Contains(args, "-i"))
        AbortWithMessage("'-i' argument must be provided");

    const auto which_table = GetVal<string>(args.at("--table"));
    const auto mode = GetVal<string>(args.at("--mode"));
    const auto input_file = GetVal<string>(args.at("-i"));
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
            ReadTableBW<double>(input_file, outfile_name);
        else
            ReadTableBW<float>(input_file, outfile_name);
    }
    else if (which_table == "QS"s)
        if (use_double)
            ReadTableQS<double>(input_file, outfile_name);
        else
            ReadTableQS<float>(input_file, outfile_name);
    else
        AbortWithMessage("'--table' must be eiter 'QS' or 'BW'");
}

template<typename RealType>
void ReadTableBW (const string& input_file, const string& outfile_name)
{
    cout << "    Reading BW table " <<
        (is_same_v<RealType, double> ? "(double "s : "(single "s) << " precision)\n"s;

    auto ifs = ifstream(input_file, std::ios::binary);
    auto raw_data = vector<char>(istreambuf_iterator<char>(ifs), {});
    ifs.close();

    auto raw_iter = raw_data.begin();
    const auto size_first = pxr_sr::get_out<uint64_t>(raw_iter);
    if(size_first <= 0 || size_first >= raw_data.size() )
        AbortWithMessage("Something is wrong with " + input_file);

    const auto raw_dndt_table = vector<char>{
        raw_iter, raw_iter+size_first};

    const auto raw_pair_prod_table = vector<char>{
        raw_iter+size_first, raw_data.end()};

    const auto dndt_table =
        pxr_bw::dndt_lookup_table<RealType, vector<RealType>>{raw_dndt_table};
    const auto pair_prod_table =
        pxr_bw::pair_prod_lookup_table<RealType, vector<RealType>>{raw_pair_prod_table};

    if (!dndt_table.is_init() || !pair_prod_table.is_init())
        AbortWithMessage("Something went wrong with lookup table initialization");

    auto of_dndt = ofstream{outfile_name + "_dndt"};
    of_dndt << std::setprecision(std::numeric_limits<RealType>::digits10 + 1);
    const auto coord_dndt = dndt_table.get_all_coordinates();
    for (const auto& cc : coord_dndt )
        of_dndt << cc << " " << dndt_table.interp(cc) << "\n";
    of_dndt.close();

    auto of_pair = ofstream{outfile_name + "_pair"};
    of_pair << std::setprecision(std::numeric_limits<RealType>::digits10 + 1);
    const auto wrapper = bw_pair_production_table_wrapper<RealType>{pair_prod_table};
    wrapper.write_table_data(of_pair);
    of_pair.close();
}

template<typename RealType>
void ReadTableQS (
    const string& input_file, const string& outfile_name)
{
    cout << "    Reading QS table " <<
        (is_same_v<RealType, double> ? "(double "s : "(single "s) << " precision)\n"s;

    auto ifs = ifstream(input_file, std::ios::binary);
    auto raw_data = vector<char>(istreambuf_iterator<char>(ifs), {});
    ifs.close();

    auto raw_iter = raw_data.begin();
    const auto size_first = pxr_sr::get_out<uint64_t>(raw_iter);
    if(size_first <= 0 || size_first >= raw_data.size() )
        AbortWithMessage("Something is wrong with " + input_file);

    const auto raw_dndt_table = vector<char>{
        raw_iter, raw_iter+size_first};

    const auto raw_phot_em_table = vector<char>{
        raw_iter+size_first, raw_data.end()};

    const auto dndt_table =
        pxr_qs::dndt_lookup_table<RealType, vector<RealType>>{raw_dndt_table};
    const auto phot_em_table =
        pxr_qs::photon_emission_lookup_table<RealType, vector<RealType>>{raw_phot_em_table};

    if (!dndt_table.is_init() || !phot_em_table.is_init())
        AbortWithMessage("Something went wrong with lookup table initialization");

    auto of_dndt = ofstream{outfile_name + "_dndt"};
    of_dndt << std::setprecision(std::numeric_limits<RealType>::digits10 + 1);
    const auto coord_dndt = dndt_table.get_all_coordinates();
    for (const auto& cc : coord_dndt )
        of_dndt << cc << " " << dndt_table.interp(cc) << "\n";
    of_dndt.close();

    auto of_phot_em = ofstream{outfile_name + "_phot_em"};
    of_phot_em << std::setprecision(std::numeric_limits<RealType>::digits10 + 1);

    const auto wrapper = qs_photon_emission_table_wrapper<RealType>{phot_em_table};
    wrapper.write_table_data(of_phot_em);
    of_phot_em.close();

    return;
}

template <typename RealType>
void
bw_pair_production_table_wrapper<RealType>::write_table_data(
    std::ofstream& of) const
{
    const auto how_many_x = this->m_table.get_how_many_x();
    const auto how_many_y = this->m_table.get_how_many_y();
    for (int i = 0; i < how_many_x; ++i){
        for (int j = 0; j < how_many_y; ++j){
            const auto xcoord = this->m_table.get_x_coord(i);
            const auto ycoord = this->m_table.get_y_coord(j);
            const auto val = this->m_table.get_val(i,j);
             of << std::exp(xcoord) << " " << ycoord*std::exp(xcoord)
                << " " << val << "\n";
        }
    }
}

template <typename RealType>
void
qs_photon_emission_table_wrapper<RealType>::write_table_data(
    std::ofstream& of) const
{
    const auto how_many_x = this->m_table.get_how_many_x();
    const auto how_many_y = this->m_table.get_how_many_y();
    for (int i = 0; i < how_many_x; ++i){
        for (int j = 0; j < how_many_y; ++j){
            const auto xcoord = this->m_table.get_x_coord(i);
            const auto ycoord = this->m_table.get_y_coord(j);
            const auto val = this->m_table.get_val(i,j);
             of << std::exp(xcoord) << " " << std::exp(ycoord)*std::exp(xcoord)
                << " " << std::exp(val) << "\n";
        }
    }
}
