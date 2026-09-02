#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "clustered_lj_snapshot_producer.h"
#include "main.h"

namespace
{

void PrintUsage()
{
    std::cerr << "Usage: SPONGE_CLUSTERED_SNAPSHOT_PRODUCER "
                 "--snapshot-prefix <path> [--full-output] <SPONGE args...>\n";
}

}  // namespace

int main(int argc, char** argv)
{
    try
    {
        std::string snapshot_prefix;
        bool write_full_output = false;
        std::vector<std::string> sponge_args;
        sponge_args.reserve(argc > 0 ? static_cast<size_t>(argc) : 1u);
        sponge_args.push_back(argc > 0 ? argv[0]
                                       : "SPONGE_CLUSTERED_SNAPSHOT_PRODUCER");

        for (int i = 1; i < argc; i += 1)
        {
            const std::string arg = argv[i];
            if (arg == "--snapshot-prefix")
            {
                if (i + 1 >= argc)
                {
                    throw std::runtime_error(
                        "--snapshot-prefix requires a path");
                }
                snapshot_prefix = argv[++i];
            }
            else if (arg == "--full-output")
            {
                write_full_output = true;
            }
            else if (arg == "--help" || arg == "-h")
            {
                PrintUsage();
                return 0;
            }
            else
            {
                sponge_args.push_back(arg);
            }
        }

        if (snapshot_prefix.empty())
        {
            PrintUsage();
            throw std::runtime_error("missing --snapshot-prefix <path>");
        }

        std::vector<char*> sponge_argv;
        sponge_argv.reserve(sponge_args.size());
        for (std::string& arg : sponge_args)
        {
            sponge_argv.push_back(arg.data());
        }
        Main_Initial(static_cast<int>(sponge_argv.size()), sponge_argv.data());
        nbnxm_microbench::WriteCurrentClusteredLJSnapshots(snapshot_prefix,
                                                           write_full_output);
        Main_Clear();
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "SPONGE_CLUSTERED_SNAPSHOT_PRODUCER error: "
                  << error.what() << '\n';
        return 1;
    }
}
