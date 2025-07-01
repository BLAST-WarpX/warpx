#include <filesystem>


int main()
{
    namespace fs = std::filesystem;

    fs::path dir{"./mkdir_test"};
    fs::create_directories(dir);

    return 0;
}
