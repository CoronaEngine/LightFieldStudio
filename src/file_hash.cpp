#include "file_hash.h"
#include <fstream>
#include <iomanip>
#include <sstream>
namespace lfs {
std::string fnv1a64_file_hex(const std::filesystem::path &path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return {};
    }

    uint64_t value = 1469598103934665603ull;
    char buffer[4096]{};
    while (stream) {
        stream.read(buffer, sizeof(buffer));
        const std::streamsize count = stream.gcount();
        for (std::streamsize i = 0; i < count; ++i) {
            value ^= static_cast<unsigned char>(buffer[i]);
            value *= 1099511628211ull;
        }
    }

    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << value;
    return out.str();
}
}
