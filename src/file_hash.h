#pragma once
#include <filesystem>
#include <string>
namespace lfs {
std::string fnv1a64_file_hex(const std::filesystem::path &path);
}
