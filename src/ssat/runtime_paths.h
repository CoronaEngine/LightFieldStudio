#pragma once

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <windows.h>

namespace lfs::denoise {

inline std::filesystem::path runtime_directory() {
    std::wstring path(32768, L'\0');
    const auto n=GetModuleFileNameW(nullptr,path.data(),static_cast<DWORD>(path.size()));
    if (!n || n==path.size()) throw std::runtime_error("Cannot locate the application runtime.");
    path.resize(n);
    return std::filesystem::path(path).parent_path();
}

inline std::filesystem::path user_data_directory() {
    wchar_t *value=nullptr;
    size_t count=0;
    if (_wdupenv_s(&value,&count,L"LOCALAPPDATA") || !value)
        throw std::runtime_error("LOCALAPPDATA is unavailable.");
    std::unique_ptr<wchar_t,decltype(&std::free)> owned(value,&std::free);
    return std::filesystem::path(owned.get()) / L"LightFieldStudio";
}

} // namespace lfs::denoise
