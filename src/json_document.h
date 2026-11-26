#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace lfs {

struct JsonValue {
    enum class Type {
        Null,
        Bool,
        Number,
        String,
        Array,
        Object,
    };

    Type type{Type::Null};
    bool boolean{};
    double number{};
    std::string string{};
    std::vector<JsonValue> array{};
    std::map<std::string, JsonValue> object{};
};

class JsonParser {
public:
    explicit JsonParser(std::string text);

    [[nodiscard]] std::optional<JsonValue> parse();
    [[nodiscard]] const std::string &error() const noexcept { return error_; }

private:
    [[nodiscard]] char peek() const noexcept;
    [[nodiscard]] bool consume(char expected) noexcept;
    void skip_ws() noexcept;
    [[nodiscard]] std::optional<JsonValue> parse_value();
    [[nodiscard]] bool match_literal(const char *literal) noexcept;
    [[nodiscard]] std::optional<JsonValue> parse_string_value();
    [[nodiscard]] std::optional<std::string> parse_string();
    [[nodiscard]] std::optional<JsonValue> parse_number();
    [[nodiscard]] std::optional<JsonValue> parse_array();
    [[nodiscard]] std::optional<JsonValue> parse_object();

    std::string text_{};
    size_t pos_{};
    std::string error_{};
};

[[nodiscard]] const JsonValue *find_member(
    const JsonValue &object,
    const std::string &key) noexcept;

[[nodiscard]] std::string read_string(
    const JsonValue &object,
    const std::string &key,
    std::vector<std::string> &errors,
    bool required = true);

[[nodiscard]] std::string read_optional_string(
    const JsonValue &object,
    const std::string &key,
    std::vector<std::string> &errors);

[[nodiscard]] double read_number(
    const JsonValue &object,
    const std::string &key,
    std::vector<std::string> &errors);

[[nodiscard]] uint32_t read_u32(
    const JsonValue &object,
    const std::string &key,
    std::vector<std::string> &errors);

[[nodiscard]] uint64_t read_u64(
    const JsonValue &object,
    const std::string &key,
    std::vector<std::string> &errors);

[[nodiscard]] bool read_bool(
    const JsonValue &object,
    const std::string &key,
    std::vector<std::string> &errors);

[[nodiscard]] const JsonValue *read_object(
    const JsonValue &object,
    const std::string &key,
    std::vector<std::string> &errors);

[[nodiscard]] const std::vector<JsonValue> *read_array(
    const JsonValue &object,
    const std::string &key,
    std::vector<std::string> &errors);

} // namespace lfs
