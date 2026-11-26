#include "json_document.h"

#include <cctype>
#include <cmath>
#include <limits>
#include <sstream>
#include <string_view>
#include <utility>

namespace lfs {

JsonParser::JsonParser(std::string text) : text_(std::move(text)) {
    if (text_.size() >= 3u &&
        static_cast<unsigned char>(text_[0]) == 0xefu &&
        static_cast<unsigned char>(text_[1]) == 0xbbu &&
        static_cast<unsigned char>(text_[2]) == 0xbfu) {
        text_.erase(0u, 3u);
    }
}

std::optional<JsonValue> JsonParser::parse() {
    skip_ws();
    auto value = parse_value();
    if (!value) {
        return std::nullopt;
    }
    skip_ws();
    if (pos_ != text_.size()) {
        error_ = "Unexpected trailing characters in JSON.";
        return std::nullopt;
    }
    return value;
}

char JsonParser::peek() const noexcept {
    return pos_ < text_.size() ? text_[pos_] : '\0';
}

bool JsonParser::consume(char expected) noexcept {
    if (peek() != expected) {
        return false;
    }
    ++pos_;
    return true;
}

void JsonParser::skip_ws() noexcept {
    while (pos_ < text_.size() &&
           std::isspace(static_cast<unsigned char>(text_[pos_]))) {
        ++pos_;
    }
}

std::optional<JsonValue> JsonParser::parse_value() {
    skip_ws();
    const char c = peek();
    if (c == '"') return parse_string_value();
    if (c == '{') return parse_object();
    if (c == '[') return parse_array();
    if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
    if (match_literal("true")) {
        JsonValue value;
        value.type = JsonValue::Type::Bool;
        value.boolean = true;
        return value;
    }
    if (match_literal("false")) {
        JsonValue value;
        value.type = JsonValue::Type::Bool;
        return value;
    }
    if (match_literal("null")) return JsonValue{};
    error_ = "Unexpected JSON token.";
    return std::nullopt;
}

bool JsonParser::match_literal(const char *literal) noexcept {
    const std::string_view view{literal};
    if (text_.compare(pos_, view.size(), view) != 0) {
        return false;
    }
    pos_ += view.size();
    return true;
}

std::optional<JsonValue> JsonParser::parse_string_value() {
    auto parsed = parse_string();
    if (!parsed) return std::nullopt;
    JsonValue value;
    value.type = JsonValue::Type::String;
    value.string = std::move(*parsed);
    return value;
}

std::optional<std::string> JsonParser::parse_string() {
    if (!consume('"')) {
        error_ = "Expected JSON string.";
        return std::nullopt;
    }
    std::string out;
    while (pos_ < text_.size()) {
        const char c = text_[pos_++];
        if (c == '"') return out;
        if (c != '\\') {
            out.push_back(c);
            continue;
        }
        if (pos_ >= text_.size()) {
            error_ = "Unterminated JSON escape.";
            return std::nullopt;
        }
        const char escaped = text_[pos_++];
        switch (escaped) {
            case '"':
            case '\\':
            case '/': out.push_back(escaped); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u':
                error_ = "Unicode JSON escapes are not supported in LFS manifests.";
                return std::nullopt;
            default:
                error_ = "Invalid JSON escape.";
                return std::nullopt;
        }
    }
    error_ = "Unterminated JSON string.";
    return std::nullopt;
}

std::optional<JsonValue> JsonParser::parse_number() {
    const size_t begin = pos_;
    if (peek() == '-') ++pos_;
    while (std::isdigit(static_cast<unsigned char>(peek()))) ++pos_;
    if (peek() == '.') {
        ++pos_;
        while (std::isdigit(static_cast<unsigned char>(peek()))) ++pos_;
    }
    if (peek() == 'e' || peek() == 'E') {
        ++pos_;
        if (peek() == '+' || peek() == '-') ++pos_;
        while (std::isdigit(static_cast<unsigned char>(peek()))) ++pos_;
    }
    double parsed{};
    std::istringstream in(text_.substr(begin, pos_ - begin));
    in >> parsed;
    if (!in || !in.eof() || !std::isfinite(parsed)) {
        error_ = "Invalid JSON number.";
        return std::nullopt;
    }
    JsonValue value;
    value.type = JsonValue::Type::Number;
    value.number = parsed;
    return value;
}

std::optional<JsonValue> JsonParser::parse_array() {
    if (!consume('[')) {
        error_ = "Expected JSON array.";
        return std::nullopt;
    }
    JsonValue value;
    value.type = JsonValue::Type::Array;
    skip_ws();
    if (consume(']')) return value;
    while (true) {
        auto item = parse_value();
        if (!item) return std::nullopt;
        value.array.push_back(std::move(*item));
        skip_ws();
        if (consume(']')) return value;
        if (!consume(',')) {
            error_ = "Expected comma in JSON array.";
            return std::nullopt;
        }
    }
}

std::optional<JsonValue> JsonParser::parse_object() {
    if (!consume('{')) {
        error_ = "Expected JSON object.";
        return std::nullopt;
    }
    JsonValue value;
    value.type = JsonValue::Type::Object;
    skip_ws();
    if (consume('}')) return value;
    while (true) {
        skip_ws();
        auto key = parse_string();
        if (!key) return std::nullopt;
        skip_ws();
        if (!consume(':')) {
            error_ = "Expected colon in JSON object.";
            return std::nullopt;
        }
        auto item = parse_value();
        if (!item) return std::nullopt;
        value.object.emplace(std::move(*key), std::move(*item));
        skip_ws();
        if (consume('}')) return value;
        if (!consume(',')) {
            error_ = "Expected comma in JSON object.";
            return std::nullopt;
        }
    }
}

const JsonValue *find_member(
    const JsonValue &object,
    const std::string &key) noexcept {
    if (object.type != JsonValue::Type::Object) return nullptr;
    const auto it = object.object.find(key);
    return it == object.object.end() ? nullptr : &it->second;
}

std::string read_string(
    const JsonValue &object,
    const std::string &key,
    std::vector<std::string> &errors,
    bool required) {
    const JsonValue *value = find_member(object, key);
    if (!value) {
        if (required) errors.push_back("Missing string field: " + key);
        return {};
    }
    if (value->type != JsonValue::Type::String) {
        errors.push_back("Field is not a string: " + key);
        return {};
    }
    return value->string;
}

std::string read_optional_string(
    const JsonValue &object,
    const std::string &key,
    std::vector<std::string> &errors) {
    const JsonValue *value = find_member(object, key);
    if (!value || value->type == JsonValue::Type::Null) return {};
    if (value->type != JsonValue::Type::String) {
        errors.push_back("Field is not a string or null: " + key);
        return {};
    }
    return value->string;
}

double read_number(
    const JsonValue &object,
    const std::string &key,
    std::vector<std::string> &errors) {
    const JsonValue *value = find_member(object, key);
    if (!value || value->type != JsonValue::Type::Number ||
        !std::isfinite(value->number)) {
        errors.push_back("Field is not a finite number: " + key);
        return 0.0;
    }
    return value->number;
}

uint32_t read_u32(
    const JsonValue &object,
    const std::string &key,
    std::vector<std::string> &errors) {
    const JsonValue *json = find_member(object, key);
    if (!json || json->type != JsonValue::Type::Number ||
        json->number < 0.0 || !std::isfinite(json->number) ||
        json->number > std::numeric_limits<uint32_t>::max()) {
        errors.push_back("Field is not a non-negative integer: " + key);
        return 0u;
    }
    const auto rounded = static_cast<uint32_t>(json->number);
    if (static_cast<double>(rounded) != json->number) {
        errors.push_back("Field is not an integer: " + key);
        return 0u;
    }
    return rounded;
}

uint64_t read_u64(
    const JsonValue &object,
    const std::string &key,
    std::vector<std::string> &errors) {
    const JsonValue *json = find_member(object, key);
    if (!json || json->type != JsonValue::Type::Number ||
        json->number < 0.0 || !std::isfinite(json->number)) {
        errors.push_back("Field is not a non-negative integer: " + key);
        return 0u;
    }
    const auto rounded = static_cast<uint64_t>(json->number);
    if (static_cast<double>(rounded) != json->number) {
        errors.push_back("Field is not an integer: " + key);
        return 0u;
    }
    return rounded;
}

bool read_bool(
    const JsonValue &object,
    const std::string &key,
    std::vector<std::string> &errors) {
    const JsonValue *value = find_member(object, key);
    if (!value || value->type != JsonValue::Type::Bool) {
        errors.push_back("Field is not a boolean: " + key);
        return false;
    }
    return value->boolean;
}

const JsonValue *read_object(
    const JsonValue &object,
    const std::string &key,
    std::vector<std::string> &errors) {
    const JsonValue *value = find_member(object, key);
    if (!value || value->type != JsonValue::Type::Object) {
        errors.push_back("Field is not an object: " + key);
        return nullptr;
    }
    return value;
}

const std::vector<JsonValue> *read_array(
    const JsonValue &object,
    const std::string &key,
    std::vector<std::string> &errors) {
    const JsonValue *value = find_member(object, key);
    if (!value || value->type != JsonValue::Type::Array) {
        errors.push_back(
            value ? "Field is not an array: " + key : "Missing array field: " + key);
        return nullptr;
    }
    return &value->array;
}

} // namespace lfs
