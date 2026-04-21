#include "json.hpp"

#include <cctype>
#include <cstdlib>
#include <sstream>

namespace puzzlescript::json {
namespace {

void appendUtf8(std::string& output, uint32_t codePoint) {
    if (codePoint <= 0x7F) {
        output.push_back(static_cast<char>(codePoint));
    } else if (codePoint <= 0x7FF) {
        output.push_back(static_cast<char>(0xC0 | (codePoint >> 6)));
        output.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
    } else if (codePoint <= 0xFFFF) {
        output.push_back(static_cast<char>(0xE0 | (codePoint >> 12)));
        output.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F)));
        output.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
    } else {
        output.push_back(static_cast<char>(0xF0 | (codePoint >> 18)));
        output.push_back(static_cast<char>(0x80 | ((codePoint >> 12) & 0x3F)));
        output.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F)));
        output.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
    }
}

class Parser {
public:
    explicit Parser(std::string_view input)
        : input_(input) {}

    Value parseValue() {
        skipWhitespace();
        if (position_ >= input_.size()) {
            throw ParseError("Unexpected end of JSON input");
        }

        switch (input_[position_]) {
            case 'n': return parseNull();
            case 't': return parseTrue();
            case 'f': return parseFalse();
            case '"': return Value(parseString());
            case '[': return Value(parseArray());
            case '{': return Value(parseObject());
            default: return parseNumber();
        }
    }

private:
    void skipWhitespace() {
        while (position_ < input_.size() && std::isspace(static_cast<unsigned char>(input_[position_]))) {
            ++position_;
        }
    }

    bool consumeLiteral(std::string_view literal) {
        if (input_.substr(position_, literal.size()) == literal) {
            position_ += literal.size();
            return true;
        }
        return false;
    }

    Value parseNull() {
        if (!consumeLiteral("null")) {
            throw ParseError("Invalid JSON literal");
        }
        return Value();
    }

    Value parseTrue() {
        if (!consumeLiteral("true")) {
            throw ParseError("Invalid JSON literal");
        }
        return Value(true);
    }

    Value parseFalse() {
        if (!consumeLiteral("false")) {
            throw ParseError("Invalid JSON literal");
        }
        return Value(false);
    }

    std::string parseString() {
        if (input_[position_] != '"') {
            throw ParseError("Expected string");
        }
        ++position_;

        std::string output;
        while (position_ < input_.size()) {
            char ch = input_[position_++];
            if (ch == '"') {
                return output;
            }
            if (ch != '\\') {
                output.push_back(ch);
                continue;
            }

            if (position_ >= input_.size()) {
                throw ParseError("Invalid escape sequence");
            }

            char escape = input_[position_++];
            switch (escape) {
                case '"': output.push_back('"'); break;
                case '\\': output.push_back('\\'); break;
                case '/': output.push_back('/'); break;
                case 'b': output.push_back('\b'); break;
                case 'f': output.push_back('\f'); break;
                case 'n': output.push_back('\n'); break;
                case 'r': output.push_back('\r'); break;
                case 't': output.push_back('\t'); break;
                case 'u': {
                    if (position_ + 4 > input_.size()) {
                        throw ParseError("Invalid unicode escape");
                    }
                    uint32_t codePoint = 0;
                    for (int index = 0; index < 4; ++index) {
                        const char hex = input_[position_++];
                        codePoint <<= 4;
                        if (hex >= '0' && hex <= '9') {
                            codePoint |= static_cast<uint32_t>(hex - '0');
                        } else if (hex >= 'a' && hex <= 'f') {
                            codePoint |= static_cast<uint32_t>(10 + hex - 'a');
                        } else if (hex >= 'A' && hex <= 'F') {
                            codePoint |= static_cast<uint32_t>(10 + hex - 'A');
                        } else {
                            throw ParseError("Invalid unicode escape");
                        }
                    }

                    if (codePoint >= 0xD800 && codePoint <= 0xDBFF) {
                        if (position_ + 6 <= input_.size() && input_[position_] == '\\' && input_[position_ + 1] == 'u') {
                            position_ += 2;
                            uint32_t low = 0;
                            for (int index = 0; index < 4; ++index) {
                                const char hex = input_[position_++];
                                low <<= 4;
                                if (hex >= '0' && hex <= '9') {
                                    low |= static_cast<uint32_t>(hex - '0');
                                } else if (hex >= 'a' && hex <= 'f') {
                                    low |= static_cast<uint32_t>(10 + hex - 'a');
                                } else if (hex >= 'A' && hex <= 'F') {
                                    low |= static_cast<uint32_t>(10 + hex - 'A');
                                } else {
                                    throw ParseError("Invalid unicode escape");
                                }
                            }
                            if (low < 0xDC00 || low > 0xDFFF) {
                                throw ParseError("Invalid surrogate pair");
                            }
                            codePoint = 0x10000 + (((codePoint - 0xD800) << 10) | (low - 0xDC00));
                        } else {
                            throw ParseError("Missing low surrogate");
                        }
                    }

                    appendUtf8(output, codePoint);
                    break;
                }
                default:
                    throw ParseError("Invalid escape sequence");
            }
        }

        throw ParseError("Unterminated string");
    }

    Value parseNumber() {
        const size_t start = position_;
        if (input_[position_] == '-') {
            ++position_;
        }
        while (position_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[position_]))) {
            ++position_;
        }
        bool isDouble = false;
        if (position_ < input_.size() && input_[position_] == '.') {
            isDouble = true;
            ++position_;
            while (position_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[position_]))) {
                ++position_;
            }
        }
        if (position_ < input_.size() && (input_[position_] == 'e' || input_[position_] == 'E')) {
            isDouble = true;
            ++position_;
            if (position_ < input_.size() && (input_[position_] == '+' || input_[position_] == '-')) {
                ++position_;
            }
            while (position_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[position_]))) {
                ++position_;
            }
        }
        const std::string token(input_.substr(start, position_ - start));
        if (token.empty()) {
            throw ParseError("Expected JSON value");
        }
        if (isDouble) {
            return Value(std::strtod(token.c_str(), nullptr));
        }
        return Value(std::strtoll(token.c_str(), nullptr, 10));
    }

    Value::Array parseArray() {
        if (input_[position_] != '[') {
            throw ParseError("Expected array");
        }
        ++position_;

        Value::Array result;
        skipWhitespace();
        if (position_ < input_.size() && input_[position_] == ']') {
            ++position_;
            return result;
        }

        while (true) {
            result.push_back(parseValue());
            skipWhitespace();
            if (position_ >= input_.size()) {
                throw ParseError("Unterminated array");
            }
            const char ch = input_[position_++];
            if (ch == ']') {
                break;
            }
            if (ch != ',') {
                throw ParseError("Expected ',' or ']'");
            }
        }
        return result;
    }

    Value::Object parseObject() {
        if (input_[position_] != '{') {
            throw ParseError("Expected object");
        }
        ++position_;

        Value::Object result;
        skipWhitespace();
        if (position_ < input_.size() && input_[position_] == '}') {
            ++position_;
            return result;
        }

        while (true) {
            skipWhitespace();
            const std::string key = parseString();
            skipWhitespace();
            if (position_ >= input_.size() || input_[position_] != ':') {
                throw ParseError("Expected ':' after object key");
            }
            ++position_;
            result.emplace(key, parseValue());
            skipWhitespace();
            if (position_ >= input_.size()) {
                throw ParseError("Unterminated object");
            }
            const char ch = input_[position_++];
            if (ch == '}') {
                break;
            }
            if (ch != ',') {
                throw ParseError("Expected ',' or '}'");
            }
        }
        return result;
    }

    std::string_view input_;
    size_t position_ = 0;
};

} // namespace

Value::Value()
    : data_(nullptr) {}

Value::Value(bool value)
    : data_(value) {}

Value::Value(int64_t value)
    : data_(value) {}

Value::Value(double value)
    : data_(value) {}

Value::Value(std::string value)
    : data_(std::move(value)) {}

Value::Value(Array value)
    : data_(std::move(value)) {}

Value::Value(Object value)
    : data_(std::move(value)) {}

Value::Kind Value::kind() const {
    switch (data_.index()) {
        case 0: return Kind::Null;
        case 1: return Kind::Bool;
        case 2: return Kind::Integer;
        case 3: return Kind::Double;
        case 4: return Kind::String;
        case 5: return Kind::Array;
        case 6: return Kind::Object;
        default: throw ParseError("Invalid JSON value");
    }
}

bool Value::isNull() const { return std::holds_alternative<std::nullptr_t>(data_); }
bool Value::isBool() const { return std::holds_alternative<bool>(data_); }
bool Value::isInteger() const { return std::holds_alternative<int64_t>(data_); }
bool Value::isDouble() const { return std::holds_alternative<double>(data_); }
bool Value::isString() const { return std::holds_alternative<std::string>(data_); }
bool Value::isArray() const { return std::holds_alternative<Array>(data_); }
bool Value::isObject() const { return std::holds_alternative<Object>(data_); }

bool Value::asBool() const { return std::get<bool>(data_); }
int64_t Value::asInteger() const { return std::get<int64_t>(data_); }
double Value::asDouble() const { return std::holds_alternative<double>(data_) ? std::get<double>(data_) : static_cast<double>(std::get<int64_t>(data_)); }
const std::string& Value::asString() const { return std::get<std::string>(data_); }
const Value::Array& Value::asArray() const { return std::get<Array>(data_); }
const Value::Object& Value::asObject() const { return std::get<Object>(data_); }

const Value* Value::find(std::string_view key) const {
    if (!isObject()) {
        return nullptr;
    }
    const auto& object = asObject();
    const auto it = object.find(std::string(key));
    if (it == object.end()) {
        return nullptr;
    }
    return &it->second;
}

Value parse(std::string_view input) {
    Parser parser(input);
    Value value = parser.parseValue();
    return value;
}

} // namespace puzzlescript::json
