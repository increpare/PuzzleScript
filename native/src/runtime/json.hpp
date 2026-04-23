#pragma once

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace puzzlescript::json {

class ParseError : public std::runtime_error {
public:
    explicit ParseError(const std::string& message)
        : std::runtime_error(message) {}
};

class Value {
public:
    using Array = std::vector<Value>;
    using Object = std::map<std::string, Value>;

    enum class Kind {
        Null,
        Bool,
        Integer,
        Double,
        String,
        Array,
        Object,
    };

    Value();
    explicit Value(bool value);
    explicit Value(int64_t value);
    explicit Value(double value);
    explicit Value(std::string value);
    explicit Value(Array value);
    explicit Value(Object value);

    Kind kind() const;

    bool isNull() const;
    bool isBool() const;
    bool isInteger() const;
    bool isDouble() const;
    bool isString() const;
    bool isArray() const;
    bool isObject() const;

    bool asBool() const;
    int64_t asInteger() const;
    double asDouble() const;
    const std::string& asString() const;
    const Array& asArray() const;
    const Object& asObject() const;

    const Value* find(std::string_view key) const;

private:
    std::variant<std::nullptr_t, bool, int64_t, double, std::string, Array, Object> data_;
};

Value parse(std::string_view input);

} // namespace puzzlescript::json
