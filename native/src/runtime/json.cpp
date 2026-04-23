#include "runtime/json.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

#include "simdjson.h"

namespace puzzlescript::json {
namespace {

Value convert(simdjson::dom::element el) {
    using simdjson::dom::element_type;
    switch (el.type()) {
        case element_type::NULL_VALUE:
            return Value();
        case element_type::BOOL:
            return Value(static_cast<bool>(el.get_bool()));
        case element_type::INT64:
            return Value(static_cast<int64_t>(el.get_int64()));
        case element_type::UINT64:
            return Value(static_cast<int64_t>(el.get_uint64()));
        case element_type::DOUBLE:
            return Value(static_cast<double>(el.get_double()));
        case element_type::STRING: {
            std::string_view sv = el.get_string();
            return Value(std::string(sv));
        }
        case element_type::ARRAY: {
            Value::Array out;
            simdjson::dom::array arr = el.get_array();
            out.reserve(arr.size());
            for (auto child : arr) {
                out.push_back(convert(child));
            }
            return Value(std::move(out));
        }
        case element_type::OBJECT: {
            Value::Object out;
            simdjson::dom::object obj = el.get_object();
            for (auto field : obj) {
                std::string_view key = field.key;
                out.emplace(std::string(key), convert(field.value));
            }
            return Value(std::move(out));
        }
        case element_type::BIGINT:
            throw ParseError("simdjson BIGINT not supported by puzzlescript::json");
    }
    throw ParseError("unhandled simdjson element_type");
}

} // namespace

Value parse(std::string_view input) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded(input.data(), input.size());
    simdjson::dom::element root;
    auto err = parser.parse(padded).get(root);
    if (err) {
        throw ParseError(std::string("simdjson: ") + simdjson::error_message(err));
    }
    return convert(root);
}

// ----- Value implementation (unchanged semantics) -------------------------

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

} // namespace puzzlescript::json
