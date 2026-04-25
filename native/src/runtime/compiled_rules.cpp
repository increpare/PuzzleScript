#include "runtime/compiled_rules.hpp"

#include "runtime/hash.hpp"

extern "C" __attribute__((weak))
const puzzlescript::CompiledRulesBackend* ps_compiled_rules_find_backend(uint64_t) {
    return nullptr;
}

namespace puzzlescript {

uint64_t compiledRulesHashSource(std::string_view source) {
    return fnv1a64String(source);
}

void attachLinkedCompiledRules(Game& game, std::string_view source) {
    game.compiledRules = ps_compiled_rules_find_backend(compiledRulesHashSource(source));
}

} // namespace puzzlescript
