#include "compiler/lower_to_runtime.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <utf8proc.h>

#include "compiler/rule_text.hpp"

namespace puzzlescript::compiler {

namespace {

std::string toLowerAsciiCopy(std::string_view input) {
    std::string out;
    out.reserve(input.size());
    for (unsigned char ch : input) {
        out.push_back(static_cast<char>(std::tolower(ch)));
    }
    return out;
}

uint32_t ceilDivU32(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

puzzlescript::MaskOffset storeMaskWords(puzzlescript::Game& game, const puzzlescript::MaskVector& words) {
    const auto offset = static_cast<puzzlescript::MaskOffset>(game.maskArena.size());
    game.maskArena.insert(game.maskArena.end(), words.begin(), words.end());
    return offset;
}

puzzlescript::MaskVector makeEmptyMask(uint32_t wordCount) {
    return puzzlescript::MaskVector(static_cast<size_t>(wordCount), 0);
}

void setMaskBit(puzzlescript::MaskVector& words, int32_t bitIndex) {
    if (bitIndex < 0) {
        return;
    }
    const uint32_t word = puzzlescript::maskWordIndex(static_cast<uint32_t>(bitIndex));
    if (word >= words.size()) {
        return;
    }
    words[word] |= puzzlescript::maskBit(static_cast<uint32_t>(bitIndex));
}

bool maskHasBit(const puzzlescript::MaskVector& words, int32_t bitIndex) {
    if (bitIndex < 0) {
        return false;
    }
    const uint32_t word = puzzlescript::maskWordIndex(static_cast<uint32_t>(bitIndex));
    return word < words.size()
        && (words[word] & puzzlescript::maskBit(static_cast<uint32_t>(bitIndex))) != 0;
}

std::vector<int32_t> objectIdsFromMask(const puzzlescript::MaskVector& words, int32_t objectCount) {
    std::vector<int32_t> ids;
    for (uint32_t word = 0; word < words.size(); ++word) {
        puzzlescript::MaskWordUnsigned bits = static_cast<puzzlescript::MaskWordUnsigned>(words[static_cast<size_t>(word)]);
        while (bits != 0) {
            const int32_t bit = puzzlescript::maskWordCountTrailingZeros(bits);
            const int32_t objectId = static_cast<int32_t>(word) * static_cast<int32_t>(puzzlescript::kMaskWordBits) + bit;
            if (objectId < objectCount) {
                ids.push_back(objectId);
            }
            bits &= bits - 1;
        }
    }
    return ids;
}

std::vector<std::vector<int32_t>> parseSpriteMatrix(const std::vector<std::string>& rows) {
    // PuzzleScript sprites are typically 5x5; treat '.' as transparent (-1) and digits as palette indices.
    std::vector<std::vector<int32_t>> result;
    result.reserve(rows.size());
    for (const auto& row : rows) {
        std::vector<int32_t> outRow;
        outRow.reserve(row.size());
        for (const char ch : row) {
            if (ch == '.') {
                outRow.push_back(-1);
            } else if (ch >= '0' && ch <= '9') {
                outRow.push_back(static_cast<int32_t>(ch - '0'));
            } else {
                outRow.push_back(-1);
            }
        }
        result.push_back(std::move(outRow));
    }
    return result;
}

std::vector<std::string> splitUtf8Codepoints(std::string_view text) {
    std::vector<std::string> out;
    const auto* bytes = reinterpret_cast<const utf8proc_uint8_t*>(text.data());
    const utf8proc_ssize_t total = static_cast<utf8proc_ssize_t>(text.size());
    utf8proc_ssize_t cursor = 0;
    while (cursor < total) {
        utf8proc_int32_t cp = 0;
        const utf8proc_ssize_t advance = utf8proc_iterate(bytes + cursor, total - cursor, &cp);
        if (advance <= 0) {
            // Fall back to byte-wise to avoid infinite loops on malformed UTF-8.
            out.emplace_back(1, static_cast<char>(bytes[cursor]));
            cursor += 1;
            continue;
        }
        out.emplace_back(text.substr(static_cast<size_t>(cursor), static_cast<size_t>(advance)));
        cursor += advance;
    }
    return out;
}

std::string takeRulePrefixBeforeComment(std::string_view line) {
    std::string prefix;
    prefix.reserve(line.size());
    for (const char ch : line) {
        if (ch == '(') {
            break;
        }
        prefix.push_back(ch);
    }
    return prefix;
}

int32_t dirMaskFromToken(std::string_view token) {
    if (token == "^") return 1;
    if (token == "up") return 1;
    if (token == "v") return 2;
    if (token == "down") return 2;
    if (token == "<") return 4;
    if (token == "left") return 4;
    if (token == ">") return 8;
    if (token == "right") return 8;
    if (token == "action") return 16;
    if (token == "moving") return 1; // canonicalized to UP in JS rule masks
    if (token == "horizontal" || token == "horizontal_par" || token == "horizontal_perp") return 4;
    if (token == "vertical" || token == "vertical_par" || token == "vertical_perp") return 1;
    if (token == "orthogonal") return 15; // up|down|left|right
    // Aggregates: non-zero so `parseSide` pairs them with the following object name
    // (reg_directions in languageConstants.js).
    if (token == "perpendicular" || token == "parallel") return 1;
    if (token == "stationary") return 0; // special-cased: goes to movementsMissing=0x1f
    return 0;
}

void setShiftedMask5(puzzlescript::MaskVector& words, int32_t shift, int32_t value5) {
    // shift is bit offset (multiple of 5).
    const int32_t wordIndex = shift / static_cast<int32_t>(puzzlescript::kMaskWordBits);
    const int32_t bitIndex = shift % static_cast<int32_t>(puzzlescript::kMaskWordBits);
    if (wordIndex < 0 || static_cast<size_t>(wordIndex) >= words.size()) {
        return;
    }
    const puzzlescript::MaskWordUnsigned mask = 0x1fU;
    puzzlescript::MaskWordUnsigned v = static_cast<puzzlescript::MaskWordUnsigned>(value5) & mask;
    puzzlescript::MaskWordUnsigned w0 = static_cast<puzzlescript::MaskWordUnsigned>(words[static_cast<size_t>(wordIndex)]);
    w0 &= ~(mask << bitIndex);
    w0 |= (v << bitIndex);
    words[static_cast<size_t>(wordIndex)] = static_cast<puzzlescript::MaskWord>(w0);
    if (bitIndex > static_cast<int32_t>(puzzlescript::kMaskWordBits - 5U)) {
        // Straddles boundary.
        const int32_t next = wordIndex + 1;
        if (static_cast<size_t>(next) >= words.size()) {
            return;
        }
        const int32_t spill = bitIndex + 5 - static_cast<int32_t>(puzzlescript::kMaskWordBits);
        puzzlescript::MaskWordUnsigned w1 = static_cast<puzzlescript::MaskWordUnsigned>(words[static_cast<size_t>(next)]);
        w1 &= ~(mask >> (5 - spill));
        w1 |= (v >> (5 - spill));
        words[static_cast<size_t>(next)] = static_cast<puzzlescript::MaskWord>(w1);
    }
}

void orShiftedMask5(puzzlescript::MaskVector& words, int32_t shift, int32_t value5) {
    const int32_t wordIndex = shift / static_cast<int32_t>(puzzlescript::kMaskWordBits);
    const int32_t bitIndex = shift % static_cast<int32_t>(puzzlescript::kMaskWordBits);
    if (wordIndex < 0 || static_cast<size_t>(wordIndex) >= words.size()) {
        return;
    }
    const puzzlescript::MaskWordUnsigned mask = 0x1fU;
    puzzlescript::MaskWordUnsigned v = static_cast<puzzlescript::MaskWordUnsigned>(value5) & mask;
    puzzlescript::MaskWordUnsigned w0 = static_cast<puzzlescript::MaskWordUnsigned>(words[static_cast<size_t>(wordIndex)]);
    w0 |= (v << bitIndex);
    words[static_cast<size_t>(wordIndex)] = static_cast<puzzlescript::MaskWord>(w0);
    if (bitIndex > static_cast<int32_t>(puzzlescript::kMaskWordBits - 5U)) {
        const int32_t next = wordIndex + 1;
        if (static_cast<size_t>(next) >= words.size()) {
            return;
        }
        const int32_t spill = bitIndex + 5 - static_cast<int32_t>(puzzlescript::kMaskWordBits);
        puzzlescript::MaskWordUnsigned w1 = static_cast<puzzlescript::MaskWordUnsigned>(words[static_cast<size_t>(next)]);
        w1 |= (v >> (5 - spill));
        words[static_cast<size_t>(next)] = static_cast<puzzlescript::MaskWord>(w1);
    }
}

int32_t getShiftedMask5(const puzzlescript::MaskVector& words, int32_t shift) {
    const int32_t wordIndex = shift / static_cast<int32_t>(puzzlescript::kMaskWordBits);
    const int32_t bitIndex = shift % static_cast<int32_t>(puzzlescript::kMaskWordBits);
    if (wordIndex < 0 || static_cast<size_t>(wordIndex) >= words.size()) {
        return 0;
    }
    const puzzlescript::MaskWordUnsigned mask = 0x1fU;
    puzzlescript::MaskWordUnsigned value = (static_cast<puzzlescript::MaskWordUnsigned>(words[static_cast<size_t>(wordIndex)]) >> bitIndex) & mask;
    if (bitIndex > static_cast<int32_t>(puzzlescript::kMaskWordBits - 5U)) {
        const int32_t next = wordIndex + 1;
        if (static_cast<size_t>(next) < words.size()) {
            const int32_t spill = bitIndex + 5 - static_cast<int32_t>(puzzlescript::kMaskWordBits);
            const puzzlescript::MaskWordUnsigned nextBits = static_cast<puzzlescript::MaskWordUnsigned>(words[static_cast<size_t>(next)]) & ((puzzlescript::MaskWordUnsigned{1} << spill) - 1U);
            value |= (nextBits << (5 - spill));
        }
    }
    return static_cast<int32_t>(value);
}

} // namespace

std::unique_ptr<puzzlescript::Error> lowerToRuntimeGame(
    const ParserState& state,
    puzzlescript::LoadedGame& outGame
) {
    auto game = std::make_shared<puzzlescript::Game>();
    puzzlescript::MetaGameState initialMetaGameState;
    game->schemaVersion = 1;

    // --- Metadata ---
    // ParserState.metadata is a flat [key, value, key, value...] list.
    game->metadata.pairs = state.metadata;
    for (size_t i = 0; i + 1 < state.metadata.size(); i += 2) {
        game->metadata.values[state.metadata[i]] = state.metadata[i + 1];
    }
    game->metadata.lines = state.metadataLines;

    // Preserve existing JS exporter behavior: background/text colors are part of IR.
    // If we don't resolve palettes yet, prefer explicit metadata.
    if (const auto it = game->metadata.values.find("text_color"); it != game->metadata.values.end()) {
        game->foregroundColor = it->second;
    }
    if (const auto it = game->metadata.values.find("background_color"); it != game->metadata.values.end()) {
        game->backgroundColor = it->second;
    }

    // Colors (best-effort): default to black/white like the canonical engine
    // when not explicitly overridden. Palette resolution is a later step; for
    // IR-diff debugging we want stable, non-empty values.
    if (game->foregroundColor.empty()) {
        game->foregroundColor = "#FFFFFF";
    }
    if (game->backgroundColor.empty()) {
        game->backgroundColor = "#000000";
    }

    // --- Objects, layers, ids ---
    game->collisionLayers = state.collisionLayers;
    game->layerCount = static_cast<int32_t>(game->collisionLayers.size());

    std::vector<int32_t> idLayerById;
    int32_t idCount = 0;
    for (int32_t layerIndex = 0; layerIndex < static_cast<int32_t>(state.collisionLayers.size()); ++layerIndex) {
        for (const auto& name : state.collisionLayers[static_cast<size_t>(layerIndex)]) {
            if (state.objects.find(name) == state.objects.end()) {
                continue;
            }
            game->idDict.push_back(name);
            idLayerById.push_back(layerIndex);
            ++idCount;
        }
    }
    game->objectCount = idCount;

    game->strideObject = static_cast<int32_t>(ceilDivU32(static_cast<uint32_t>(game->objectCount), puzzlescript::kMaskWordBits));
    game->wordCount = static_cast<uint32_t>(game->strideObject);
    game->strideMovement = static_cast<int32_t>(puzzlescript::movementStrideWordCount(static_cast<uint32_t>(game->layerCount)));
    game->movementWordCount = static_cast<uint32_t>(game->strideMovement);

    game->objectsById.resize(static_cast<size_t>(game->objectCount));

    // Build object defs and objectMaskTable.
    game->objectMaskTable.clear();
    game->objectMaskTable.reserve(static_cast<size_t>(game->objectCount));

    // Name -> id lookup by *last* idDict index. Some games accidentally
    // duplicate object names in collision layers; JS ends up using the later id
    // as the canonical one (and leaves earlier slots without an object def in
    // the IR objects list), but ids still exist in id_dict/collision_layers.
    std::map<std::string, int32_t> objectIdByName;
    for (int32_t id = 0; id < static_cast<int32_t>(game->idDict.size()); ++id) {
        const auto& nm = game->idDict[static_cast<size_t>(id)];
        objectIdByName[nm] = id;
    }

    // Fill objectsById for every id entry. (Even if JS omits duplicates from
    // the serialized `objects` list, runtime logic still expects stable ids,
    // layers, and names.)
    for (int32_t id = 0; id < static_cast<int32_t>(game->idDict.size()); ++id) {
            const auto& name = game->idDict[static_cast<size_t>(id)];
            const auto it = state.objects.find(name);
            if (it == state.objects.end()) {
                continue;
            }
            const int32_t layerIndex = (static_cast<size_t>(id) < idLayerById.size()) ? idLayerById[static_cast<size_t>(id)] : 0;
            const auto canonIt = objectIdByName.find(name);
            const bool isCanonical = (canonIt != objectIdByName.end() && canonIt->second == id);
            puzzlescript::ObjectDef def;
            def.name = name;
            def.id = id;
            // Non-canonical duplicate ids exist in id_dict, but JS does not
            // treat them as real objects for layer masks / clearing.
            def.layer = isCanonical ? layerIndex : -1;
            if (isCanonical) {
                def.colors = it->second.colors;
                if (!it->second.spritematrix.empty()) {
                    def.sprite = parseSpriteMatrix(it->second.spritematrix);
                } else {
                    def.sprite = std::vector<std::vector<int32_t>>(5, std::vector<int32_t>(5, 0));
                }
            }
            game->objectsById[static_cast<size_t>(id)] = std::move(def);

            // objectMaskTable is name-keyed; keep one entry per name (canonical id).
            if (canonIt != objectIdByName.end() && canonIt->second == id) {
                auto mask = makeEmptyMask(game->wordCount);
                setMaskBit(mask, id);
                const auto offset = storeMaskWords(*game, mask);
                game->objectMaskTable.push_back({name, offset});
            }
    }

    // layer masks
    game->layerMaskOffsets.clear();
    game->layerMaskOffsets.reserve(static_cast<size_t>(game->layerCount));
    for (int32_t layerIndex = 0; layerIndex < game->layerCount; ++layerIndex) {
        auto mask = makeEmptyMask(game->wordCount);
        for (int32_t id = 0; id < game->objectCount; ++id) {
            if (game->objectsById[static_cast<size_t>(id)].layer == layerIndex) {
                setMaskBit(mask, id);
            }
        }
        game->layerMaskOffsets.push_back(storeMaskWords(*game, mask));
    }

    // --- Background / player masks ---

    // --- Legend resolution (name -> object mask) ---
    std::map<std::string, puzzlescript::MaskVector> resolvedMasks;
    std::map<std::string, std::string> synonymOf;
    std::map<std::string, std::vector<std::string>> aggregateOf;
    std::map<std::string, std::vector<std::string>> propertyOf;

    for (const auto& entry : state.legendSynonyms) {
        if (!entry.items.empty()) {
            synonymOf[toLowerAsciiCopy(entry.name)] = toLowerAsciiCopy(entry.items.front());
        }
    }
    for (const auto& entry : state.legendAggregates) {
        std::vector<std::string> items;
        items.reserve(entry.items.size());
        for (const auto& item : entry.items) items.push_back(toLowerAsciiCopy(item));
        aggregateOf[toLowerAsciiCopy(entry.name)] = std::move(items);
    }
    for (const auto& entry : state.legendProperties) {
        std::vector<std::string> items;
        items.reserve(entry.items.size());
        for (const auto& item : entry.items) items.push_back(toLowerAsciiCopy(item));
        propertyOf[toLowerAsciiCopy(entry.name)] = std::move(items);
    }
    {
        bool modified = true;
        while (modified) {
            modified = false;

            std::vector<std::string> synonymKeys;
            synonymKeys.reserve(synonymOf.size());
            for (const auto& [name, _] : synonymOf) {
                synonymKeys.push_back(name);
            }
            for (const auto& name : synonymKeys) {
                auto it = synonymOf.find(name);
                if (it == synonymOf.end()) {
                    continue;
                }
                const std::string value = it->second;
                if (const auto propIt = propertyOf.find(value); propIt != propertyOf.end()) {
                    propertyOf[name] = propIt->second;
                    synonymOf.erase(it);
                    modified = true;
                } else if (const auto aggIt = aggregateOf.find(value); aggIt != aggregateOf.end()) {
                    aggregateOf[name] = aggIt->second;
                    synonymOf.erase(it);
                    modified = true;
                } else if (const auto synIt = synonymOf.find(value); synIt != synonymOf.end()) {
                    it->second = synIt->second;
                }
            }

            std::vector<std::string> propertyKeys;
            propertyKeys.reserve(propertyOf.size());
            for (const auto& [name, _] : propertyOf) {
                propertyKeys.push_back(name);
            }
            for (const auto& name : propertyKeys) {
                auto it = propertyOf.find(name);
                if (it == propertyOf.end()) {
                    continue;
                }
                auto& values = it->second;
                for (size_t i = 0; i < values.size(); ++i) {
                    const std::string value = values[i];
                    if (const auto synIt = synonymOf.find(value); synIt != synonymOf.end()) {
                        values[i] = synIt->second;
                        modified = true;
                        continue;
                    }
                    const auto propIt = propertyOf.find(value);
                    if (propIt != propertyOf.end()) {
                        values.erase(values.begin() + static_cast<std::ptrdiff_t>(i));
                        for (const auto& expanded : propIt->second) {
                            if (std::find(values.begin(), values.end(), expanded) == values.end()) {
                                values.push_back(expanded);
                            }
                        }
                        modified = true;
                        --i;
                    }
                }
            }

            std::vector<std::string> aggregateKeys;
            aggregateKeys.reserve(aggregateOf.size());
            for (const auto& [name, _] : aggregateOf) {
                aggregateKeys.push_back(name);
            }
            for (const auto& name : aggregateKeys) {
                auto it = aggregateOf.find(name);
                if (it == aggregateOf.end()) {
                    continue;
                }
                auto& values = it->second;
                for (size_t i = 0; i < values.size(); ++i) {
                    const std::string value = values[i];
                    if (const auto synIt = synonymOf.find(value); synIt != synonymOf.end()) {
                        values[i] = synIt->second;
                        modified = true;
                        continue;
                    }
                    const auto aggIt = aggregateOf.find(value);
                    if (aggIt != aggregateOf.end()) {
                        values.erase(values.begin() + static_cast<std::ptrdiff_t>(i));
                        for (const auto& expanded : aggIt->second) {
                            if (std::find(values.begin(), values.end(), expanded) == values.end()) {
                                values.push_back(expanded);
                            }
                        }
                        modified = true;
                        --i;
                    }
                }
            }
        }
    }

    // Mirrors compiler.js `propertiesSingleLayer`: OR-properties whose members all
    // share one collision layer (used to skip `concretizePropertyRule` explosion).
    std::map<std::string, int32_t> propertiesSingleLayer;
    for (const auto& [propName, aliases] : propertyOf) {
        if (aliases.empty()) {
            continue;
        }
        std::optional<int32_t> commonLayer;
        bool ok = true;
        for (const auto& al : aliases) {
            const auto it = objectIdByName.find(al);
            if (it == objectIdByName.end()) {
                ok = false;
                break;
            }
            const int32_t L = game->objectsById[static_cast<size_t>(it->second)].layer;
            if (!commonLayer.has_value()) {
                commonLayer = L;
            } else if (*commonLayer != L) {
                ok = false;
                break;
            }
        }
        if (ok && commonLayer.has_value()) {
            propertiesSingleLayer[propName] = *commonLayer;
        }
    }

    // JS-style glyphDict: maps a glyph name to a per-layer concrete object id,
    // where -1 means "no object for this layer". This includes:
    // - concrete objects
    // - synonyms
    // - aggregates (AND)
    // Properties (OR) are intentionally *excluded* (ambiguous in maps).
    const std::vector<int32_t> blankGlyph(static_cast<size_t>(game->layerCount), -1);
    std::map<std::string, std::vector<int32_t>> glyphDict;

    for (const auto& [name, id] : objectIdByName) {
        const int32_t layer = game->objectsById[static_cast<size_t>(id)].layer;
        auto glyph = blankGlyph;
        if (layer >= 0 && layer < game->layerCount) {
            glyph[static_cast<size_t>(layer)] = id;
        }
        glyphDict.emplace(name, std::move(glyph));
    }

    bool added = true;
    while (added) {
        added = false;
        // synonyms
        for (const auto& entry : state.legendSynonyms) {
            if (entry.items.empty()) {
                continue;
            }
            const auto key = toLowerAsciiCopy(entry.name);
            const auto val = toLowerAsciiCopy(entry.items.front());
            if (glyphDict.find(key) == glyphDict.end()) {
                const auto it = glyphDict.find(val);
                if (it != glyphDict.end()) {
                    glyphDict.emplace(key, it->second);
                    added = true;
                }
            }
        }
        // aggregates (AND)
        for (const auto& entry : state.legendAggregates) {
            const auto key = toLowerAsciiCopy(entry.name);
            if (glyphDict.find(key) != glyphDict.end()) {
                continue;
            }
            bool allFound = true;
            for (const auto& item : entry.items) {
                if (glyphDict.find(toLowerAsciiCopy(item)) == glyphDict.end()) {
                    allFound = false;
                    break;
                }
            }
            if (!allFound) {
                continue;
            }
            auto glyph = blankGlyph;
            for (const auto& item : entry.items) {
                const auto& sub = glyphDict[toLowerAsciiCopy(item)];
                for (size_t layer = 0; layer < glyph.size(); ++layer) {
                    if (sub[layer] >= 0) {
                        glyph[layer] = sub[layer];
                    }
                }
            }
            glyphDict.emplace(key, std::move(glyph));
            added = true;
        }
        // properties (OR) intentionally skipped for glyphDict (ambiguous in maps)
    }

    auto resolveMask = [&](auto&& self, const std::string& name, std::set<std::string>& visiting) -> puzzlescript::MaskVector {
        if (auto it = resolvedMasks.find(name); it != resolvedMasks.end()) {
            return it->second;
        }
        if (!visiting.insert(name).second) {
            throw std::runtime_error("Legend cycle detected at '" + name + "'");
        }

        puzzlescript::MaskVector mask = makeEmptyMask(game->wordCount);

        if (auto it = objectIdByName.find(name); it != objectIdByName.end()) {
            setMaskBit(mask, it->second);
        } else if (auto it = synonymOf.find(name); it != synonymOf.end()) {
            mask = self(self, it->second, visiting);
        } else if (auto it = aggregateOf.find(name); it != aggregateOf.end()) {
            for (const auto& item : it->second) {
                auto itemMask = self(self, item, visiting);
                for (size_t w = 0; w < mask.size(); ++w) {
                    mask[w] |= itemMask[w];
                }
            }
        } else if (auto it = propertyOf.find(name); it != propertyOf.end()) {
            for (const auto& item : it->second) {
                auto itemMask = self(self, item, visiting);
                for (size_t w = 0; w < mask.size(); ++w) {
                    mask[w] |= itemMask[w];
                }
            }
        } else {
            // Unknown legend key: leave as empty mask for now.
        }

        visiting.erase(name);
        resolvedMasks.emplace(name, mask);
        return mask;
    };

    // Player mask: prefer a concrete object named "player"; otherwise resolve
    // the legend key "player" (common: Player = Foo or Bar).
    {
        auto playerMaskWords = makeEmptyMask(game->wordCount);
        const auto playerIt = objectIdByName.find("player");
        if (playerIt != objectIdByName.end()) {
            setMaskBit(playerMaskWords, playerIt->second);
            game->playerMaskAggregate = false;
        } else {
            try {
                std::set<std::string> visiting;
                playerMaskWords = resolveMask(resolveMask, "player", visiting);
                game->playerMaskAggregate = (aggregateOf.find("player") != aggregateOf.end());
            } catch (...) {
                // leave empty
            }
        }
        game->playerMask = storeMaskWords(*game, playerMaskWords);
    }

    // Resolve background object/property.
    puzzlescript::MaskVector backgroundMaskWords = makeEmptyMask(game->wordCount);
    int32_t backgroundLayer = -1;
    {
        // Prefer a concrete object named "background" (matches JS).
        const auto bgObjIt = objectIdByName.find("background");
        if (bgObjIt != objectIdByName.end()) {
            setMaskBit(backgroundMaskWords, bgObjIt->second);
            backgroundLayer = game->objectsById[static_cast<size_t>(bgObjIt->second)].layer;
            game->backgroundId = bgObjIt->second;
            game->backgroundLayer = backgroundLayer;
        } else {
            try {
                std::set<std::string> visiting;
                backgroundMaskWords = resolveMask(resolveMask, "background", visiting);
                // Infer background layer from first set bit.
                for (int32_t id = 0; id < game->objectCount; ++id) {
                    if (maskHasBit(backgroundMaskWords, id)) {
                        backgroundLayer = game->objectsById[static_cast<size_t>(id)].layer;
                        game->backgroundId = id;
                        game->backgroundLayer = backgroundLayer;
                        break;
                    }
                }
                // JS semantics: background must be a *single concrete object* for
                // map default fills. If background is a property/aggregate, pick
                // the first concrete object id and use only that bit.
                if (game->backgroundId >= 0) {
                    backgroundMaskWords = makeEmptyMask(game->wordCount);
                    setMaskBit(backgroundMaskWords, game->backgroundId);
                }
            } catch (...) {
                // Leave unset; suite-green will enforce correctness.
            }
        }
    }

    // --- Levels ---
    game->levels.clear();
    game->levels.reserve(state.levels.size());
    for (const auto& srcLevel : state.levels) {
        // Parser may retain empty placeholder level entries (notably a trailing
        // one); JS compiler drops them.
        if (!srcLevel.isMessage && srcLevel.rows.empty() && !srcLevel.lineNumber.has_value()) {
            continue;
        }
        puzzlescript::LevelTemplate level;
        level.isMessage = srcLevel.isMessage;
        if (level.isMessage) {
            level.message = srcLevel.message;
            game->levels.push_back(std::move(level));
            continue;
        }
        if (srcLevel.lineNumber.has_value()) {
            level.lineNumber = *srcLevel.lineNumber;
        }
        level.height = static_cast<int32_t>(srcLevel.rows.size());
        if (!srcLevel.rows.empty()) {
            level.width = static_cast<int32_t>(splitUtf8Codepoints(srcLevel.rows.front()).size());
        } else {
            level.width = 0;
        }
        const int32_t tileCount = level.width * level.height;
        level.objects.assign(static_cast<size_t>(tileCount * game->strideObject), 0);

        const auto glyphAt = [](const std::vector<std::string>& glyphs, int32_t x) -> std::string {
            if (glyphs.empty()) {
                return {};
            }
            if (x < static_cast<int32_t>(glyphs.size())) {
                return glyphs[static_cast<size_t>(x)];
            }
            // JS levelFromString repeats the row's last character for ragged rows.
            return glyphs.back();
        };

        // Per-level default background: if the level explicitly uses a concrete
        // background-layer glyph (e.g. WoodenFloor), JS effectively treats that
        // as the fill under obstacles/walls in cells without background glyphs.
        int32_t levelBackgroundId = game->backgroundId;
        bool foundLevelBackground = false;
        if (backgroundLayer >= 0) {
            for (int32_t y = 0; y < level.height && !foundLevelBackground; ++y) {
                const auto glyphs = splitUtf8Codepoints(srcLevel.rows[static_cast<size_t>(y)]);
                for (int32_t x = 0; x < level.width; ++x) {
                    const std::string glyph = glyphAt(glyphs, x);
                    if (glyph.empty()) {
                        continue;
                    }
                    const auto it = glyphDict.find(glyph);
                    if (it == glyphDict.end()) {
                        continue;
                    }
                    const auto& perLayer = it->second;
                    if (static_cast<size_t>(backgroundLayer) < perLayer.size()) {
                        const int32_t id = perLayer[static_cast<size_t>(backgroundLayer)];
                        if (id >= 0) {
                            levelBackgroundId = id;
                            foundLevelBackground = true;
                            break;
                        }
                    }
                }
            }
        }
        auto levelBackgroundMaskWords = makeEmptyMask(game->wordCount);
        if (levelBackgroundId >= 0) {
            setMaskBit(levelBackgroundMaskWords, levelBackgroundId);
        }

        for (int32_t y = 0; y < level.height; ++y) {
            const auto glyphs = splitUtf8Codepoints(srcLevel.rows[static_cast<size_t>(y)]);
            for (int32_t x = 0; x < level.width; ++x) {
                const std::string glyph = glyphAt(glyphs, x);
                puzzlescript::MaskVector cellMask = makeEmptyMask(game->wordCount);
                if (!glyph.empty()) {
                    try {
                        const auto it = glyphDict.find(glyph);
                        if (it != glyphDict.end()) {
                            const auto& perLayer = it->second;
                            for (size_t layer = 0; layer < perLayer.size(); ++layer) {
                                const int32_t id = perLayer[layer];
                                if (id >= 0) {
                                    setMaskBit(cellMask, id);
                                }
                            }
                        }
                    } catch (...) {
                        // Ignore glyph failures for now; suite-green will harden this.
                    }
                }

                // JS semantics: if background layer is empty in this cell, add background mask.
                if (backgroundLayer >= 0 && static_cast<size_t>(backgroundLayer) < game->layerMaskOffsets.size()) {
                    const auto layerMaskOffset = game->layerMaskOffsets[static_cast<size_t>(backgroundLayer)];
                    const puzzlescript::MaskWord* layerMask = game->maskArena.data() + layerMaskOffset;
                    bool anyBackgroundLayer = false;
                    for (uint32_t w = 0; w < game->wordCount; ++w) {
                        if ((cellMask[w] & layerMask[w]) != 0) {
                            anyBackgroundLayer = true;
                            break;
                        }
                    }
                    if (!anyBackgroundLayer) {
                        for (size_t w = 0; w < cellMask.size(); ++w) {
                            cellMask[w] |= levelBackgroundMaskWords[w];
                        }
                    }
                }

                // Runtime tile indexing is column-major: tileIndex = x*height + y.
                const int32_t tileIndex = x * level.height + y;
                const size_t base = static_cast<size_t>(tileIndex * game->strideObject);
                for (int32_t w = 0; w < game->strideObject; ++w) {
                    level.objects[base + static_cast<size_t>(w)] = cellMask[static_cast<size_t>(w)];
                }
            }
        }
        game->levels.push_back(std::move(level));
    }

    // --- Prepared session ---
    initialMetaGameState.currentLevelIndex = 0;
    initialMetaGameState.currentLevelTarget.reset();
    initialMetaGameState.titleScreen = false;
    initialMetaGameState.textMode = !game->levels.empty() && game->levels.front().isMessage;
    initialMetaGameState.titleMode = 0;
    initialMetaGameState.titleSelection = 0;
    initialMetaGameState.titleSelected = false;
    initialMetaGameState.messageSelected = false;
    initialMetaGameState.messageText.clear();
    initialMetaGameState.winning = false;
    initialMetaGameState.loadedLevelSeed = "native";
    initialMetaGameState.hasRandomState = false;
    initialMetaGameState.randomStateValid = false;
    initialMetaGameState.randomStateS.clear();
    initialMetaGameState.oldFlickscreenDat.clear();
    if (!game->levels.empty()) {
        initialMetaGameState.level = game->levels.front();
        puzzlescript::fillCompactOccupancyBitsFromInterpreterBoardData(
            *game,
            initialMetaGameState.level.width,
            initialMetaGameState.level.height,
            initialMetaGameState.level.objects,
            initialMetaGameState.restart.objectBits
        );
        initialMetaGameState.restart.oldFlickscreenDat.clear();
    }

    // --- Rules / winconditions / sounds / loop points ---
    game->rules.clear();
    game->lateRules.clear();
    std::vector<int32_t> loopStartStack;
    std::vector<std::pair<int32_t, int32_t>> loopRanges;

    // Precompute (best-effort) single-layer info for legend names: if a mask's
    // set bits all live on the same collision layer, we can treat it as
    // single-layer for rule movement masks.
    auto maskSingleLayer = [&](const puzzlescript::MaskVector& mask) -> std::optional<int32_t> {
        std::optional<int32_t> layer;
        for (int32_t id = 0; id < game->objectCount; ++id) {
            if (!maskHasBit(mask, id)) {
                continue;
            }
            const int32_t objLayer = game->objectsById[static_cast<size_t>(id)].layer;
            if (!layer.has_value()) {
                layer = objLayer;
            } else if (*layer != objLayer) {
                return std::nullopt;
            }
        }
        return layer;
    };

    std::vector<std::vector<std::string>> earlyRuleSignatures;
    std::vector<std::vector<std::string>> lateRuleSignatures;

    // Rule lowering: a subset of JS rulesToMask (enough to start converging).
    for (const auto& entry : state.rules) {
        const auto tokens = ruletext::tokenizeRuleLine(entry.rule);
        const auto mixedCaseTokens = ruletext::tokenizeRuleLine(takeRulePrefixBeforeComment(entry.mixedCase));
        if (tokens.empty()) {
            continue;
        }
        // Handle loop markers.
        const std::string marker = tokens.front();
        if (marker == "startloop" || marker == "endloop") {
            if (marker == "startloop") {
                loopStartStack.push_back(entry.lineNumber);
            } else {
                if (!loopStartStack.empty()) {
                    const int32_t startLine = loopStartStack.back();
                    loopStartStack.pop_back();
                    loopRanges.emplace_back(startLine, entry.lineNumber);
                }
            }
            continue;
        }
        const bool hasAnyArrowToken = std::find(tokens.begin(), tokens.end(), "->") != tokens.end();
        auto arrowIt = tokens.end();
        int32_t arrowSearchBracketDepth = 0;
        for (auto it = tokens.begin(); it != tokens.end(); ++it) {
            if (*it == "[") {
                ++arrowSearchBracketDepth;
                continue;
            }
            if (*it == "]") {
                --arrowSearchBracketDepth;
                continue;
            }
            if (*it == "->" && arrowSearchBracketDepth == 0) {
                arrowIt = it;
                break;
            }
        }
        if (arrowIt == tokens.end() && !hasAnyArrowToken) {
            continue;
        }

        // Directions/modifiers at start (optional). JS defaults rules with no
        // explicit direction to orthogonal, then (for non-directional rules only)
        // keeps the first scan direction only — see `directionalRule` + `splice(1)`
        // in compiler.js `processRuleString`.
        size_t cursor = 0;
        bool rigidRule = false;
        bool randomRule = false;
        bool lateRule = false;
        bool sameGroup = false;
        std::vector<std::string> ruleDirections;
        auto addDirectionAggregate = [&](const std::string& token) {
            if (token == "horizontal") {
                ruleDirections.push_back("left");
                ruleDirections.push_back("right");
            } else if (token == "vertical") {
                ruleDirections.push_back("up");
                ruleDirections.push_back("down");
            } else if (token == "orthogonal") {
                ruleDirections.push_back("up");
                ruleDirections.push_back("down");
                ruleDirections.push_back("left");
                ruleDirections.push_back("right");
            }
        };
        while (cursor < tokens.size() && tokens[cursor] != "[") {
            const std::string token = tokens[cursor];
            if (token == "up" || token == "down" || token == "left" || token == "right") {
                ruleDirections.push_back(token);
            } else if (token == "horizontal" || token == "vertical" || token == "orthogonal") {
                addDirectionAggregate(token);
            } else if (token == "rigid") {
                rigidRule = true;
            } else if (token == "random") {
                randomRule = true;
            } else if (token == "late") {
                lateRule = true;
            } else if (token == "+") {
                sameGroup = true;
            }
            ++cursor;
        }
        if (ruleDirections.empty()) {
            addDirectionAggregate("orthogonal");
        }

        // JS grouping: each rule line starts a new group unless prefixed by "+".
        std::vector<std::vector<puzzlescript::Rule>>& groups = lateRule ? game->lateRules : game->rules;
        std::vector<std::vector<std::string>>& groupSignatures = lateRule ? lateRuleSignatures : earlyRuleSignatures;
        if (groups.empty() || !sameGroup) {
            groups.emplace_back();
            groupSignatures.emplace_back();
        }
        std::vector<puzzlescript::Rule>* outputGroup = &groups.back();
        std::vector<std::string>* outputSignatures = &groupSignatures.back();

        struct ParsedItem {
            std::string dir;
            std::string name;
        };
        struct ParsedCell {
            bool isEllipsis = false;
            std::vector<ParsedItem> items;
        };
        using ParsedRow = std::vector<ParsedCell>;

        // Mirrors src/js/languageConstants.js `commandwords` for tokens that may
        // appear inside RHS brackets (legacy): sfxN / cancel / checkpoint / …
        // Sound-only names like `Sfx1` lower-case to `sfx1` and are not in
        // `state.names` in JS, so they become commands rather than cell objects.
        auto cellNameRefersToLegendOrObject = [&](const std::string& name) -> bool {
            return objectIdByName.find(name) != objectIdByName.end()
                || synonymOf.find(name) != synonymOf.end()
                || aggregateOf.find(name) != aggregateOf.end()
                || propertyOf.find(name) != propertyOf.end();
        };
        auto isJsBracketPostfixCommand = [](const std::string& name) -> bool {
            static constexpr std::array<const char*, 17> kWords = {
                "sfx0", "sfx1", "sfx2", "sfx3", "sfx4", "sfx5", "sfx6", "sfx7", "sfx8", "sfx9", "sfx10",
                "cancel", "checkpoint", "restart", "win", "message", "again",
            };
            for (const char* w : kWords) {
                if (name == w) {
                    return true;
                }
            }
            return false;
        };

        auto parseSide = [&](size_t start, size_t end, std::vector<puzzlescript::RuleCommand>* inlineCommandSink) -> std::vector<ParsedRow> {
            std::vector<ParsedRow> rows;
            size_t i = start;
            while (i < end) {
                if (tokens[i] != "[") {
                    ++i;
                    continue;
                }
                ++i; // consume '['
                ParsedRow current;
                ParsedCell cell;
                while (i < end && tokens[i] != "]") {
                    if (tokens[i] == "|") {
                        current.push_back(std::move(cell));
                        cell = ParsedCell{};
                        ++i;
                        continue;
                    }
                    if (tokens[i] == "...") {
                        cell.isEllipsis = true;
                        ++i;
                        continue;
                    }
                    std::string dir;
                    std::string name = tokens[i];
                    const std::string tokLower = toLowerAsciiCopy(tokens[i]);
                    if ((dirMaskFromToken(tokLower) != 0 || tokLower == "stationary" || tokLower == "no" || tokLower == "random"
                         || tokLower == "randomdir")
                        && (i + 1) < end) {
                        dir = tokLower;
                        name = tokens[i + 1];
                        i += 2;
                    } else {
                        i += 1;
                    }
                    if (name == "|") {
                        continue;
                    }
                    // Legend/property maps are lower-case (see legend parsing). JS rule
                    // matching is case-insensitive for object/property names and movement
                    // keywords like MOVING / STATIONARY.
                    const std::string nameNorm = toLowerAsciiCopy(name);
                    std::string dirNorm = dir;
                    if (!dirNorm.empty()) {
                        dirNorm = toLowerAsciiCopy(dirNorm);
                    }
                    if (inlineCommandSink != nullptr && dirNorm.empty() && isJsBracketPostfixCommand(nameNorm)
                        && !cellNameRefersToLegendOrObject(nameNorm)) {
                        puzzlescript::RuleCommand cmd;
                        cmd.name = nameNorm;
                        inlineCommandSink->push_back(std::move(cmd));
                        continue;
                    }
                    if (!cellNameRefersToLegendOrObject(nameNorm)) {
                        continue;
                    }
                    cell.items.push_back({std::move(dirNorm), std::move(nameNorm)});
                }
                // Always push the last cell, even if empty. This is required for
                // rules like `[ | | ]` where empty RHS cells represent clearing.
                current.push_back(std::move(cell));
                if (i < end && tokens[i] == "]") {
                    ++i;
                }
                if (!current.empty()) {
                    rows.push_back(std::move(current));
                }
            }
            return rows;
        };

        const size_t arrowPos = arrowIt == tokens.end()
            ? tokens.size()
            : static_cast<size_t>(std::distance(tokens.begin(), arrowIt));
        const size_t rhsEnd = tokens.size();
        std::vector<puzzlescript::RuleCommand> parsedCommands;
        auto lhsRows = parseSide(cursor, arrowPos, nullptr);
        auto rhsRows = arrowPos < tokens.size()
            ? parseSide(arrowPos + 1, rhsEnd, &parsedCommands)
            : std::vector<ParsedRow>{};

        auto maskTouchesLayer = [&](const puzzlescript::MaskVector& mask, int32_t layer) -> bool {
            if (layer < 0 || layer >= game->layerCount) {
                return false;
            }
            const auto off = game->layerMaskOffsets[static_cast<size_t>(layer)];
            for (uint32_t w = 0; w < game->wordCount; ++w) {
                const puzzlescript::MaskWord layerWord = game->maskArena[static_cast<size_t>(off + w)];
                if ((mask[static_cast<size_t>(w)] & layerWord) != 0) {
                    return true;
                }
            }
            return false;
        };
        auto trimSuperfluousLhsNegations = [&]() {
            for (auto& row : lhsRows) {
                for (auto& cell : row) {
                    if (cell.isEllipsis) {
                        continue;
                    }
                    auto requiredObjects = makeEmptyMask(game->wordCount);
                    std::vector<int32_t> requiredLayers(static_cast<size_t>(game->layerCount), 0);
                    for (const auto& item : cell.items) {
                        if (item.dir == "no") {
                            continue;
                        }
                        auto addRequiredObject = [&](const std::string& objectName) {
                            const auto objectIt = objectIdByName.find(objectName);
                            if (objectIt == objectIdByName.end()) {
                                return;
                            }
                            setMaskBit(requiredObjects, objectIt->second);
                            const auto& object = game->objectsById[static_cast<size_t>(objectIt->second)];
                            if (object.layer >= 0 && object.layer < game->layerCount) {
                                requiredLayers[static_cast<size_t>(object.layer)] = 1;
                            }
                        };
                        if (objectIdByName.find(item.name) != objectIdByName.end()) {
                            addRequiredObject(item.name);
                        } else if (const auto aggregateIt = aggregateOf.find(item.name); aggregateIt != aggregateOf.end()) {
                            for (const auto& objectName : aggregateIt->second) {
                                addRequiredObject(objectName);
                            }
                        } else if (const auto propertyLayerIt = propertiesSingleLayer.find(item.name);
                                   propertyLayerIt != propertiesSingleLayer.end()) {
                            requiredLayers[static_cast<size_t>(propertyLayerIt->second)] = 1;
                            if (const auto propertyIt = propertyOf.find(item.name); propertyIt != propertyOf.end()) {
                                for (const auto& objectName : propertyIt->second) {
                                    const auto objectIt = objectIdByName.find(objectName);
                                    if (objectIt != objectIdByName.end()) {
                                        setMaskBit(requiredObjects, objectIt->second);
                                    }
                                }
                            }
                        }
                    }

                    std::vector<ParsedItem> trimmed;
                    trimmed.reserve(cell.items.size());
                    for (const auto& item : cell.items) {
                        if (item.dir != "no") {
                            trimmed.push_back(item);
                            continue;
                        }
                        std::set<std::string> visiting;
                        const auto noMask = resolveMask(resolveMask, item.name, visiting);
                        bool disjointObjects = true;
                        for (uint32_t w = 0; w < game->wordCount; ++w) {
                            if ((noMask[static_cast<size_t>(w)] & requiredObjects[static_cast<size_t>(w)]) != 0) {
                                disjointObjects = false;
                                break;
                            }
                        }
                        bool layersCovered = true;
                        for (int32_t layer = 0; layer < game->layerCount; ++layer) {
                            if (maskTouchesLayer(noMask, layer)
                                && requiredLayers[static_cast<size_t>(layer)] == 0) {
                                layersCovered = false;
                                break;
                            }
                        }
                        if (disjointObjects && layersCovered) {
                            continue;
                        }
                        trimmed.push_back(item);
                    }
                    cell.items = std::move(trimmed);
                }
            }
        };
        auto removeRedundantRhsNegations = [&]() {
            const size_t rowCount = std::min(lhsRows.size(), rhsRows.size());
            for (size_t rowIndex = 0; rowIndex < rowCount; ++rowIndex) {
                const size_t cellCount = std::min(lhsRows[rowIndex].size(), rhsRows[rowIndex].size());
                for (size_t cellIndex = 0; cellIndex < cellCount; ++cellIndex) {
                    const auto& lhsCell = lhsRows[rowIndex][cellIndex];
                    auto& rhsCell = rhsRows[rowIndex][cellIndex];
                    if (lhsCell.isEllipsis || rhsCell.isEllipsis) {
                        continue;
                    }
                    for (size_t rhsIndex = 0; rhsIndex < rhsCell.items.size(); ++rhsIndex) {
                        const auto& rhsItem = rhsCell.items[rhsIndex];
                        if (rhsItem.dir == "no") {
                            for (const auto& lhsItem : lhsCell.items) {
                                if (lhsItem.dir == "no" && lhsItem.name == rhsItem.name) {
                                    // JS splices while iterating token pairs, so adjacent
                                    // redundant RHS negations leave every second one behind.
                                    rhsCell.items.erase(rhsCell.items.begin() + static_cast<std::ptrdiff_t>(rhsIndex));
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        };
        removeRedundantRhsNegations();
        trimSuperfluousLhsNegations();

        // JS `processRuleString`: if `directionalRule(rule_line) === false` and
        // `rule_line.directions.length > 1`, only the first direction is kept.
        const auto isDirectionalRule = [](const std::vector<ParsedRow>& lhs, const std::vector<ParsedRow>& rhs) -> bool {
            static const std::set<std::string> kRelativeDirs = {"^", "v", "<", ">", "perpendicular", "parallel"};
            const auto rowsDirectional = [](const std::vector<ParsedRow>& rows, const std::set<std::string>& rel) -> bool {
                for (const auto& row : rows) {
                    if (row.size() > 1) {
                        return true;
                    }
                    for (const auto& cell : row) {
                        if (cell.isEllipsis) {
                            continue;
                        }
                        for (const auto& item : cell.items) {
                            if (rel.find(item.dir) != rel.end()) {
                                return true;
                            }
                        }
                    }
                }
                return false;
            };
            return rowsDirectional(lhs, kRelativeDirs) || rowsDirectional(rhs, kRelativeDirs);
        };
        if (!isDirectionalRule(lhsRows, rhsRows) && ruleDirections.size() > 1) {
            ruleDirections.erase(ruleDirections.begin() + 1, ruleDirections.end());
        }

        {
            int32_t bracketDepth = 0;
            for (size_t i = arrowPos + 1; i < tokens.size(); ++i) {
                if (tokens[i] == "[") {
                    ++bracketDepth;
                    continue;
                }
                if (tokens[i] == "]") {
                    if (bracketDepth > 0) {
                        --bracketDepth;
                    }
                    continue;
                }
                if (bracketDepth != 0 || !isJsBracketPostfixCommand(tokens[i])) {
                    continue;
                }
                puzzlescript::RuleCommand command;
                command.name = tokens[i];
                if (command.name == "message") {
                    std::string message;
                    for (size_t j = i + 1; j < tokens.size(); ++j) {
                        if (!message.empty()) {
                            message.push_back(' ');
                        }
                        if (mixedCaseTokens.size() == tokens.size()) {
                            message.append(mixedCaseTokens[j]);
                        } else {
                            message.append(tokens[j]);
                        }
                    }
                    command.argument = message;
                    parsedCommands.push_back(std::move(command));
                    break;
                }
                parsedCommands.push_back(std::move(command));
            }
        }

        auto absolutizeDir = [](const std::string& forward, const std::string& dir) -> std::string {
            // Match JS `absolutifyRuleCell` + `relativeDict` / `relativeDirs` in compiler.js.
            // `horizontal`, `vertical`, `orthogonal`, `moving`, etc. stay as aggregates
            // for aggregate expansion; ^ v < > parallel perpendicular become concrete/_par/_perp.
            if (dir == ">") return forward;
            if (dir == "<") {
                if (forward == "up") return "down";
                if (forward == "down") return "up";
                if (forward == "left") return "right";
                if (forward == "right") return "left";
            }
            if (dir == "^") {
                if (forward == "up") return "left";
                if (forward == "down") return "right";
                if (forward == "left") return "down";
                if (forward == "right") return "up";
            }
            if (dir == "v") {
                if (forward == "up") return "right";
                if (forward == "down") return "left";
                if (forward == "left") return "up";
                if (forward == "right") return "down";
            }
            if (dir == "parallel") {
                if (forward == "right" || forward == "left") return "horizontal_par";
                if (forward == "up" || forward == "down") return "vertical_par";
            }
            if (dir == "perpendicular") {
                if (forward == "right" || forward == "left") return "vertical_perp";
                if (forward == "up" || forward == "down") return "horizontal_perp";
            }
            return dir;
        };
        auto absolutizeRows = [&](std::vector<ParsedRow>& rows, const std::string& forward) {
            for (auto& row : rows) {
                for (auto& cell : row) {
                    for (auto& item : cell.items) {
                        item.dir = absolutizeDir(forward, item.dir);
                    }
                }
            }
        };
        auto rephraseSynonymsRows = [&](std::vector<ParsedRow>& lhs, std::vector<ParsedRow>& rhs) {
            auto processRows = [&](std::vector<ParsedRow>& rows) {
                for (auto& row : rows) {
                    for (auto& cell : row) {
                        if (cell.isEllipsis) {
                            continue;
                        }
                        for (auto& item : cell.items) {
                            if (const auto it = synonymOf.find(item.name); it != synonymOf.end()) {
                                item.name = it->second;
                            }
                        }
                    }
                }
            };
            processRows(lhs);
            processRows(rhs);
        };
        auto atomizeAggregatesRows = [&](std::vector<ParsedRow>& lhs, std::vector<ParsedRow>& rhs) {
            auto processRows = [&](std::vector<ParsedRow>& rows) {
                for (auto& row : rows) {
                    for (auto& cell : row) {
                        if (cell.isEllipsis) {
                            continue;
                        }
                        for (size_t i = 0; i < cell.items.size(); ++i) {
                            const auto aggregateIt = aggregateOf.find(cell.items[i].name);
                            if (aggregateIt == aggregateOf.end()) {
                                continue;
                            }
                            if (cell.items[i].dir == "no") {
                                throw std::runtime_error(
                                    "Rule at line " + std::to_string(entry.lineNumber)
                                    + " excludes aggregate " + cell.items[i].name + " with 'no', which JS forbids.");
                            }
                            const auto& equivalents = aggregateIt->second;
                            if (equivalents.empty()) {
                                continue;
                            }
                            cell.items[i].name = equivalents.front();
                            for (size_t j = 1; j < equivalents.size(); ++j) {
                                cell.items.push_back({cell.items[i].dir, equivalents[j]});
                            }
                        }
                    }
                }
            };
            processRows(lhs);
            processRows(rhs);
        };
        auto containsEllipsis = [](const std::vector<ParsedRow>& rows) {
            for (const auto& row : rows) {
                for (const auto& cell : row) {
                    if (cell.isEllipsis) {
                        return true;
                    }
                }
            }
            return false;
        };

        // Mirrors JS `concretizeMovingRule` closely, including split order and the
        // two RHS disambiguation passes (`movingReplacement` and
        // `aggregateDirReplacement`). Rule-group order is observable, so the push /
        // erase behavior intentionally follows the JS implementation.
        auto expandConcretizeMovingRows = [&](std::vector<ParsedRow> lhs0, std::vector<ParsedRow> rhs0)
            -> std::vector<std::pair<std::vector<ParsedRow>, std::vector<ParsedRow>>> {
            struct MovingReplacement {
                std::string concreteDirection;
                int occurrenceCount = 1;
                std::string ambiguousMovement;
                std::string attachedObject;
                size_t row = 0;
                size_t cell = 0;
            };
            struct AggregateDirReplacement {
                std::string concreteDirection;
                int occurrenceCount = 1;
                std::string ambiguousMovement;
            };
            struct WorkRule {
                std::vector<ParsedRow> lhs;
                std::vector<ParsedRow> rhs;
                std::map<std::string, MovingReplacement> movingReplacement;
                std::map<std::string, AggregateDirReplacement> aggregateDirReplacement;
            };

            const auto concreteDirsForAggregate = [](const std::string& dir) -> const std::vector<std::string>* {
                static const std::vector<std::string> kHorizDirs = {"left", "right"};
                static const std::vector<std::string> kVertDirs = {"up", "down"};
                static const std::vector<std::string> kMovingDirs = {"up", "down", "left", "right", "action"};
                static const std::vector<std::string> kOrthDirs = {"up", "down", "left", "right"};
                static const std::vector<std::string> kPerpDirs = {"^", "v"};
                static const std::vector<std::string> kParDirs = {"<", ">"};
                if (dir == "horizontal" || dir == "horizontal_par" || dir == "horizontal_perp") return &kHorizDirs;
                if (dir == "vertical" || dir == "vertical_par" || dir == "vertical_perp") return &kVertDirs;
                if (dir == "moving") return &kMovingDirs;
                if (dir == "orthogonal") return &kOrthDirs;
                if (dir == "perpendicular") return &kPerpDirs;
                if (dir == "parallel") return &kParDirs;
                return nullptr;
            };
            auto getMovingsParsed = [&](const ParsedCell& cell) {
                std::vector<std::pair<std::string, std::string>> result;
                if (cell.isEllipsis) {
                    return result;
                }
                for (const auto& item : cell.items) {
                    if (concreteDirsForAggregate(item.dir) != nullptr) {
                        result.push_back({item.name, item.dir});
                    }
                }
                return result;
            };
            auto concretizeMovingInCell = [](ParsedCell& cell,
                                             const std::string& ambiguousMovement,
                                             const std::string& nameToMove,
                                             const std::string& concreteDirection) {
                if (cell.isEllipsis) {
                    return;
                }
                for (auto& item : cell.items) {
                    if (item.dir == ambiguousMovement && item.name == nameToMove) {
                        item.dir = concreteDirection;
                    }
                }
            };
            auto concretizeMovingInCellByAmbiguousMovementName = [](ParsedCell& cell,
                                                                    const std::string& ambiguousMovement,
                                                                    const std::string& concreteDirection) {
                if (cell.isEllipsis) {
                    return;
                }
                for (auto& item : cell.items) {
                    if (item.dir == ambiguousMovement) {
                        item.dir = concreteDirection;
                    }
                }
            };

            std::vector<WorkRule> result;
            result.push_back({std::move(lhs0), std::move(rhs0), {}, {}});

            bool modified = true;
            while (modified) {
                modified = false;
                for (size_t i = 0; i < result.size(); ++i) {
                    bool shouldRemove = false;
                    for (size_t j = 0; j < result[i].lhs.size(); ++j) {
                        auto& currentRuleRow = result[i].lhs[j];
                        for (size_t k = 0; k < currentRuleRow.size(); ++k) {
                            const auto movings = getMovingsParsed(currentRuleRow[k]);
                            if (movings.empty()) {
                                continue;
                            }

                            shouldRemove = true;
                            modified = true;
                            const std::string& candName = movings[0].first;
                            const std::string& ambiguousDir = movings[0].second;
                            const auto* concreteDirs = concreteDirsForAggregate(ambiguousDir);
                            if (concreteDirs == nullptr) {
                                continue;
                            }

                            const WorkRule baseRule = result[i];
                            for (const auto& concreteDirection : *concreteDirs) {
                                WorkRule newRule = baseRule;
                                concretizeMovingInCell(newRule.lhs[j][k], ambiguousDir, candName, concreteDirection);
                                if (!newRule.rhs.empty() && j < newRule.rhs.size() && k < newRule.rhs[j].size()) {
                                    concretizeMovingInCell(newRule.rhs[j][k], ambiguousDir, candName, concreteDirection);
                                }

                                const std::string movingKey = candName + ambiguousDir;
                                auto movingIt = newRule.movingReplacement.find(movingKey);
                                if (movingIt == newRule.movingReplacement.end()) {
                                    newRule.movingReplacement[movingKey] =
                                        MovingReplacement{concreteDirection, 1, ambiguousDir, candName, j, k};
                                } else if (j != movingIt->second.row || k != movingIt->second.cell) {
                                    movingIt->second.occurrenceCount += 1;
                                }

                                auto aggregateIt = newRule.aggregateDirReplacement.find(ambiguousDir);
                                if (aggregateIt == newRule.aggregateDirReplacement.end()) {
                                    newRule.aggregateDirReplacement[ambiguousDir] =
                                        AggregateDirReplacement{concreteDirection, 1, ambiguousDir};
                                } else {
                                    aggregateIt->second.occurrenceCount += 1;
                                }

                                result.push_back(std::move(newRule));
                            }
                        }
                    }
                    if (shouldRemove) {
                        result.erase(result.begin() + static_cast<std::ptrdiff_t>(i));
                        --i;
                    }
                }
            }

            std::vector<std::pair<std::vector<ParsedRow>, std::vector<ParsedRow>>> out;
            out.reserve(result.size());
            for (auto& currentRule : result) {
                for (const auto& [_, replacementInfo] : currentRule.movingReplacement) {
                    if (replacementInfo.occurrenceCount != 1) {
                        continue;
                    }
                    for (auto& rhsRow : currentRule.rhs) {
                        for (auto& cell : rhsRow) {
                            concretizeMovingInCell(
                                cell,
                                replacementInfo.ambiguousMovement,
                                replacementInfo.attachedObject,
                                replacementInfo.concreteDirection
                            );
                        }
                    }
                }

                std::map<std::string, std::string> ambiguousMovementNames;
                for (const auto& [_, replacementInfo] : currentRule.aggregateDirReplacement) {
                    const auto existing = ambiguousMovementNames.find(replacementInfo.ambiguousMovement);
                    if (existing != ambiguousMovementNames.end() || replacementInfo.occurrenceCount != 1) {
                        ambiguousMovementNames[replacementInfo.ambiguousMovement] = "INVALID";
                    } else {
                        ambiguousMovementNames[replacementInfo.ambiguousMovement] = replacementInfo.concreteDirection;
                    }
                }
                for (const auto& [ambiguousMovement, concreteMovement] : ambiguousMovementNames) {
                    if (concreteMovement == "INVALID") {
                        continue;
                    }
                    for (auto& rhsRow : currentRule.rhs) {
                        for (auto& cell : rhsRow) {
                            concretizeMovingInCellByAmbiguousMovementName(cell, ambiguousMovement, concreteMovement);
                        }
                    }
                }

                std::string rhsAmbiguousMovementRemains;
                for (const auto& rhsRow : currentRule.rhs) {
                    for (const auto& cell : rhsRow) {
                        const auto movings = getMovingsParsed(cell);
                        if (!movings.empty()) {
                            rhsAmbiguousMovementRemains = movings[0].second;
                            break;
                        }
                    }
                    if (!rhsAmbiguousMovementRemains.empty()) {
                        break;
                    }
                }
                if (!rhsAmbiguousMovementRemains.empty()) {
                    throw std::runtime_error(
                        "Rule at line " + std::to_string(entry.lineNumber)
                        + " has an ambiguous movement on the right-hand side, \"" + rhsAmbiguousMovementRemains
                        + "\", that can't be inferred from the left-hand side.");
                }

                out.push_back({std::move(currentRule.lhs), std::move(currentRule.rhs)});
            }
            return out;
        };

        // Mirrors src/js/compiler.js `concretizePropertyRule` for ParsedRow/cell form:
        // expandNoPrefixedProperties, ambiguousProperties (RHS vs LHS), per-cell
        // property explosion with propertyReplacement bookkeeping, then RHS cleanup
        // when a property was concretized exactly once on the LHS.
        auto concretizePropertyInCell = [](ParsedCell& cell, const std::string& property, const std::string& concreteType) {
            if (cell.isEllipsis) {
                return;
            }
            for (auto& it : cell.items) {
                if (it.dir != "random" && it.name == property) {
                    it.name = concreteType;
                }
            }
        };
        auto getPropertiesFromCellParsed = [&](const ParsedCell& cell) -> std::vector<std::string> {
            std::vector<std::string> out;
            if (cell.isEllipsis) {
                return out;
            }
            for (const auto& it : cell.items) {
                if (it.dir == "random") {
                    continue;
                }
                if (propertyOf.find(it.name) != propertyOf.end()) {
                    out.push_back(it.name);
                }
            }
            return out;
        };
        auto expandNoPrefixedCell = [&](ParsedCell& cell) {
            if (cell.isEllipsis) {
                return;
            }
            std::vector<ParsedItem> expanded;
            expanded.reserve(cell.items.size() * 2);
            for (const auto& it : cell.items) {
                if (it.dir == "no" && propertyOf.find(it.name) != propertyOf.end()) {
                    for (const auto& alias : propertyOf.at(it.name)) {
                        expanded.push_back({"no", alias});
                    }
                } else {
                    expanded.push_back(it);
                }
            }
            cell.items = std::move(expanded);
        };
        auto expandNoPrefixedRows = [&](std::vector<ParsedRow>& lhs, std::vector<ParsedRow>& rhs) {
            for (size_t ri = 0; ri < lhs.size(); ++ri) {
                auto& lhsRow = lhs[ri];
                for (size_t ci = 0; ci < lhsRow.size(); ++ci) {
                    expandNoPrefixedCell(lhsRow[ci]);
                    if (ri < rhs.size() && ci < rhs[ri].size()) {
                        expandNoPrefixedCell(rhs[ri][ci]);
                    }
                }
            }
        };
        auto buildAmbiguousPropertiesSet = [&](const std::vector<ParsedRow>& lhs, const std::vector<ParsedRow>& rhs) {
            std::set<std::string> ambiguous;
            const size_t nRows = std::min(lhs.size(), rhs.size());
            for (size_t j = 0; j < nRows; ++j) {
                const auto& rowL = lhs[j];
                const auto& rowR = rhs[j];
                const size_t nCols = std::min(rowL.size(), rowR.size());
                for (size_t k = 0; k < nCols; ++k) {
                    const auto propsL = getPropertiesFromCellParsed(rowL[k]);
                    const std::set<std::string> setL(propsL.begin(), propsL.end());
                    for (const std::string& p : getPropertiesFromCellParsed(rowR[k])) {
                        if (setL.find(p) == setL.end()) {
                            ambiguous.insert(p);
                        }
                    }
                }
            }
            return ambiguous;
        };
        auto expandConcretizePropertyRows = [&](std::vector<ParsedRow> lhs0, std::vector<ParsedRow> rhs0)
            -> std::vector<std::pair<std::vector<ParsedRow>, std::vector<ParsedRow>>> {
            struct Work {
                std::vector<ParsedRow> lhs;
                std::vector<ParsedRow> rhs;
                std::map<std::string, std::pair<std::string, int>> propRepl;
            };
            std::vector<Work> work;
            work.push_back({std::move(lhs0), std::move(rhs0), {}});
            expandNoPrefixedRows(work.front().lhs, work.front().rhs);
            // JS freezes `ambiguousProperties` after no-prefix expand; it is not
            // recomputed as concrete names appear on the LHS during splitting.
            const std::set<std::string> ambiguousInitial =
                buildAmbiguousPropertiesSet(work.front().lhs, work.front().rhs);

            bool modified = true;
            while (modified) {
                modified = false;
                for (size_t i = 0; i < work.size(); ++i) {
                    size_t splitJ = 0;
                    size_t splitK = 0;
                    std::string splitProperty;
                    bool found = false;
                    for (size_t j = 0; j < work[i].lhs.size() && !found; ++j) {
                        for (size_t k = 0; k < work[i].lhs[j].size() && !found; ++k) {
                            for (const std::string& property : getPropertiesFromCellParsed(work[i].lhs[j][k])) {
                                if (propertiesSingleLayer.find(property) != propertiesSingleLayer.end()
                                    && ambiguousInitial.find(property) == ambiguousInitial.end()) {
                                    continue;
                                }
                                if (propertyOf.find(property) == propertyOf.end()) {
                                    continue;
                                }
                                splitJ = j;
                                splitK = k;
                                splitProperty = property;
                                found = true;
                                break;
                            }
                        }
                    }
                    if (!found) {
                        continue;
                    }

                    const Work base = work[i];
                    work.erase(work.begin() + static_cast<std::ptrdiff_t>(i));
                    const std::vector<std::string>& aliases = propertyOf.at(splitProperty);
                    std::vector<Work> newOnes;
                    newOnes.reserve(aliases.size());
                    for (const std::string& concreteType : aliases) {
                        Work nw = base;
                        concretizePropertyInCell(nw.lhs[splitJ][splitK], splitProperty, concreteType);
                        if (!nw.rhs.empty() && splitJ < nw.rhs.size() && splitK < nw.rhs[splitJ].size()) {
                            concretizePropertyInCell(nw.rhs[splitJ][splitK], splitProperty, concreteType);
                        }
                        const auto repIt = nw.propRepl.find(splitProperty);
                        if (repIt == nw.propRepl.end()) {
                            nw.propRepl[splitProperty] = {concreteType, 1};
                        } else {
                            repIt->second.second += 1;
                        }
                        newOnes.push_back(std::move(nw));
                    }
                    work.insert(
                        work.begin() + static_cast<std::ptrdiff_t>(i),
                        std::make_move_iterator(newOnes.begin()),
                        std::make_move_iterator(newOnes.end()));
                    modified = true;
                    break;
                }
            }

            std::vector<std::pair<std::vector<ParsedRow>, std::vector<ParsedRow>>> out;
            out.reserve(work.size());
            for (Work& w : work) {
                for (const auto& [prop, info] : w.propRepl) {
                    if (info.second != 1) {
                        continue;
                    }
                    const std::string& concreteType = info.first;
                    for (auto& row : w.rhs) {
                        for (auto& cell : row) {
                            concretizePropertyInCell(cell, prop, concreteType);
                        }
                    }
                }

                std::string rhsPropertyRemains;
                for (const auto& row : w.rhs) {
                    for (const auto& cell : row) {
                        for (const std::string& p : getPropertiesFromCellParsed(cell)) {
                            if (ambiguousInitial.find(p) != ambiguousInitial.end()) {
                                rhsPropertyRemains = p;
                            }
                        }
                    }
                }
                if (!rhsPropertyRemains.empty()) {
                    throw std::runtime_error(
                        "Rule at line " + std::to_string(entry.lineNumber)
                        + " has a property on the right-hand side, \"" + rhsPropertyRemains
                        + "\", that can't be inferred from the left-hand side.");
                }
                out.push_back({std::move(w.lhs), std::move(w.rhs)});
            }
            return out;
        };

        // JS `makeSpawnedObjectsStationary` (compiler.js): after moving/property
        // concretization, any RHS object token with an empty direction prefix whose
        // collision layer is not represented among possible LHS objects in the
        // aligned cell gets `stationary`, so old movement bits clear (#492).
        auto getPossibleObjectNamesFromParsedCell = [&](const ParsedCell& cell) -> std::vector<std::string> {
            std::vector<std::string> out;
            if (cell.isEllipsis) {
                return out;
            }
            for (const auto& it : cell.items) {
                if (it.dir == "random") {
                    continue;
                }
                const std::string nameLower = toLowerAsciiCopy(it.name);
                const auto propIt = propertyOf.find(nameLower);
                if (propIt != propertyOf.end()) {
                    for (const auto& al : propIt->second) {
                        out.push_back(al);
                    }
                } else if (objectIdByName.find(nameLower) != objectIdByName.end()) {
                    out.push_back(nameLower);
                }
            }
            return out;
        };
        auto objectLayerByLowerName = [&](const std::string& nameLower) -> std::optional<int32_t> {
            const auto it = objectIdByName.find(nameLower);
            if (it == objectIdByName.end()) {
                return std::nullopt;
            }
            const int32_t layer = game->objectsById[static_cast<size_t>(it->second)].layer;
            if (layer < 0) {
                return std::nullopt;
            }
            return layer;
        };
        auto makeSpawnedObjectsStationaryRows = [&](std::vector<ParsedRow>& lhs, std::vector<ParsedRow>& rhs) {
            const size_t nRows = std::min(lhs.size(), rhs.size());
            for (size_t j = 0; j < nRows; ++j) {
                const ParsedRow& rowL = lhs[j];
                ParsedRow& rowR = rhs[j];
                const size_t nCols = std::min(rowL.size(), rowR.size());
                for (size_t k = 0; k < nCols; ++k) {
                    const ParsedCell& cellL = rowL[k];
                    ParsedCell& cellR = rowR[k];
                    if (cellR.isEllipsis) {
                        continue;
                    }
                    const std::vector<std::string> possible = getPossibleObjectNamesFromParsedCell(cellL);
                    std::vector<int32_t> lhsLayers;
                    lhsLayers.reserve(possible.size());
                    for (const auto& name : possible) {
                        if (const auto layer = objectLayerByLowerName(name)) {
                            lhsLayers.push_back(*layer);
                        }
                    }
                    for (auto& it : cellR.items) {
                        if (it.dir == "random" || !it.dir.empty()) {
                            continue;
                        }
                        const std::string nameLower = toLowerAsciiCopy(it.name);
                        if (propertyOf.find(nameLower) != propertyOf.end()) {
                            continue;
                        }
                        if (std::find(possible.begin(), possible.end(), nameLower) != possible.end()) {
                            continue;
                        }
                        const auto rLayer = objectLayerByLowerName(nameLower);
                        if (!rLayer.has_value()) {
                            continue;
                        }
                        if (std::find(lhsLayers.begin(), lhsLayers.end(), *rLayer) != lhsLayers.end()) {
                            continue;
                        }
                        it.dir = "stationary";
                    }
                }
            }
        };

        auto appendParsedRowsToSignature = [](std::string& sig, const std::vector<ParsedRow>& rows) {
            for (const auto& row : rows) {
                sig.push_back('[');
                for (const auto& cell : row) {
                    sig.push_back('{');
                    if (cell.isEllipsis) {
                        sig.append("...");
                    } else {
                        for (const auto& item : cell.items) {
                            if (!item.dir.empty()) {
                                sig.append(item.dir);
                                sig.push_back(' ');
                            }
                            sig.append(item.name);
                            sig.push_back(',');
                        }
                    }
                    sig.push_back('}');
                }
                sig.push_back(']');
            }
        };
        auto ruleVariantSignature = [&](int32_t lineNumber,
                                       const std::string& forward,
                                       bool rigid,
                                       bool random,
                                       bool late,
                                       const std::vector<ParsedRow>& lhs,
                                       const std::vector<ParsedRow>& rhs,
                                       const std::vector<puzzlescript::RuleCommand>& commands) {
            std::string sig;
            sig.reserve(256);
            sig.append(std::to_string(lineNumber));
            sig.push_back('|');
            bool directed = false;
            for (const auto& row : lhs) {
                if (row.size() > 1) {
                    directed = true;
                    break;
                }
            }
            if (rigid) sig.append("RIGID|");
            if (random) sig.append("RANDOM|");
            if (late) sig.append("LATE|");
            if (directed) {
                sig.append(forward);
                sig.push_back('|');
            }
            appendParsedRowsToSignature(sig, lhs);
            sig.push_back('|');
            appendParsedRowsToSignature(sig, rhs);
            sig.push_back('|');
            for (const auto& command : commands) {
                sig.append(command.name);
                sig.push_back(':');
                if (command.argument.has_value()) {
                    sig.append(*command.argument);
                }
                sig.push_back('|');
            }
            return sig;
        };
        auto lhsHasLayerOverlap = [&](const std::vector<ParsedRow>& rows) {
            for (const auto& row : rows) {
                for (const auto& cell : row) {
                    if (cell.isEllipsis) {
                        continue;
                    }
                    std::vector<uint8_t> usedLayers(static_cast<size_t>(game->layerCount), 0);
                    for (const auto& item : cell.items) {
                        if (item.dir == "no" || item.dir == "random") {
                            continue;
                        }
                        std::set<std::string> visiting;
                        const auto mask = resolveMask(resolveMask, item.name, visiting);
                        const auto layer = maskSingleLayer(mask);
                        if (!layer.has_value() || *layer < 0 || *layer >= game->layerCount) {
                            continue;
                        }
                        auto& used = usedLayers[static_cast<size_t>(*layer)];
                        if (used != 0) {
                            return true;
                        }
                        used = 1;
                    }
                }
            }
            return false;
        };

        for (const auto& rawRuleDirection : ruleDirections) {
        auto variantLhsRows = lhsRows;
        auto variantRhsRows = rhsRows;
        std::string concreteRuleDirection = rawRuleDirection;
        absolutizeRows(variantLhsRows, concreteRuleDirection);
        absolutizeRows(variantRhsRows, concreteRuleDirection);
        if (!containsEllipsis(variantLhsRows)) {
            if (concreteRuleDirection == "up") {
                concreteRuleDirection = "down";
                for (auto& row : variantLhsRows) std::reverse(row.begin(), row.end());
                for (auto& row : variantRhsRows) std::reverse(row.begin(), row.end());
            } else if (concreteRuleDirection == "left") {
                concreteRuleDirection = "right";
                for (auto& row : variantLhsRows) std::reverse(row.begin(), row.end());
                for (auto& row : variantRhsRows) std::reverse(row.begin(), row.end());
            }
        }
        atomizeAggregatesRows(variantLhsRows, variantRhsRows);
        rephraseSynonymsRows(variantLhsRows, variantRhsRows);

        const auto movingVariants = expandConcretizeMovingRows(std::move(variantLhsRows), std::move(variantRhsRows));
        for (const auto& movingVariant : movingVariants) {
            const auto propertyConcreteChunks =
                expandConcretizePropertyRows(movingVariant.first, movingVariant.second);
            for (const auto& propChunk : propertyConcreteChunks) {
                std::vector<ParsedRow> variantLhsRowsExpanded = propChunk.first;
            std::vector<ParsedRow> variantRhsRowsExpanded = propChunk.second;
            if (lhsHasLayerOverlap(variantLhsRowsExpanded)) {
                continue;
            }
            if (!lateRule) {
                makeSpawnedObjectsStationaryRows(variantLhsRowsExpanded, variantRhsRowsExpanded);
            }

        puzzlescript::Rule rule;
        rule.direction = dirMaskFromToken(concreteRuleDirection);
        rule.lineNumber = entry.lineNumber;
        if (sameGroup && outputGroup != nullptr && !outputGroup->empty()) {
            rule.groupNumber = outputGroup->front().groupNumber;
        } else {
            rule.groupNumber = entry.lineNumber;
        }
        rule.rigid = rigidRule;
        rule.isRandom = randomRule;
        rule.hasReplacements = !variantRhsRowsExpanded.empty();
        rule.commands = parsedCommands;

        auto buildPatternRow = [&](const ParsedRow& row, const ParsedRow* rhsRow) -> std::vector<puzzlescript::Pattern> {
            std::vector<puzzlescript::Pattern> out;
            out.reserve(row.size());
            for (size_t cellIndex = 0; cellIndex < row.size(); ++cellIndex) {
                const ParsedCell& cell = row[cellIndex];
                if (cell.isEllipsis) {
                    puzzlescript::Pattern pat;
                    pat.kind = puzzlescript::Pattern::Kind::Ellipsis;
                    out.push_back(std::move(pat));
                    continue;
                }

                auto objectsPresent = makeEmptyMask(game->wordCount);
                auto objectsMissing = makeEmptyMask(game->wordCount);
                auto movementsPresent = puzzlescript::MaskVector(static_cast<size_t>(game->movementWordCount), 0);
                auto movementsMissing = puzzlescript::MaskVector(static_cast<size_t>(game->movementWordCount), 0);
                std::vector<puzzlescript::MaskOffset> anyOffsets;
                std::vector<std::vector<int32_t>> anyAnchorIds;

                // Per-layer occupancy names (JS `layersUsed_l`): any LHS token with a
                // resolved single layer, including properties.
                std::vector<int32_t> layersUsedL(game->layerCount, 0);
                // Movement-bitvec lanes where LHS had a *concrete* object (JS `objectlayers_l`).
                puzzlescript::MaskVector lhsObjectLayersMovement(static_cast<size_t>(game->movementWordCount), 0);
                for (const auto& item : cell.items) {
                    if (item.dir == "random") {
                        continue; // handled on RHS replacement
                    }
                    std::set<std::string> visiting;
                    const auto mask = resolveMask(resolveMask, item.name, visiting);
                    const auto singleLayer = maskSingleLayer(mask);
                    const bool isProperty = propertyOf.find(item.name) != propertyOf.end();

                    if (item.dir == "no") {
                        for (size_t w = 0; w < objectsMissing.size(); ++w) {
                            objectsMissing[w] |= mask[w];
                        }
                        continue;
                    }

                    if (isProperty) {
                        // JS semantics: OR properties become anyObjectsPresent even
                        // when they live on a single collision layer.
                        const auto off = storeMaskWords(*game, mask);
                        game->anyObjectOffsets.push_back(off);
                        anyOffsets.push_back(off);
                        anyAnchorIds.push_back(objectIdsFromMask(mask, game->objectCount));
                        if (singleLayer.has_value()) {
                            layersUsedL[static_cast<size_t>(*singleLayer)] = 1;
                        }
                    } else if (singleLayer.has_value()) {
                        for (size_t w = 0; w < objectsPresent.size(); ++w) {
                            objectsPresent[w] |= mask[w];
                        }
                        layersUsedL[static_cast<size_t>(*singleLayer)] = 1;
                        orShiftedMask5(lhsObjectLayersMovement, 5 * (*singleLayer), 0x1f);
                    } else {
                        const auto off = storeMaskWords(*game, mask);
                        game->anyObjectOffsets.push_back(off);
                        anyOffsets.push_back(off);
                        anyAnchorIds.push_back(objectIdsFromMask(mask, game->objectCount));
                    }

                    if (singleLayer.has_value()) {
                        const int32_t layer = *singleLayer;
                        if (item.dir == "stationary") {
                            orShiftedMask5(movementsMissing, 5 * layer, 0x1f);
                        } else if (!item.dir.empty()) {
                            const int32_t dm = dirMaskFromToken(item.dir);
                            if (dm != 0) {
                                orShiftedMask5(movementsPresent, 5 * layer, dm);
                            }
                        }
                    }
                }

                puzzlescript::Pattern pat;
                pat.kind = puzzlescript::Pattern::Kind::CellPattern;
                auto maskHasAnyBit = [](const puzzlescript::MaskVector& words) {
                    return std::any_of(words.begin(), words.end(), [](puzzlescript::MaskWord word) { return word != 0; });
                };
                pat.hasObjectsPresent = maskHasAnyBit(objectsPresent);
                pat.hasObjectsMissing = maskHasAnyBit(objectsMissing);
                pat.hasMovementsPresent = maskHasAnyBit(movementsPresent);
                pat.hasMovementsMissing = maskHasAnyBit(movementsMissing);
                pat.objectsPresent = storeMaskWords(*game, objectsPresent);
                pat.objectsMissing = storeMaskWords(*game, objectsMissing);
                pat.movementsPresent = storeMaskWords(*game, movementsPresent);
                pat.movementsMissing = storeMaskWords(*game, movementsMissing);
                pat.objectAnchorIds = objectIdsFromMask(objectsPresent, game->objectCount);
                pat.anyObjectsFirst = static_cast<uint32_t>(game->anyObjectOffsets.size() - anyOffsets.size());
                pat.anyObjectsCount = static_cast<uint32_t>(anyOffsets.size());
                pat.anyObjectAnchorIds = std::move(anyAnchorIds);

                if (rhsRow && cellIndex < rhsRow->size()) {
                    const ParsedCell& rhsCell = (*rhsRow)[cellIndex];
                    if (!rhsCell.isEllipsis) {
                        // Compute clear/set per layer for RHS. Best-effort:
                        // clear all layers used by either lhs cell single-layer masks
                        // or rhs cell single-layer masks, then set rhs objects.
                        auto objectsClear = makeEmptyMask(game->wordCount);
                        auto objectsSet = makeEmptyMask(game->wordCount);
                        auto movementsClear = puzzlescript::MaskVector(static_cast<size_t>(game->movementWordCount), 0);
                        auto movementsSet = puzzlescript::MaskVector(static_cast<size_t>(game->movementWordCount), 0);
                        auto movementsLayerMask = puzzlescript::MaskVector(static_cast<size_t>(game->movementWordCount), 0);
                        auto randomEntityMask = puzzlescript::MaskVector(static_cast<size_t>(game->wordCount), 0);
                        auto randomDirMask = puzzlescript::MaskVector(static_cast<size_t>(game->movementWordCount), 0);
                        std::vector<int32_t> layersUsedR(game->layerCount, 0);
                        puzzlescript::MaskVector rhsObjectLayersMovement(static_cast<size_t>(game->movementWordCount), 0);

                        auto markLayerClear = [&](int32_t layer) {
                            if (layer < 0 || layer >= game->layerCount) return;
                            const auto off = game->layerMaskOffsets[static_cast<size_t>(layer)];
                            for (uint32_t w = 0; w < game->wordCount; ++w) {
                                objectsClear[static_cast<size_t>(w)] |= game->maskArena[static_cast<size_t>(off + w)];
                            }
                            // JS semantics: mark movement layers that should be reset.
                            orShiftedMask5(movementsLayerMask, 5 * layer, 0x1f);
                        };

                        auto orLayerMaskToObjectsClear = [&](int32_t layer) {
                            if (layer < 0 || layer >= game->layerCount) return;
                            const auto off = game->layerMaskOffsets[static_cast<size_t>(layer)];
                            for (uint32_t w = 0; w < game->wordCount; ++w) {
                                objectsClear[static_cast<size_t>(w)] |= game->maskArena[static_cast<size_t>(off + w)];
                            }
                        };

                        // Only clear object layers if the RHS actually writes objects
                        // (i.e. concrete objects or explicit deletes). For property
                        // rules like Moveable -> Moveable, JS leaves objects_clear/set empty.
                        bool rhsWritesObjects = rhsCell.items.empty(); // empty cell => clear
                        if (!rhsWritesObjects) {
                            for (const auto& rhsItem : rhsCell.items) {
                                if (rhsItem.dir == "random") {
                                    rhsWritesObjects = true;
                                    break;
                                }
                                if (rhsItem.dir == "random") {
                                    continue;
                                }
                                if (rhsItem.dir == "no") {
                                    rhsWritesObjects = true;
                                    break;
                                }
                                if (propertyOf.find(rhsItem.name) == propertyOf.end()) {
                                    rhsWritesObjects = true;
                                    break;
                                }
                            }
                        }

                        for (const auto& item : rhsCell.items) {
                            if (item.dir == "random") {
                                std::set<std::string> visiting;
                                const auto mask = resolveMask(resolveMask, item.name, visiting);
                                for (size_t w = 0; w < randomEntityMask.size() && w < mask.size(); ++w) {
                                    randomEntityMask[static_cast<size_t>(w)] |= mask[static_cast<size_t>(w)];
                                }
                                continue;
                            }
                            // JS `dirMasks.randomdir` === parseInt('00101', 2) === 5; OR'd into randomDirMask_r
                            // at STRIDE_5 * layerIndex (compiler.js rulesToMask).
                            if (item.dir == "randomdir") {
                                std::set<std::string> visiting;
                                const auto mask = resolveMask(resolveMask, item.name, visiting);
                                const auto singleLayer = maskSingleLayer(mask);
                                const bool isProperty = propertyOf.find(item.name) != propertyOf.end();
                                if (singleLayer.has_value()) {
                                    layersUsedR[static_cast<size_t>(*singleLayer)] = 1;
                                }
                                if (!isProperty) {
                                    auto oneMask = makeEmptyMask(game->wordCount);
                                    std::set<std::string> rhsVisiting;
                                    const auto resolved = resolveMask(resolveMask, item.name, rhsVisiting);
                                    for (int32_t id = 0; id < game->objectCount; ++id) {
                                        if (maskHasBit(resolved, id)) {
                                            setMaskBit(oneMask, id);
                                            break;
                                        }
                                    }
                                    for (size_t w = 0; w < objectsSet.size(); ++w) {
                                        objectsSet[static_cast<size_t>(w)] |= oneMask[static_cast<size_t>(w)];
                                    }
                                    if (rhsWritesObjects && singleLayer.has_value()) {
                                        const int32_t layer = *singleLayer;
                                        orLayerMaskToObjectsClear(layer);
                                        orShiftedMask5(rhsObjectLayersMovement, 5 * layer, 0x1f);
                                    }
                                }
                                if (singleLayer.has_value()) {
                                    const int32_t layer = *singleLayer;
                                    orShiftedMask5(movementsLayerMask, 5 * layer, 0x1f);
                                    orShiftedMask5(randomDirMask, 5 * layer, 5);
                                }
                                continue;
                            }
                            std::set<std::string> visiting;
                            const auto mask = resolveMask(resolveMask, item.name, visiting);
                            const auto singleLayer = maskSingleLayer(mask);
                            const bool isProperty = propertyOf.find(item.name) != propertyOf.end();

                            if (item.dir == "no") {
                                // Explicit delete.
                                for (size_t w = 0; w < objectsClear.size(); ++w) {
                                    objectsClear[w] |= mask[w];
                                }
                                continue;
                            }

                            if (singleLayer.has_value()) {
                                layersUsedR[static_cast<size_t>(*singleLayer)] = 1;
                            }

                            if (!isProperty) {
                                // Concrete objects: set only the first id represented
                                // by this token (handles legend aliases like 1/2/3/4).
                                auto oneMask = makeEmptyMask(game->wordCount);
                                std::set<std::string> rhsVisiting;
                                const auto resolved = resolveMask(resolveMask, item.name, rhsVisiting);
                                for (int32_t id = 0; id < game->objectCount; ++id) {
                                    if (maskHasBit(resolved, id)) {
                                        setMaskBit(oneMask, id);
                                        break;
                                    }
                                }
                                for (size_t w = 0; w < objectsSet.size(); ++w) {
                                    objectsSet[w] |= oneMask[w];
                                }
                                if (rhsWritesObjects && singleLayer.has_value()) {
                                    const int32_t layer = *singleLayer;
                                    orLayerMaskToObjectsClear(layer);
                                    orShiftedMask5(rhsObjectLayersMovement, 5 * layer, 0x1f);
                                }
                            }
                            if (singleLayer.has_value()) {
                                const int32_t layer = *singleLayer;
                                // JS: any non-empty direction on RHS sets postMovementsLayerMask first.
                                if (!item.dir.empty() && item.dir != "no") {
                                    orShiftedMask5(movementsLayerMask, 5 * layer, 0x1f);
                                }
                                if (item.dir == "stationary") {
                                    orShiftedMask5(movementsClear, 5 * layer, 0x1f);
                                } else if (!item.dir.empty() && item.dir != "no") {
                                    const int32_t dm = dirMaskFromToken(item.dir);
                                    if (dm != 0) {
                                        orShiftedMask5(movementsSet, 5 * layer, dm);
                                    }
                                }
                            }
                        }

                        // JS rulesToMask: if RHS objectsSet doesn't cover LHS objectsPresent,
                        // OR LHS objectsPresent into objectsClear.
                        {
                            bool lhsCovered = true;
                            for (uint32_t w = 0; w < game->wordCount; ++w) {
                                const puzzlescript::MaskWord pres = objectsPresent[static_cast<size_t>(w)];
                                const puzzlescript::MaskWord setv = objectsSet[static_cast<size_t>(w)];
                                if ((pres & setv) != pres) {
                                    lhsCovered = false;
                                    break;
                                }
                            }
                            if (!lhsCovered) {
                                for (uint32_t w = 0; w < game->wordCount; ++w) {
                                    objectsClear[static_cast<size_t>(w)] |= objectsPresent[static_cast<size_t>(w)];
                                }
                            }
                        }
                        // Same for movementsPresent vs movementsSet.
                        {
                            bool movCovered = true;
                            for (uint32_t w = 0; w < game->movementWordCount; ++w) {
                                const puzzlescript::MaskWord pres = movementsPresent[static_cast<size_t>(w)];
                                const puzzlescript::MaskWord setv = movementsSet[static_cast<size_t>(w)];
                                if ((pres & setv) != pres) {
                                    movCovered = false;
                                    break;
                                }
                            }
                            if (!movCovered) {
                                for (uint32_t w = 0; w < game->movementWordCount; ++w) {
                                    movementsClear[static_cast<size_t>(w)] |= movementsPresent[static_cast<size_t>(w)];
                                }
                            }
                        }

                        // JS rulesToMask always clears layers mentioned on the LHS
                        // when the corresponding RHS cell omits that layer.
                        for (int32_t layer = 0; layer < game->layerCount; ++layer) {
                            if (layersUsedL[static_cast<size_t>(layer)] != 0
                                && layersUsedR[static_cast<size_t>(layer)] == 0) {
                                markLayerClear(layer);
                            }
                        }

                        // JS: postMovementsLayerMask |= (objectlayers_l & ~objectlayers_r)
                        {
                            puzzlescript::MaskVector residual = lhsObjectLayersMovement;
                            for (size_t w = 0; w < residual.size(); ++w) {
                                residual[static_cast<size_t>(w)] &= ~rhsObjectLayersMovement[static_cast<size_t>(w)];
                            }
                            for (size_t w = 0; w < movementsLayerMask.size(); ++w) {
                                movementsLayerMask[static_cast<size_t>(w)] |= residual[static_cast<size_t>(w)];
                            }
                        }

                        auto randomEntityTouchesLayer = [&](int32_t layer) {
                            if (layer < 0 || layer >= game->layerCount) {
                                return false;
                            }
                            const auto off = game->layerMaskOffsets[static_cast<size_t>(layer)];
                            for (uint32_t w = 0; w < game->wordCount; ++w) {
                                const puzzlescript::MaskWord layerWord = game->maskArena[static_cast<size_t>(off + w)];
                                if ((randomEntityMask[static_cast<size_t>(w)] & layerWord) != 0) {
                                    return true;
                                }
                            }
                            return false;
                        };

                        // If we set a concrete object on a layer, JS does not
                        // clear that layer's movement bits implicitly.
                        if (rhsWritesObjects) {
                            for (int32_t layer = 0; layer < game->layerCount; ++layer) {
                                const auto off = game->layerMaskOffsets[static_cast<size_t>(layer)];
                                bool overlaps = false;
                                for (uint32_t w = 0; w < game->wordCount; ++w) {
                                    const puzzlescript::MaskWord layerWord = game->maskArena[static_cast<size_t>(off + w)];
                                    if ((objectsSet[static_cast<size_t>(w)] & layerWord) != 0) {
                                        overlaps = true;
                                        break;
                                    }
                                }
                                if (overlaps) {
                                    const int32_t moveSetBits = getShiftedMask5(movementsSet, 5 * layer);
                                    const int32_t moveClearBits = getShiftedMask5(movementsClear, 5 * layer);
                                    const int32_t randomDirBits = getShiftedMask5(randomDirMask, 5 * layer);
                                    const bool randomEntOnLayer = randomEntityTouchesLayer(layer);
                                    // Only suppress implicit layer-mask clearing when there is no explicit
                                    // movement directive on the layer (including randomDir / random entity).
                                    if (moveSetBits == 0 && moveClearBits == 0 && randomDirBits == 0 && !randomEntOnLayer) {
                                        setShiftedMask5(movementsLayerMask, 5 * layer, 0);
                                    }
                                }
                            }
                        }

                        auto anyNonZero = [](const puzzlescript::MaskVector& words) {
                            for (const puzzlescript::MaskWord w : words) {
                                if (w != 0) return true;
                            }
                            return false;
                        };

                        puzzlescript::Replacement repl;
                        repl.objectsClear = storeMaskWords(*game, objectsClear);
                        repl.objectsSet = storeMaskWords(*game, objectsSet);
                        repl.movementsClear = storeMaskWords(*game, movementsClear);
                        repl.movementsSet = storeMaskWords(*game, movementsSet);
                        repl.movementsLayerMask = storeMaskWords(*game, movementsLayerMask);
                        repl.hasMovementsLayerMask = anyNonZero(movementsLayerMask);
                        if (anyNonZero(randomEntityMask)) {
                            repl.randomEntityMask = storeMaskWords(*game, randomEntityMask);
                            repl.randomEntityMaskWidth = game->wordCount;
                            repl.hasRandomEntityMask = true;
                            for (int32_t objectId = 0; objectId < game->objectCount; ++objectId) {
                                if (maskHasBit(randomEntityMask, objectId)) {
                                    repl.randomEntityChoices.push_back(objectId);
                                }
                            }
                        }
                        if (anyNonZero(randomDirMask)) {
                            repl.randomDirMask = storeMaskWords(*game, randomDirMask);
                            repl.randomDirMaskWidth = game->movementWordCount;
                            repl.hasRandomDirMask = true;
                            for (int32_t layer = 0; layer < game->layerCount; ++layer) {
                                if (getShiftedMask5(randomDirMask, 5 * layer) != 0) {
                                    repl.randomDirLayers.push_back(layer);
                                }
                            }
                        }
                        if (anyNonZero(objectsClear) || anyNonZero(objectsSet) || anyNonZero(movementsClear)
                            || anyNonZero(movementsSet) || anyNonZero(movementsLayerMask)
                            || anyNonZero(randomEntityMask) || anyNonZero(randomDirMask)) {
                            pat.replacement = std::move(repl);
                        }
                    }
                }

                out.push_back(std::move(pat));
            }
            return out;
        };

        rule.patterns.clear();
        rule.ellipsisCount.clear();
        for (size_t rowIndex = 0; rowIndex < variantLhsRowsExpanded.size(); ++rowIndex) {
            const ParsedRow& lhsRow = variantLhsRowsExpanded[rowIndex];
            const ParsedRow* rhsRow = (rowIndex < variantRhsRowsExpanded.size()) ? &variantRhsRowsExpanded[rowIndex] : nullptr;
            auto loweredRow = buildPatternRow(lhsRow, rhsRow);
            int32_t ellipsisInRow = 0;
            for (const auto& pat : loweredRow) {
                if (pat.kind == puzzlescript::Pattern::Kind::Ellipsis) {
                    ++ellipsisInRow;
                }
            }
            rule.patterns.push_back(std::move(loweredRow));
            rule.ellipsisCount.push_back(ellipsisInRow);
        }

        // Build row/rule masks so runtime fast-paths don't deref null.
        auto ruleMaskWords = makeEmptyMask(game->wordCount);
        const uint32_t rowMasksFirst = static_cast<uint32_t>(game->cellRowMaskOffsets.size());
        for (const auto& row : rule.patterns) {
            auto rowMaskWords = makeEmptyMask(game->wordCount);
            for (const auto& pat : row) {
                if (pat.kind != puzzlescript::Pattern::Kind::CellPattern) {
                    continue;
                }
                const auto off = pat.objectsPresent;
                if (off == puzzlescript::kNullMaskOffset) {
                    continue;
                }
                for (uint32_t w = 0; w < game->wordCount; ++w) {
                    const puzzlescript::MaskWord word = game->maskArena[static_cast<size_t>(off + w)];
                    rowMaskWords[static_cast<size_t>(w)] |= word;
                    ruleMaskWords[static_cast<size_t>(w)] |= word;
                }
            }
            game->cellRowMaskOffsets.push_back(storeMaskWords(*game, rowMaskWords));
        }
        rule.cellRowMasksFirst = rowMasksFirst;
        rule.cellRowMasksCount = static_cast<uint32_t>(game->cellRowMaskOffsets.size()) - rowMasksFirst;
        rule.ruleMask = storeMaskWords(*game, ruleMaskWords);

        // Movement row masks: JS IR includes these; build them similarly to
        // cell_row_masks but over movement masks.
        auto ruleMovementMaskWords = makeEmptyMask(game->movementWordCount);
        const uint32_t rowMoveMasksFirst = static_cast<uint32_t>(game->cellRowMaskMovementsOffsets.size());
        for (const auto& row : rule.patterns) {
            auto rowMoveMaskWords = makeEmptyMask(game->movementWordCount);
            for (const auto& pat : row) {
                if (pat.kind != puzzlescript::Pattern::Kind::CellPattern) {
                    continue;
                }
                const auto off = pat.movementsPresent;
                if (off == puzzlescript::kNullMaskOffset) {
                    continue;
                }
                for (uint32_t w = 0; w < game->movementWordCount; ++w) {
                    const puzzlescript::MaskWord word = game->maskArena[static_cast<size_t>(off + w)];
                    rowMoveMaskWords[static_cast<size_t>(w)] |= word;
                    ruleMovementMaskWords[static_cast<size_t>(w)] |= word;
                }
            }
            game->cellRowMaskMovementsOffsets.push_back(storeMaskWords(*game, rowMoveMaskWords));
        }
        rule.cellRowMasksMovementsFirst = rowMoveMasksFirst;
        rule.cellRowMasksMovementsCount =
            static_cast<uint32_t>(game->cellRowMaskMovementsOffsets.size()) - rowMoveMasksFirst;
        rule.hasRuleMovementMask = std::any_of(
            ruleMovementMaskWords.begin(),
            ruleMovementMaskWords.end(),
            [](int32_t word) { return word != 0; });
        rule.ruleMovementMask = storeMaskWords(*game, ruleMovementMaskWords);

        const std::string signature = ruleVariantSignature(
            entry.lineNumber,
            concreteRuleDirection,
            rigidRule,
            randomRule,
            lateRule,
            variantLhsRowsExpanded,
            variantRhsRowsExpanded,
            parsedCommands
        );
        outputGroup->push_back(std::move(rule));
        outputSignatures->push_back(signature);
        }
        }
        }
    }

    auto dedupeRuleGroups = [](std::vector<std::vector<puzzlescript::Rule>>& groups,
                               std::vector<std::vector<std::string>>& signatures) {
        for (size_t groupIndex = 0; groupIndex < groups.size() && groupIndex < signatures.size(); ++groupIndex) {
            auto& group = groups[groupIndex];
            auto& groupSigs = signatures[groupIndex];
            if (group.size() != groupSigs.size() || group.empty()) {
                continue;
            }
            std::set<std::string> seen;
            std::vector<uint8_t> keep(group.size(), 0);
            for (size_t i = group.size(); i-- > 0;) {
                const auto [_, inserted] = seen.insert(groupSigs[i]);
                if (inserted) {
                    keep[i] = 1;
                }
            }
            std::vector<puzzlescript::Rule> filteredGroup;
            std::vector<std::string> filteredSigs;
            filteredGroup.reserve(group.size());
            filteredSigs.reserve(groupSigs.size());
            for (size_t i = 0; i < group.size(); ++i) {
                if (keep[i] == 0) {
                    continue;
                }
                filteredGroup.push_back(std::move(group[i]));
                filteredSigs.push_back(std::move(groupSigs[i]));
            }
            group = std::move(filteredGroup);
            groupSigs = std::move(filteredSigs);
        }
    };
    dedupeRuleGroups(game->rules, earlyRuleSignatures);
    dedupeRuleGroups(game->lateRules, lateRuleSignatures);

    // Rigid bookkeeping tables used by runtime conflict resolution.
    game->rigid = false;
    game->rigidGroups.clear();
    game->rigidGroupIndexToGroupIndex.clear();
    game->groupIndexToRigidGroupIndex.clear();
    game->groupNumberToRigidGroupIndex.clear();
    game->groupIndexToRigidGroupIndex.reserve(game->rules.size());

    int32_t maxGroupNumber = -1;
    for (const auto& group : game->rules) {
        for (const auto& rule : group) {
            if (rule.groupNumber > maxGroupNumber) {
                maxGroupNumber = rule.groupNumber;
            }
        }
    }
    if (maxGroupNumber >= 0) {
        game->groupNumberToRigidGroupIndex.assign(static_cast<size_t>(maxGroupNumber + 1), -1);
    }

    for (int32_t groupIndex = 0; groupIndex < static_cast<int32_t>(game->rules.size()); ++groupIndex) {
        const auto& group = game->rules[static_cast<size_t>(groupIndex)];
        bool anyRigid = false;
        for (const auto& rule : group) {
            if (rule.rigid) {
                anyRigid = true;
                break;
            }
        }
        if (!anyRigid) {
            game->groupIndexToRigidGroupIndex.push_back(-1);
            continue;
        }

        game->rigid = true;
        const int32_t rigidGroupIndex = static_cast<int32_t>(game->rigidGroups.size());
        game->rigidGroups.push_back(true);
        game->rigidGroupIndexToGroupIndex.push_back(groupIndex);
        game->groupIndexToRigidGroupIndex.push_back(rigidGroupIndex);
        for (const auto& rule : group) {
            if (rule.groupNumber >= 0
                && static_cast<size_t>(rule.groupNumber) < game->groupNumberToRigidGroupIndex.size()) {
                game->groupNumberToRigidGroupIndex[static_cast<size_t>(rule.groupNumber)] = rigidGroupIndex;
            }
        }
    }

    auto calculateLoopPoints = [](const std::vector<std::pair<int32_t, int32_t>>& loopRanges,
                                  const std::vector<std::vector<puzzlescript::Rule>>& ruleGroups) {
        std::map<int32_t, int32_t> loopPoint;
        for (const auto& [loopStartLine, loopEndLine] : loopRanges) {
            int32_t initGroupIndex = -1;
            for (int32_t groupIndex = 0; groupIndex < static_cast<int32_t>(ruleGroups.size()); ++groupIndex) {
                const auto& ruleGroup = ruleGroups[static_cast<size_t>(groupIndex)];
                if (ruleGroup.empty()) {
                    continue;
                }

                const int32_t firstRuleLine = ruleGroup.front().lineNumber;
                if (loopEndLine < firstRuleLine) {
                    break;
                }

                const bool ruleInLoop = loopStartLine <= firstRuleLine && firstRuleLine <= loopEndLine;
                if (!ruleInLoop) {
                    continue;
                }

                if (initGroupIndex == -1) {
                    initGroupIndex = groupIndex;
                }
                const auto prev = loopPoint.find(groupIndex - 1);
                if (groupIndex > 0 && prev != loopPoint.end() && prev->second == initGroupIndex) {
                    loopPoint.erase(prev);
                }
                loopPoint[groupIndex] = initGroupIndex;
            }
        }
        return loopPoint;
    };
    auto buildLoopPointTable = [](const std::map<int32_t, int32_t>& points) {
        puzzlescript::LoopPointTable table;
        if (points.empty()) {
            return table;
        }
        const int32_t maxKey = points.rbegin()->first;
        table.entries.assign(static_cast<size_t>(maxKey + 1), std::nullopt);
        for (const auto& [k, v] : points) {
            if (k >= 0 && static_cast<size_t>(k) < table.entries.size()) {
                table.entries[static_cast<size_t>(k)] = v;
            }
        }
        return table;
    };
    const auto earlyLoopPointMap = calculateLoopPoints(loopRanges, game->rules);
    const auto lateLoopPointMap = calculateLoopPoints(loopRanges, game->lateRules);
    game->loopPoint = buildLoopPointTable(earlyLoopPointMap);
    game->lateLoopPoint = buildLoopPointTable(lateLoopPointMap);

    game->winConditions.clear();
    for (const auto& entry : state.winconditions) {
        if (entry.tokens.size() < 2) {
            continue;
        }

        puzzlescript::WinCondition condition;
        if (entry.tokens[0] == "no") {
            condition.quantifier = -1;
        } else if (entry.tokens[0] == "all") {
            condition.quantifier = 1;
        } else {
            condition.quantifier = 0; // "some"
        }
        condition.lineNumber = entry.lineNumber;

        auto allObjectsMask = [&]() {
            auto mask = makeEmptyMask(game->wordCount);
            for (int32_t id = 0; id < game->objectCount; ++id) {
                setMaskBit(mask, id);
            }
            return mask;
        };

        auto resolveWinMask = [&](const std::string& name, bool& aggregate) {
            aggregate = false;
            if (name == "\nall\n") {
                return allObjectsMask();
            }
            if (objectIdByName.find(name) != objectIdByName.end()
                || synonymOf.find(name) != synonymOf.end()
                || propertyOf.find(name) != propertyOf.end()) {
                std::set<std::string> visiting;
                return resolveMask(resolveMask, name, visiting);
            }
            if (aggregateOf.find(name) != aggregateOf.end()) {
                aggregate = true;
                std::set<std::string> visiting;
                return resolveMask(resolveMask, name, visiting);
            }
            return makeEmptyMask(game->wordCount);
        };

        bool aggr1 = false;
        bool aggr2 = false;
        const auto filter1 = resolveWinMask(entry.tokens[1], aggr1);
        const std::string filter2Name = entry.tokens.size() == 4 ? entry.tokens[3] : std::string("\nall\n");
        const auto filter2 = resolveWinMask(filter2Name, aggr2);

        condition.filter1 = storeMaskWords(*game, filter1);
        condition.filter2 = storeMaskWords(*game, filter2);
        condition.aggr1 = aggr1;
        condition.aggr2 = aggr2;
        game->winConditions.push_back(std::move(condition));
    }
    auto soundDirectionMask = [](const std::string& direction) -> int32_t {
        if (direction == "up") return 1;
        if (direction == "down") return 2;
        if (direction == "left") return 4;
        if (direction == "right") return 8;
        if (direction == "horizontal") return 12;
        if (direction == "vertical") return 3;
        if (direction == "orthogonal") return 15;
        if (direction == "___action____") return 16;
        return 0;
    };

    auto parseSeed = [](const std::string& text) -> int32_t {
        try {
            return static_cast<int32_t>(std::stol(text));
        } catch (const std::exception&) {
            return 0;
        }
    };

    auto expandSoundTargets = [&](auto&& self, const std::string& name, std::set<std::string>& visiting) -> std::vector<std::string> {
        if (!visiting.insert(name).second) {
            return {};
        }
        std::vector<std::string> targets;
        if (objectIdByName.find(name) != objectIdByName.end()) {
            targets.push_back(name);
        } else if (auto synonym = synonymOf.find(name); synonym != synonymOf.end()) {
            targets = self(self, synonym->second, visiting);
        } else if (auto property = propertyOf.find(name); property != propertyOf.end()) {
            for (const auto& item : property->second) {
                auto expanded = self(self, item, visiting);
                targets.insert(targets.end(), expanded.begin(), expanded.end());
            }
        }
        visiting.erase(name);
        return targets;
    };

    game->sfxEvents.clear();
    game->sfxCreationMasks.clear();
    game->sfxDestructionMasks.clear();
    game->sfxMovementMasks.assign(static_cast<size_t>(game->layerCount), {});
    game->sfxMovementFailureMasks.clear();
    for (const auto& entry : state.sounds) {
        if (entry.tokens.size() < 2) {
            continue;
        }
        const auto& seedToken = entry.tokens.back();
        if (seedToken.kind != "SOUND") {
            continue;
        }
        const int32_t seed = parseSeed(seedToken.text);
        const auto& first = entry.tokens.front();
        if (first.kind == "SOUNDEVENT") {
            game->sfxEvents[first.text] = seed;
            continue;
        }
        if (entry.tokens.size() < 3) {
            continue;
        }

        const std::string target = first.text;
        std::string verb = entry.tokens[1].text;
        std::vector<std::string> directions;
        for (size_t tokenIndex = 2; tokenIndex + 1 < entry.tokens.size(); ++tokenIndex) {
            if (entry.tokens[tokenIndex].kind == "DIRECTION") {
                directions.push_back(entry.tokens[tokenIndex].text);
            }
        }
        if (verb == "action") {
            verb = "move";
            directions = {"___action____"};
        }
        if (directions.empty()) {
            directions = {"orthogonal"};
        }

        int32_t directionMaskBits = 0;
        for (const auto& direction : directions) {
            directionMaskBits |= soundDirectionMask(direction);
        }

        puzzlescript::MaskVector objectMask = makeEmptyMask(game->wordCount);
        try {
            std::set<std::string> visiting;
            objectMask = resolveMask(resolveMask, target, visiting);
        } catch (const std::exception&) {
            objectMask = makeEmptyMask(game->wordCount);
        }

        if (verb == "move" || verb == "cantmove") {
            std::set<std::string> visiting;
            const auto targets = expandSoundTargets(expandSoundTargets, target, visiting);
            for (const auto& targetName : targets) {
                const auto objectIt = objectIdByName.find(targetName);
                if (objectIt == objectIdByName.end()) {
                    continue;
                }
                const int32_t objectId = objectIt->second;
                if (objectId < 0 || objectId >= static_cast<int32_t>(game->objectsById.size())) {
                    continue;
                }
                const int32_t layer = game->objectsById[static_cast<size_t>(objectId)].layer;
                if (layer < 0 || layer >= game->layerCount) {
                    continue;
                }
                puzzlescript::MaskVector concreteObjectMask = makeEmptyMask(game->wordCount);
                setMaskBit(concreteObjectMask, objectId);
                puzzlescript::MaskVector directionMaskWords = makeEmptyMask(game->movementWordCount);
                orShiftedMask5(directionMaskWords, 5 * layer, directionMaskBits);

                puzzlescript::SoundMaskEntry lowered;
                lowered.objectMask = storeMaskWords(*game, concreteObjectMask);
                lowered.directionMask = storeMaskWords(*game, directionMaskWords);
                lowered.directionMaskWidth = game->movementWordCount;
                lowered.seed = seed;
                if (verb == "move") {
                    game->sfxMovementMasks[static_cast<size_t>(layer)].push_back(lowered);
                } else {
                    game->sfxMovementFailureMasks.push_back(lowered);
                }
            }
            continue;
        }

        if (verb == "create" || verb == "destroy") {
            puzzlescript::SoundMaskEntry lowered;
            lowered.objectMask = storeMaskWords(*game, objectMask);
            lowered.directionMask = puzzlescript::kNullMaskOffset;
            lowered.directionMaskWidth = 0;
            lowered.seed = seed;
            if (verb == "create") {
                game->sfxCreationMasks.push_back(lowered);
            } else {
                game->sfxDestructionMasks.push_back(lowered);
            }
        }
    }

    outGame.information = std::move(game);
    outGame.initialMetaGameState = std::move(initialMetaGameState);
    return nullptr;
}

} // namespace puzzlescript::compiler
