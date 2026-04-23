#include "compiler/lower_to_runtime.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <utf8proc.h>

namespace puzzlescript::compiler {

namespace {

uint32_t ceilDivU32(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

puzzlescript::MaskOffset storeMaskWords(puzzlescript::Game& game, const std::vector<int32_t>& words) {
    const auto offset = static_cast<puzzlescript::MaskOffset>(game.maskArena.size());
    game.maskArena.insert(game.maskArena.end(), words.begin(), words.end());
    return offset;
}

std::vector<int32_t> makeEmptyMask(uint32_t wordCount) {
    return std::vector<int32_t>(static_cast<size_t>(wordCount), 0);
}

void setMaskBit(std::vector<int32_t>& words, int32_t bitIndex) {
    if (bitIndex < 0) {
        return;
    }
    const uint32_t word = static_cast<uint32_t>(bitIndex) / 32U;
    const uint32_t bit = static_cast<uint32_t>(bitIndex) % 32U;
    if (word >= words.size()) {
        return;
    }
    words[word] |= (1U << bit);
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

std::vector<std::string> tokenizeRuleLine(std::string line) {
    auto replaceAll = [&](const std::string& from, const std::string& to) {
        size_t pos = 0;
        while ((pos = line.find(from, pos)) != std::string::npos) {
            line.replace(pos, from.size(), to);
            pos += to.size();
        }
    };
    replaceAll("[", " [ ");
    replaceAll("]", " ] ");
    replaceAll("|", " | ");
    replaceAll("->", " -> ");
    // Trim.
    while (!line.empty() && std::isspace(static_cast<unsigned char>(line.front()))) {
        line.erase(line.begin());
    }
    while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) {
        line.pop_back();
    }
    if (!line.empty() && line.front() == '+') {
        line.insert(1, " ");
    }
    std::vector<std::string> tokens;
    std::string cur;
    for (char ch : line) {
        if (std::isspace(static_cast<unsigned char>(ch))) {
            if (!cur.empty()) {
                tokens.push_back(cur);
                cur.clear();
            }
        } else {
            cur.push_back(ch);
        }
    }
    if (!cur.empty()) {
        tokens.push_back(cur);
    }
    return tokens;
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
    if (token == "moving") return 15;
    if (token == "stationary") return 0; // special-cased: goes to movementsMissing=0x1f
    return 0;
}

void setShiftedMask5(std::vector<int32_t>& words, int32_t shift, int32_t value5) {
    // shift is bit offset (multiple of 5).
    const int32_t wordIndex = shift / 32;
    const int32_t bitIndex = shift % 32;
    if (wordIndex < 0 || static_cast<size_t>(wordIndex) >= words.size()) {
        return;
    }
    const uint32_t mask = 0x1fU;
    uint32_t v = static_cast<uint32_t>(value5) & mask;
    uint32_t w0 = static_cast<uint32_t>(words[static_cast<size_t>(wordIndex)]);
    w0 &= ~(mask << bitIndex);
    w0 |= (v << bitIndex);
    words[static_cast<size_t>(wordIndex)] = static_cast<int32_t>(w0);
    if (bitIndex > 27) {
        // Straddles boundary.
        const int32_t next = wordIndex + 1;
        if (static_cast<size_t>(next) >= words.size()) {
            return;
        }
        const int32_t spill = bitIndex + 5 - 32;
        uint32_t w1 = static_cast<uint32_t>(words[static_cast<size_t>(next)]);
        w1 &= ~(mask >> (5 - spill));
        w1 |= (v >> (5 - spill));
        words[static_cast<size_t>(next)] = static_cast<int32_t>(w1);
    }
}

void orShiftedMask5(std::vector<int32_t>& words, int32_t shift, int32_t value5) {
    const int32_t wordIndex = shift / 32;
    const int32_t bitIndex = shift % 32;
    if (wordIndex < 0 || static_cast<size_t>(wordIndex) >= words.size()) {
        return;
    }
    const uint32_t mask = 0x1fU;
    uint32_t v = static_cast<uint32_t>(value5) & mask;
    uint32_t w0 = static_cast<uint32_t>(words[static_cast<size_t>(wordIndex)]);
    w0 |= (v << bitIndex);
    words[static_cast<size_t>(wordIndex)] = static_cast<int32_t>(w0);
    if (bitIndex > 27) {
        const int32_t next = wordIndex + 1;
        if (static_cast<size_t>(next) >= words.size()) {
            return;
        }
        const int32_t spill = bitIndex + 5 - 32;
        uint32_t w1 = static_cast<uint32_t>(words[static_cast<size_t>(next)]);
        w1 |= (v >> (5 - spill));
        words[static_cast<size_t>(next)] = static_cast<int32_t>(w1);
    }
}

} // namespace

std::unique_ptr<puzzlescript::Error> lowerToRuntimeGame(
    const ParserState& state,
    std::shared_ptr<const puzzlescript::Game>& outGame
) {
    auto game = std::make_shared<puzzlescript::Game>();
    game->schemaVersion = 1;

    // --- Metadata ---
    // ParserState.metadata is a flat [key, value, key, value...] list.
    game->metadataPairs = state.metadata;
    for (size_t i = 0; i + 1 < state.metadata.size(); i += 2) {
        game->metadataMap[state.metadata[i]] = state.metadata[i + 1];
    }
    game->metadataLines = state.metadataLines;

    // Preserve existing JS exporter behavior: background/text colors are part of IR.
    // If we don't resolve palettes yet, prefer explicit metadata.
    if (const auto it = game->metadataMap.find("text_color"); it != game->metadataMap.end()) {
        game->foregroundColor = it->second;
    }
    if (const auto it = game->metadataMap.find("background_color"); it != game->metadataMap.end()) {
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

    int32_t idCount = 0;
    for (int32_t layerIndex = 0; layerIndex < static_cast<int32_t>(state.collisionLayers.size()); ++layerIndex) {
        for (const auto& name : state.collisionLayers[static_cast<size_t>(layerIndex)]) {
            if (state.objects.find(name) == state.objects.end()) {
                continue;
            }
            game->idDict.push_back(name);
            ++idCount;
        }
    }
    game->objectCount = idCount;

    game->strideObject = static_cast<int32_t>(ceilDivU32(static_cast<uint32_t>(game->objectCount), 32U));
    game->wordCount = static_cast<uint32_t>(game->strideObject);
    game->strideMovement = static_cast<int32_t>(ceilDivU32(static_cast<uint32_t>(game->layerCount), 5U));
    game->movementWordCount = static_cast<uint32_t>(game->strideMovement);

    game->objectsById.resize(static_cast<size_t>(game->objectCount));

    // Build object defs and objectMaskTable.
    game->objectMaskTable.clear();
    game->objectMaskTable.reserve(static_cast<size_t>(game->objectCount));

    // Name -> id lookup by idDict index.
    std::map<std::string, int32_t> objectIdByName;
    for (int32_t id = 0; id < static_cast<int32_t>(game->idDict.size()); ++id) {
        objectIdByName[game->idDict[static_cast<size_t>(id)]] = id;
    }

    for (int32_t layerIndex = 0; layerIndex < static_cast<int32_t>(state.collisionLayers.size()); ++layerIndex) {
        for (const auto& name : state.collisionLayers[static_cast<size_t>(layerIndex)]) {
            const auto it = state.objects.find(name);
            if (it == state.objects.end()) {
                continue;
            }
            const auto idIt = objectIdByName.find(name);
            if (idIt == objectIdByName.end()) {
                continue;
            }
            const int32_t id = idIt->second;
            puzzlescript::ObjectDef def;
            def.name = name;
            def.id = id;
            def.layer = layerIndex;
            def.colors = it->second.colors;
            if (!it->second.spritematrix.empty()) {
                def.sprite = parseSpriteMatrix(it->second.spritematrix);
            } else {
                def.sprite = std::vector<std::vector<int32_t>>(5, std::vector<int32_t>(5, 0));
            }
            game->objectsById[static_cast<size_t>(id)] = std::move(def);

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
    const auto playerIt = objectIdByName.find("player");
    if (playerIt != objectIdByName.end()) {
        auto playerMask = makeEmptyMask(game->wordCount);
        setMaskBit(playerMask, playerIt->second);
        game->playerMaskAggregate = false;
        game->playerMask = storeMaskWords(*game, playerMask);
    }

    // --- Legend resolution (name -> object mask) ---
    std::map<std::string, std::vector<int32_t>> resolvedMasks;
    std::map<std::string, std::string> synonymOf;
    std::map<std::string, std::vector<std::string>> aggregateOf;
    std::map<std::string, std::vector<std::string>> propertyOf;

    for (const auto& entry : state.legendSynonyms) {
        if (!entry.items.empty()) {
            synonymOf[entry.name] = entry.items.front();
        }
    }
    for (const auto& entry : state.legendAggregates) {
        aggregateOf[entry.name] = entry.items;
    }
    for (const auto& entry : state.legendProperties) {
        propertyOf[entry.name] = entry.items;
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
            const auto& key = entry.name;
            const auto& val = entry.items.front();
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
            const auto& key = entry.name;
            if (glyphDict.find(key) != glyphDict.end()) {
                continue;
            }
            bool allFound = true;
            for (const auto& item : entry.items) {
                if (glyphDict.find(item) == glyphDict.end()) {
                    allFound = false;
                    break;
                }
            }
            if (!allFound) {
                continue;
            }
            auto glyph = blankGlyph;
            for (const auto& item : entry.items) {
                const auto& sub = glyphDict[item];
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

    auto resolveMask = [&](auto&& self, const std::string& name, std::set<std::string>& visiting) -> std::vector<int32_t> {
        if (auto it = resolvedMasks.find(name); it != resolvedMasks.end()) {
            return it->second;
        }
        if (!visiting.insert(name).second) {
            throw std::runtime_error("Legend cycle detected at '" + name + "'");
        }

        std::vector<int32_t> mask = makeEmptyMask(game->wordCount);

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

    // Resolve background object/property.
    std::vector<int32_t> backgroundMaskWords = makeEmptyMask(game->wordCount);
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
                    const uint32_t word = static_cast<uint32_t>(id) / 32U;
                    const uint32_t bit = static_cast<uint32_t>(id) % 32U;
                    if (word < backgroundMaskWords.size() && (backgroundMaskWords[word] & (1U << bit)) != 0) {
                        backgroundLayer = game->objectsById[static_cast<size_t>(id)].layer;
                        game->backgroundId = id;
                        game->backgroundLayer = backgroundLayer;
                        break;
                    }
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
        puzzlescript::LevelTemplate level;
        level.isMessage = srcLevel.isMessage;
        if (level.isMessage) {
            level.message = srcLevel.message;
            game->levels.push_back(std::move(level));
            continue;
        }
        level.layerCount = game->layerCount;
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

        for (int32_t y = 0; y < level.height; ++y) {
            const auto glyphs = splitUtf8Codepoints(srcLevel.rows[static_cast<size_t>(y)]);
            for (int32_t x = 0; x < level.width; ++x) {
                const std::string glyph = x < static_cast<int32_t>(glyphs.size()) ? glyphs[static_cast<size_t>(x)] : std::string{};
                std::vector<int32_t> cellMask = makeEmptyMask(game->wordCount);
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
                            cellMask[w] |= backgroundMaskWords[w];
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
    game->preparedSession.currentLevelIndex = 0;
    game->preparedSession.currentLevelTarget.reset();
    game->preparedSession.titleScreen = false;
    game->preparedSession.textMode = !game->levels.empty() && game->levels.front().isMessage;
    game->preparedSession.titleMode = 0;
    game->preparedSession.titleSelection = 0;
    game->preparedSession.titleSelected = false;
    game->preparedSession.messageSelected = false;
    game->preparedSession.winning = false;
    game->preparedSession.loadedLevelSeed = "native";
    game->preparedSession.hasRandomState = false;
    game->preparedSession.randomStateValid = false;
    game->preparedSession.randomStateS.clear();
    game->preparedSession.oldFlickscreenDat.clear();
    if (!game->levels.empty()) {
        game->preparedSession.level = game->levels.front();
        game->preparedSession.restart.width = game->preparedSession.level.width;
        game->preparedSession.restart.height = game->preparedSession.level.height;
        game->preparedSession.restart.objects = game->preparedSession.level.objects;
        game->preparedSession.restart.oldFlickscreenDat.clear();
    }

    // --- Rules / winconditions / sounds / loop points ---
    game->rules.clear();
    game->lateRules.clear();
    game->rules.emplace_back();
    auto& ruleGroup = game->rules.back();

    // Precompute (best-effort) single-layer info for legend names: if a mask's
    // set bits all live on the same collision layer, we can treat it as
    // single-layer for rule movement masks.
    auto maskSingleLayer = [&](const std::vector<int32_t>& mask) -> std::optional<int32_t> {
        std::optional<int32_t> layer;
        for (int32_t id = 0; id < game->objectCount; ++id) {
            const uint32_t word = static_cast<uint32_t>(id) / 32U;
            const uint32_t bit = static_cast<uint32_t>(id) % 32U;
            if (word >= mask.size()) {
                continue;
            }
            if ((mask[word] & (1U << bit)) == 0) {
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

    // Rule lowering: a subset of JS rulesToMask (enough to start converging).
    for (const auto& entry : state.rules) {
        const auto tokens = tokenizeRuleLine(entry.rule);
        if (tokens.empty()) {
            continue;
        }
        // Skip loop markers for now.
        if (tokens.size() == 1 && (tokens.front() == "startloop" || tokens.front() == "endloop")) {
            continue;
        }
        auto arrowIt = std::find(tokens.begin(), tokens.end(), "->");
        if (arrowIt == tokens.end()) {
            continue;
        }

        // Directions/modifiers at start (optional). JS defaults rules with no
        // explicit direction to orthogonal, then expands one rule per direction.
        size_t cursor = 0;
        bool rigidRule = false;
        bool randomRule = false;
        bool lateRule = false;
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
            }
            ++cursor;
        }
        if (ruleDirections.empty()) {
            addDirectionAggregate("orthogonal");
        }

        struct ParsedItem {
            std::string dir;
            std::string name;
        };
        struct ParsedCell {
            bool isEllipsis = false;
            std::vector<ParsedItem> items;
        };
        using ParsedRow = std::vector<ParsedCell>;

        auto parseSide = [&](size_t start, size_t end) -> std::vector<ParsedRow> {
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
                    if ((dirMaskFromToken(tokens[i]) != 0 || tokens[i] == "stationary" || tokens[i] == "no" || tokens[i] == "random") && (i + 1) < end) {
                        dir = tokens[i];
                        name = tokens[i + 1];
                        i += 2;
                    } else {
                        i += 1;
                    }
                    if (name == "|") {
                        continue;
                    }
                    cell.items.push_back({std::move(dir), std::move(name)});
                }
                if (cell.isEllipsis || !cell.items.empty()) {
                    current.push_back(std::move(cell));
                }
                if (i < end && tokens[i] == "]") {
                    ++i;
                }
                if (!current.empty()) {
                    rows.push_back(std::move(current));
                }
            }
            return rows;
        };

        const size_t arrowPos = static_cast<size_t>(std::distance(tokens.begin(), arrowIt));
        const auto lhsRows = parseSide(cursor, arrowPos);
        const auto rhsRows = parseSide(arrowPos + 1, tokens.size());

        auto absolutizeDir = [](const std::string& forward, const std::string& dir) -> std::string {
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

        puzzlescript::Rule rule;
        rule.direction = dirMaskFromToken(concreteRuleDirection);
        rule.lineNumber = entry.lineNumber;
        rule.groupNumber = entry.lineNumber;
        rule.rigid = rigidRule;
        rule.isRandom = randomRule;
        rule.hasReplacements = !variantRhsRows.empty();

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
                auto movementsPresent = std::vector<int32_t>(static_cast<size_t>(game->movementWordCount), 0);
                auto movementsMissing = std::vector<int32_t>(static_cast<size_t>(game->movementWordCount), 0);
                std::vector<puzzlescript::MaskOffset> anyOffsets;

                std::vector<int32_t> layersUsed(game->layerCount, 0);
                for (const auto& item : cell.items) {
                    if (item.dir == "random") {
                        continue; // handled on RHS replacement
                    }
                    std::set<std::string> visiting;
                    const auto mask = resolveMask(resolveMask, item.name, visiting);
                    const auto singleLayer = maskSingleLayer(mask);

                    if (item.dir == "no") {
                        for (size_t w = 0; w < objectsMissing.size(); ++w) {
                            objectsMissing[w] |= mask[w];
                        }
                        continue;
                    }

                    if (singleLayer.has_value()) {
                        for (size_t w = 0; w < objectsPresent.size(); ++w) {
                            objectsPresent[w] |= mask[w];
                        }
                        layersUsed[static_cast<size_t>(*singleLayer)] = 1;
                    } else {
                        // "or" properties end up as "anyObjectsPresent" in JS.
                        const auto off = storeMaskWords(*game, mask);
                        game->anyObjectOffsets.push_back(off);
                        anyOffsets.push_back(off);
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
                pat.objectsPresent = storeMaskWords(*game, objectsPresent);
                pat.objectsMissing = storeMaskWords(*game, objectsMissing);
                pat.movementsPresent = storeMaskWords(*game, movementsPresent);
                pat.movementsMissing = storeMaskWords(*game, movementsMissing);
                pat.anyObjectsFirst = static_cast<uint32_t>(game->anyObjectOffsets.size() - anyOffsets.size());
                pat.anyObjectsCount = static_cast<uint32_t>(anyOffsets.size());

                if (rhsRow && cellIndex < rhsRow->size()) {
                    const ParsedCell& rhsCell = (*rhsRow)[cellIndex];
                    if (!rhsCell.isEllipsis) {
                        // Compute clear/set per layer for RHS. Best-effort:
                        // clear all layers used by either lhs cell single-layer masks
                        // or rhs cell single-layer masks, then set rhs objects.
                        auto objectsClear = makeEmptyMask(game->wordCount);
                        auto objectsSet = makeEmptyMask(game->wordCount);
                        auto movementsClear = std::vector<int32_t>(static_cast<size_t>(game->movementWordCount), 0);
                        auto movementsSet = std::vector<int32_t>(static_cast<size_t>(game->movementWordCount), 0);
                        auto movementsLayerMask = std::vector<int32_t>(static_cast<size_t>(game->movementWordCount), 0);

                        auto markLayerClear = [&](int32_t layer) {
                            if (layer < 0 || layer >= game->layerCount) return;
                            const auto off = game->layerMaskOffsets[static_cast<size_t>(layer)];
                            for (uint32_t w = 0; w < game->wordCount; ++w) {
                                objectsClear[static_cast<size_t>(w)] |= static_cast<int32_t>(game->maskArena[static_cast<size_t>(off + w)]);
                            }
                        };

                        // LHS layersUsed already collected.
                        for (int32_t layer = 0; layer < game->layerCount; ++layer) {
                            if (layersUsed[static_cast<size_t>(layer)] != 0) {
                                markLayerClear(layer);
                            }
                        }

                        for (const auto& item : rhsCell.items) {
                            if (item.dir == "random") {
                                continue;
                            }
                            std::set<std::string> visiting;
                            const auto mask = resolveMask(resolveMask, item.name, visiting);
                            const auto singleLayer = maskSingleLayer(mask);

                            if (item.dir == "no") {
                                // Explicit delete.
                                for (size_t w = 0; w < objectsClear.size(); ++w) {
                                    objectsClear[w] |= mask[w];
                                }
                                if (singleLayer.has_value()) {
                                    markLayerClear(*singleLayer);
                                }
                                continue;
                            }

                            for (size_t w = 0; w < objectsSet.size(); ++w) {
                                objectsSet[w] |= mask[w];
                            }
                            if (singleLayer.has_value()) {
                                markLayerClear(*singleLayer);
                                const int32_t layer = *singleLayer;
                                if (item.dir == "stationary") {
                                    orShiftedMask5(movementsClear, 5 * layer, 0x1f);
                                    orShiftedMask5(movementsLayerMask, 5 * layer, 0x1f);
                                } else if (!item.dir.empty()) {
                                    const int32_t dm = dirMaskFromToken(item.dir);
                                    if (dm != 0) {
                                        orShiftedMask5(movementsClear, 5 * layer, 0x1f);
                                        orShiftedMask5(movementsSet, 5 * layer, dm);
                                        orShiftedMask5(movementsLayerMask, 5 * layer, 0x1f);
                                    }
                                }
                            }
                        }

                        puzzlescript::Replacement repl;
                        repl.objectsClear = storeMaskWords(*game, objectsClear);
                        repl.objectsSet = storeMaskWords(*game, objectsSet);
                        repl.movementsClear = storeMaskWords(*game, movementsClear);
                        repl.movementsSet = storeMaskWords(*game, movementsSet);
                        repl.movementsLayerMask = storeMaskWords(*game, movementsLayerMask);
                        pat.replacement = std::move(repl);
                    }
                }

                out.push_back(std::move(pat));
            }
            return out;
        };

        rule.patterns.clear();
        for (size_t rowIndex = 0; rowIndex < variantLhsRows.size(); ++rowIndex) {
            const ParsedRow& lhsRow = variantLhsRows[rowIndex];
            const ParsedRow* rhsRow = (rowIndex < variantRhsRows.size()) ? &variantRhsRows[rowIndex] : nullptr;
            rule.patterns.push_back(buildPatternRow(lhsRow, rhsRow));
        }
        rule.ellipsisCount.assign(rule.patterns.size(), 0);

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
                    const int32_t word = game->maskArena[static_cast<size_t>(off + w)];
                    rowMaskWords[static_cast<size_t>(w)] |= word;
                    ruleMaskWords[static_cast<size_t>(w)] |= word;
                }
            }
            game->cellRowMaskOffsets.push_back(storeMaskWords(*game, rowMaskWords));
        }
        rule.cellRowMasksFirst = rowMasksFirst;
        rule.cellRowMasksCount = static_cast<uint32_t>(game->cellRowMaskOffsets.size()) - rowMasksFirst;
        rule.ruleMask = storeMaskWords(*game, ruleMaskWords);

        if (lateRule) {
            // TODO: lower late rules once we parse the "late" keyword.
            // For now, treat these as early rules to avoid emitting a non-empty
            // late_rules group in IR diff mode.
            ruleGroup.push_back(std::move(rule));
        } else {
        // TODO: detect "late" rules properly; for now everything is early.
        ruleGroup.push_back(std::move(rule));
        }
        }
    }

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
    game->sfxEvents.clear();

    outGame = std::move(game);
    return nullptr;
}

} // namespace puzzlescript::compiler
