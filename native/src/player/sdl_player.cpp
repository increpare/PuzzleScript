#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <SDL.h>

#include "player/sfxr.hpp"
#include "puzzlescript/puzzlescript.h"

#ifndef PS_FONT_JS_PATH
#define PS_FONT_JS_PATH "src/js/font.js"
#endif

namespace {

constexpr int kTerminalWidth = 34;
constexpr int kTerminalHeight = 13;
constexpr int kSfxrSampleRate = 22050;
constexpr int kTextCellPixelWidth = 6;
constexpr int kTextCellPixelHeight = 13;
constexpr const char* kBlankRow = "..................................";

const char* kIntroTemplate[kTerminalHeight] = {
    "..................................",
    "..................................",
    "..................................",
    "......Puzzle Script Terminal......",
    "..............v 1.8...............",
    "..................................",
    "..................................",
    "..................................",
    ".........insert cartridge.........",
    "..................................",
    "..................................",
    "..................................",
    "..................................",
};

const char* kMessageTemplate[kTerminalHeight] = {
    "..................................",
    "..................................",
    "..................................",
    "..................................",
    "..................................",
    "..................................",
    "..................................",
    "..................................",
    "..................................",
    "..................................",
    "..........X to continue...........",
    "..................................",
    "..................................",
};

std::string readFile(const std::string& path) {
    FILE* file = fopen(path.c_str(), "rb");
    if (!file) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    fseek(file, 0, SEEK_END);
    const long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    std::string data(static_cast<size_t>(size), '\0');
    fread(data.data(), 1, static_cast<size_t>(size), file);
    fclose(file);
    return data;
}

std::string trim(std::string value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front())) != 0) {
        value.erase(value.begin());
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())) != 0) {
        value.pop_back();
    }
    return value;
}

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::string replaceDots(std::string row) {
    std::replace(row.begin(), row.end(), '.', ' ');
    return row;
}

std::vector<std::string> wordWrap(const std::string& input, int width) {
    std::vector<std::string> lines;
    std::string remaining = input;
    while (static_cast<int>(remaining.size()) > width) {
        int split = width;
        for (int index = width; index >= 0; --index) {
            if (std::isspace(static_cast<unsigned char>(remaining[static_cast<size_t>(index)])) != 0) {
                split = index;
                break;
            }
        }
        if (split <= 0) {
            split = width;
        }
        lines.push_back(trim(remaining.substr(0, static_cast<size_t>(split))));
        remaining = trim(remaining.substr(static_cast<size_t>(split)));
    }
    if (!remaining.empty()) {
        lines.push_back(remaining);
    }
    if (lines.empty()) {
        lines.push_back("");
    }
    return lines;
}

std::string alignCentre(std::string value, int width) {
    if (static_cast<int>(value.size()) >= width) {
        return value.substr(0, static_cast<size_t>(width));
    }
    const int free = width - static_cast<int>(value.size());
    const int left = free / 2;
    return std::string(static_cast<size_t>(left), '.') + value + std::string(static_cast<size_t>(free - left), '.');
}

std::string alignRight(std::string value, int width) {
    if (static_cast<int>(value.size()) >= width) {
        return value.substr(0, static_cast<size_t>(width));
    }
    if (static_cast<int>(value.size()) < width - 1) {
        value.push_back('.');
    }
    return std::string(static_cast<size_t>(width - value.size()), '.') + value;
}

struct Color {
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
    uint8_t a = 255;
};

Color parseColor(std::string value) {
    value = trim(value);
    if (value.empty()) {
        return Color{255, 255, 255, 255};
    }
    if (value == "transparent") {
        return Color{0, 0, 0, 0};
    }
    if (value[0] == '#') {
        const std::string hex = value.substr(1);
        auto hexByte = [](std::string_view digits) -> uint8_t {
            return static_cast<uint8_t>(std::strtoul(std::string(digits).c_str(), nullptr, 16));
        };
        if (hex.size() == 3) {
            const uint8_t r = hexByte(std::string(2, hex[0]));
            const uint8_t g = hexByte(std::string(2, hex[1]));
            const uint8_t b = hexByte(std::string(2, hex[2]));
            return Color{r, g, b, 255};
        }
        if (hex.size() >= 6) {
            return Color{hexByte(std::string_view(hex).substr(0, 2)), hexByte(std::string_view(hex).substr(2, 2)), hexByte(std::string_view(hex).substr(4, 2)), 255};
        }
    }
    static const std::unordered_map<std::string, Color> kArneColors = {
        {"black", {0x00, 0x00, 0x00, 255}}, {"white", {0xff, 0xff, 0xff, 255}},
        {"grey", {0x9d, 0x9d, 0x9d, 255}}, {"gray", {0x9d, 0x9d, 0x9d, 255}},
        {"darkgrey", {0x69, 0x71, 0x75, 255}}, {"darkgray", {0x69, 0x71, 0x75, 255}},
        {"lightgrey", {0xcc, 0xcc, 0xcc, 255}}, {"lightgray", {0xcc, 0xcc, 0xcc, 255}},
        {"red", {0xbe, 0x26, 0x33, 255}}, {"darkred", {0x73, 0x29, 0x30, 255}}, {"lightred", {0xe0, 0x6f, 0x8b, 255}},
        {"brown", {0xa4, 0x64, 0x22, 255}}, {"darkbrown", {0x49, 0x3c, 0x2b, 255}}, {"lightbrown", {0xee, 0xb6, 0x2f, 255}},
        {"orange", {0xeb, 0x89, 0x31, 255}}, {"yellow", {0xf7, 0xe2, 0x6b, 255}},
        {"green", {0x44, 0x89, 0x1a, 255}}, {"darkgreen", {0x2f, 0x48, 0x4e, 255}}, {"lightgreen", {0xa3, 0xce, 0x27, 255}},
        {"blue", {0x1d, 0x57, 0xf7, 255}}, {"darkblue", {0x1b, 0x26, 0x32, 255}}, {"lightblue", {0xb2, 0xdc, 0xef, 255}},
        {"purple", {0x34, 0x2a, 0x97, 255}}, {"pink", {0xde, 0x65, 0xe2, 255}},
    };
    const auto found = kArneColors.find(toLower(value));
    return found == kArneColors.end() ? Color{255, 255, 255, 255} : found->second;
}

void setDrawColor(SDL_Renderer* renderer, Color color) {
    SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
}

struct Font {
    std::map<char, std::vector<std::string>> glyphs;
};

Font loadFont() {
    Font font;
    std::vector<std::string> candidates = {PS_FONT_JS_PATH, "src/js/font.js"};
    for (const auto& path : candidates) {
        std::ifstream input(path);
        if (!input) {
            continue;
        }
        std::string line;
        while (std::getline(input, line)) {
            const size_t quote = line.find('\'');
            if (quote == std::string::npos || quote + 2 >= line.size() || line[quote + 2] != '\'') {
                continue;
            }
            const char key = line[quote + 1];
            if (line.find('`', quote + 3) == std::string::npos) {
                continue;
            }
            std::vector<std::string> rows;
            while (std::getline(input, line)) {
                const size_t endQuote = line.find('`');
                if (endQuote != std::string::npos) {
                    line = trim(line.substr(0, endQuote));
                    if (!line.empty()) {
                        rows.push_back(line);
                    }
                    break;
                }
                line = trim(line);
                if (!line.empty()) {
                    rows.push_back(line);
                }
            }
            if (!rows.empty()) {
                font.glyphs[key] = rows;
            }
        }
        if (!font.glyphs.empty()) {
            return font;
        }
    }
    font.glyphs['X'] = {"10001","01010","00100","00100","00100","00100","00100","00100","00100","01010","10001","00000"};
    return font;
}

struct Audio {
    SDL_AudioDeviceID device = 0;
    SDL_AudioSpec spec{};
    std::map<int32_t, std::vector<float>> cache;

    bool init() {
        SDL_AudioSpec want{};
        want.freq = kSfxrSampleRate;
        want.format = AUDIO_F32SYS;
        want.channels = 1;
        want.samples = 1024;
        device = SDL_OpenAudioDevice(nullptr, 0, &want, &spec, 0);
        if (device == 0) {
            SDL_Log("SDL_OpenAudioDevice failed: %s", SDL_GetError());
            return false;
        }
        SDL_PauseAudioDevice(device, 0);
        return true;
    }

    void playSeed(int32_t seed) {
        if (device == 0) {
            return;
        }
        auto [it, inserted] = cache.emplace(seed, std::vector<float>{});
        if (inserted) {
            it->second = puzzlescript::player::generateSfxrFromSeed(seed, spec.freq);
        }
        SDL_QueueAudio(device, it->second.data(), static_cast<Uint32>(it->second.size() * sizeof(float)));
    }

    void close() {
        if (device != 0) {
            SDL_CloseAudioDevice(device);
            device = 0;
        }
    }
};

struct Player {
    ps_game* game = nullptr;
    ps_session* session = nullptr;
    std::string saveKey;
    std::string savePath;
    bool title = true;
    bool hasSave = false;
    int savedLevel = 0;
    int titleSelection = 0;
    bool titleSelected = false;
    bool titleContinue = false;
    Uint32 titleSelectedAt = 0;
    Uint32 lastAgainTick = 0;
    Uint32 lastRealtimeTick = 0;
    bool hasLastViewport = false;
    int lastMinX = 0;
    int lastMinY = 0;
    int lastMaxX = 0;
    int lastMaxY = 0;
    Font font;
    Audio audio;
};

struct Viewport {
    int minX = 0;
    int minY = 0;
    int maxX = 0;
    int maxY = 0;
};

std::optional<std::pair<int, int>> parseScreenSize(const char* value) {
    if (value == nullptr || value[0] == '\0') {
        return std::nullopt;
    }
    std::vector<int> numbers;
    const std::string text = value;
    for (size_t index = 0; index < text.size();) {
        while (index < text.size() && std::isdigit(static_cast<unsigned char>(text[index])) == 0) {
            ++index;
        }
        if (index >= text.size()) {
            break;
        }
        size_t end = index;
        while (end < text.size() && std::isdigit(static_cast<unsigned char>(text[end])) != 0) {
            ++end;
        }
        numbers.push_back(std::stoi(text.substr(index, end - index)));
        index = end;
    }
    if (numbers.size() < 2 || numbers[0] <= 0 || numbers[1] <= 0) {
        return std::nullopt;
    }
    return std::make_pair(numbers[0], numbers[1]);
}

Uint32 parseIntervalMs(const char* value, Uint32 fallback) {
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char* end = nullptr;
    const double seconds = std::strtod(value, &end);
    if (end == value || seconds < 0.0 || !std::isfinite(seconds)) {
        return fallback;
    }
    return static_cast<Uint32>(std::max(0.0, seconds * 1000.0));
}

std::string canonicalSaveKey(const std::string& key) {
    try {
        return std::filesystem::weakly_canonical(std::filesystem::path(key)).string();
    } catch (const std::exception&) {
        return key;
    }
}

std::string makeSavePath(const std::string& key) {
    char* pref = SDL_GetPrefPath("PuzzleScript", "NativePlayer");
    std::string dir = pref ? pref : "";
    if (pref) {
        SDL_free(pref);
    }
    const size_t hash = std::hash<std::string>{}(canonicalSaveKey(key));
    return dir + "save-" + std::to_string(static_cast<unsigned long long>(hash)) + ".txt";
}

void refreshSave(Player& player) {
    player.hasSave = false;
    player.savedLevel = 0;
    std::ifstream input(player.savePath);
    int level = 0;
    if (input >> level && level > 0 && level < ps_game_level_count(player.game)) {
        player.hasSave = true;
        player.savedLevel = level;
        player.titleSelection = 1;
    } else {
        player.titleSelection = 0;
    }
}

void saveProgress(Player& player) {
    ps_session_status_info status{};
    ps_session_status(player.session, &status);
    if (status.current_level_index <= 0 || status.current_level_index >= ps_game_level_count(player.game)) {
        std::remove(player.savePath.c_str());
        refreshSave(player);
        return;
    }
    std::ofstream output(player.savePath, std::ios::trunc);
    output << status.current_level_index << "\n";
    refreshSave(player);
}

void clearSave(Player& player) {
    std::remove(player.savePath.c_str());
    refreshSave(player);
}

void playNamed(Player& player, const char* name) {
    int32_t seed = 0;
    if (ps_game_sound_seed(player.game, name, &seed)) {
        player.audio.playSeed(seed);
    }
}

void playEvents(Player& player, const ps_step_result& result) {
    for (size_t i = 0; i < result.audio_event_count; ++i) {
        player.audio.playSeed(result.audio_events[i].seed);
    }
    for (size_t i = 0; i < result.ui_audio_event_count; ++i) {
        player.audio.playSeed(result.ui_audio_events[i].seed);
    }
}

std::string animateSelectedTitleRow(const std::string& row, Uint32 elapsedMs) {
    int frame = static_cast<int>(std::floor((static_cast<double>(elapsedMs) / 300.0) * 10.0)) + 2;
    const bool loadingText = frame > 12;
    std::string animated = loadingText ? "------------ loading  ------------" : row;
    frame %= 23;
    if (frame > 11) {
        frame = 11 - (frame % 12);
    }
    const int left = 11 - frame;
    const int right = 22 + frame;
    if (left >= 0 && right < static_cast<int>(animated.size())) {
        if (!loadingText) {
            animated = std::string(static_cast<size_t>(left), '.') + "#" + animated.substr(static_cast<size_t>(left + 1), static_cast<size_t>(right - left - 1)) + "#" + std::string(static_cast<size_t>(left), '.');
        } else {
            animated[static_cast<size_t>(left)] = '#';
            animated[static_cast<size_t>(right)] = '#';
        }
    }
    return animated;
}

std::vector<std::string> generateTitleRows(Player& player, Uint32 now) {
    if (ps_game_level_count(player.game) == 0) {
        std::vector<std::string> rows;
        for (const char* row : kIntroTemplate) {
            rows.push_back(replaceDots(row));
        }
        return rows;
    }

    const bool titleMode = player.hasSave;
    std::string title = ps_game_metadata_value(player.game, "title");
    if (title.empty()) {
        title = "PuzzleScript Game";
    }
    std::vector<std::string> titleLines = wordWrap(title, kTerminalWidth);
    for (auto& row : titleLines) {
        row = alignCentre(row, kTerminalWidth);
    }
    std::vector<std::string> authorLines;
    const std::string author = ps_game_metadata_value(player.game, "author");
    if (!author.empty()) {
        authorLines = wordWrap("by " + author, kTerminalWidth);
        for (auto& row : authorLines) {
            row = alignRight(row, kTerminalWidth);
        }
    }

    std::vector<std::string> controls = {".arrow keys to move..............."};
    int extraHeaderRows = 0;
    if (!ps_game_has_metadata(player.game, "noaction")) {
        controls.push_back(".X to action......................");
    } else {
        ++extraHeaderRows;
    }
    const bool hasUndo = !ps_game_has_metadata(player.game, "noundo");
    const bool hasRestart = !ps_game_has_metadata(player.game, "norestart");
    if (hasUndo && hasRestart) {
        controls.push_back(".Z to undo, R to restart..........");
    } else if (hasUndo) {
        controls.push_back(".Z to undo........................");
    } else if (hasRestart) {
        controls.push_back(".R to restart.....................");
    } else {
        ++extraHeaderRows;
    }
    if (extraHeaderRows > 1) {
        controls.push_back(kBlankRow);
        --extraHeaderRows;
    }

    const int headerSize = 5 + extraHeaderRows;
    while (static_cast<int>(titleLines.size() + authorLines.size()) > headerSize) {
        if (authorLines.size() > 1) {
            authorLines.pop_back();
        } else if (titleLines.size() > 1) {
            titleLines.pop_back();
        } else {
            break;
        }
    }
    int used = static_cast<int>(titleLines.size() + authorLines.size());
    int top = 0;
    int between = 0;
    int bottom = 0;
    if (used + top + between + bottom < headerSize) ++bottom;
    if (used + top + between + bottom < headerSize) ++between;
    while (used + top + between + bottom < headerSize) ++top;

    std::vector<std::string> rows;
    for (int i = 0; i < top; ++i) rows.push_back(kBlankRow);
    rows.insert(rows.end(), titleLines.begin(), titleLines.end());
    for (int i = 0; i < between; ++i) rows.push_back(kBlankRow);
    rows.insert(rows.end(), authorLines.begin(), authorLines.end());
    for (int i = 0; i < bottom; ++i) rows.push_back(kBlankRow);

    int selectionRow = -1;
    if (!titleMode) {
        rows.push_back(kBlankRow);
        selectionRow = static_cast<int>(rows.size());
        rows.push_back(player.titleSelected ? "-----------.start game.-----------" : "..........#.start game.#..........");
        rows.push_back(kBlankRow);
    } else if (player.titleSelection == 0) {
        selectionRow = static_cast<int>(rows.size());
        rows.push_back(player.titleSelected ? "------------.new game.------------" : "...........#.new game.#...........");
        rows.push_back(kBlankRow);
        rows.push_back(".............continue.............");
    } else {
        rows.push_back(".............new game.............");
        rows.push_back(kBlankRow);
        selectionRow = static_cast<int>(rows.size());
        rows.push_back(player.titleSelected ? "------------.continue.------------" : "...........#.continue.#...........");
    }
    rows.push_back(kBlankRow);
    rows.insert(rows.end(), controls.begin(), controls.end());
    rows.push_back(kBlankRow);
    while (static_cast<int>(rows.size()) < kTerminalHeight) {
        rows.push_back(kBlankRow);
    }
    rows.resize(kTerminalHeight);
    if (player.titleSelected && selectionRow >= 0 && selectionRow < static_cast<int>(rows.size())) {
        rows[static_cast<size_t>(selectionRow)] = animateSelectedTitleRow(rows[static_cast<size_t>(selectionRow)], now - player.titleSelectedAt);
    }
    for (auto& row : rows) {
        row = replaceDots(row);
    }
    return rows;
}

std::vector<std::string> generateMessageRows(Player& player) {
    std::vector<std::string> rows;
    for (const char* row : kMessageTemplate) {
        rows.push_back(replaceDots(row));
    }
    const std::string emptyLine = rows[9];
    const std::string xToContinue = rows[10];
    rows[10] = emptyLine;

    const std::vector<std::string> messageLines = wordWrap(ps_session_message_text(player.session), kTerminalWidth);
    int offset = 5 - (static_cast<int>(messageLines.size()) / 2);
    if (offset < 0) {
        offset = 0;
    }
    const int count = std::min<int>(messageLines.size(), 12);
    for (int i = 0; i < count; ++i) {
        const std::string& message = messageLines[static_cast<size_t>(i)];
        const int left = std::max(0, (kTerminalWidth - static_cast<int>(message.size())) / 2);
        std::string row = rows[static_cast<size_t>(offset + i)];
        row.replace(static_cast<size_t>(left), std::min(message.size(), row.size() - static_cast<size_t>(left)), message.substr(0, static_cast<size_t>(kTerminalWidth - left)));
        rows[static_cast<size_t>(offset + i)] = row;
    }
    int endPos = 10;
    if (count >= 10) {
        endPos = count < 12 ? count + 1 : 12;
    }
    rows[static_cast<size_t>(endPos)] = xToContinue;
    return rows;
}

void drawTextRows(SDL_Renderer* renderer, const Player& player, const std::vector<std::string>& rows, Color fg, Color bg, int winW, int winH) {
    setDrawColor(renderer, bg);
    SDL_RenderClear(renderer);
    const int scale = std::max(1, std::min(winW / (kTerminalWidth * kTextCellPixelWidth), winH / (kTerminalHeight * kTextCellPixelHeight)));
    const int cellW = kTextCellPixelWidth * scale;
    const int cellH = kTextCellPixelHeight * scale;
    const int x0 = (winW - kTerminalWidth * cellW) / 2;
    const int y0 = (winH - kTerminalHeight * cellH) / 2;
    setDrawColor(renderer, fg);
    for (int y = 0; y < static_cast<int>(rows.size()) && y < kTerminalHeight; ++y) {
        for (int x = 0; x < static_cast<int>(rows[static_cast<size_t>(y)].size()) && x < kTerminalWidth; ++x) {
            char ch = rows[static_cast<size_t>(y)][static_cast<size_t>(x)];
            auto it = player.font.glyphs.find(ch);
            if (it == player.font.glyphs.end()) {
                it = player.font.glyphs.find(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
            }
            if (it == player.font.glyphs.end()) {
                continue;
            }
            const auto& glyph = it->second;
            for (int gy = 0; gy < static_cast<int>(glyph.size()); ++gy) {
                for (int gx = 0; gx < static_cast<int>(glyph[static_cast<size_t>(gy)].size()); ++gx) {
                    if (glyph[static_cast<size_t>(gy)][static_cast<size_t>(gx)] != '1') {
                        continue;
                    }
                    SDL_Rect pixel{x0 + x * cellW + gx * scale, y0 + y * cellH + gy * scale, scale, scale};
                    SDL_RenderFillRect(renderer, &pixel);
                }
            }
        }
    }
}

Viewport computeViewport(Player& player, const ps_session_status_info& status) {
    Viewport viewport{0, 0, status.width, status.height};
    const auto flickscreen = parseScreenSize(ps_game_metadata_value(player.game, "flickscreen"));
    const auto zoomscreen = parseScreenSize(ps_game_metadata_value(player.game, "zoomscreen"));
    const auto screen = flickscreen.has_value() ? flickscreen : zoomscreen;
    if (!screen.has_value()) {
        player.hasLastViewport = false;
        return viewport;
    }

    const int screenW = std::min(screen->first, status.width);
    const int screenH = std::min(screen->second, status.height);
    int playerX = 0;
    int playerY = 0;
    if (ps_session_first_player_position(player.session, &playerX, &playerY)) {
        if (flickscreen.has_value()) {
            const int screenX = playerX / screenW;
            const int screenY = playerY / screenH;
            viewport.minX = screenX * screenW;
            viewport.minY = screenY * screenH;
        } else {
            viewport.minX = std::max(std::min(playerX - (screenW / 2), status.width - screenW), 0);
            viewport.minY = std::max(std::min(playerY - (screenH / 2), status.height - screenH), 0);
        }
        viewport.maxX = std::min(viewport.minX + screenW, status.width);
        viewport.maxY = std::min(viewport.minY + screenH, status.height);
        player.hasLastViewport = true;
        player.lastMinX = viewport.minX;
        player.lastMinY = viewport.minY;
        player.lastMaxX = viewport.maxX;
        player.lastMaxY = viewport.maxY;
        return viewport;
    }

    if (player.hasLastViewport) {
        viewport.minX = std::clamp(player.lastMinX, 0, status.width);
        viewport.minY = std::clamp(player.lastMinY, 0, status.height);
        viewport.maxX = std::clamp(player.lastMaxX, viewport.minX, status.width);
        viewport.maxY = std::clamp(player.lastMaxY, viewport.minY, status.height);
        return viewport;
    }

    viewport.maxX = screenW;
    viewport.maxY = screenH;
    return viewport;
}

void drawLevel(SDL_Renderer* renderer, Player& player, Color bg, int winW, int winH) {
    ps_session_status_info status{};
    ps_session_status(player.session, &status);
    setDrawColor(renderer, bg);
    SDL_RenderClear(renderer);
    if (status.width <= 0 || status.height <= 0) {
        return;
    }
    const Viewport viewport = computeViewport(player, status);
    const int viewW = std::max(1, viewport.maxX - viewport.minX);
    const int viewH = std::max(1, viewport.maxY - viewport.minY);
    const int tile = std::max(1, std::min(winW / viewW, winH / viewH));
    const int x0 = (winW - viewW * tile) / 2;
    const int y0 = (winH - viewH * tile) / 2;
    const int objectCount = ps_game_object_count(player.game);
    for (int x = viewport.minX; x < viewport.maxX; ++x) {
        for (int y = viewport.minY; y < viewport.maxY; ++y) {
            for (int objectId = 0; objectId < objectCount; ++objectId) {
                if (!ps_session_cell_has_object(player.session, x, y, objectId)) {
                    continue;
                }
                ps_object_info info{};
                if (!ps_game_object_info(player.game, objectId, &info) || info.sprite_width <= 0 || info.sprite_height <= 0) {
                    continue;
                }
                for (int sy = 0; sy < info.sprite_height; ++sy) {
                    for (int sx = 0; sx < info.sprite_width; ++sx) {
                        const int colorIndex = ps_game_object_sprite_value(player.game, objectId, sx, sy);
                        if (colorIndex < 0) {
                            continue;
                        }
                        Color color = parseColor(ps_game_object_color(player.game, objectId, static_cast<size_t>(colorIndex)));
                        if (color.a == 0) {
                            continue;
                        }
                        setDrawColor(renderer, color);
                        const int left = (sx * tile) / info.sprite_width;
                        const int right = ((sx + 1) * tile) / info.sprite_width;
                        const int top = (sy * tile) / info.sprite_height;
                        const int bottom = ((sy + 1) * tile) / info.sprite_height;
                        SDL_Rect rect{x0 + (x - viewport.minX) * tile + left, y0 + (y - viewport.minY) * tile + top, std::max(1, right - left), std::max(1, bottom - top)};
                        SDL_RenderFillRect(renderer, &rect);
                    }
                }
            }
        }
    }
}

bool loadLevel(Player& player, int levelIndex) {
    ps_error* error = nullptr;
    if (!ps_session_load_level(player.session, levelIndex, &error)) {
        SDL_Log("%s", ps_error_message(error));
        ps_free_error(error);
        return false;
    }
    ps_session_status_info status{};
    ps_session_status(player.session, &status);
    if (status.mode == PS_SESSION_MODE_MESSAGE) {
        playNamed(player, "showmessage");
    } else {
        playNamed(player, "startlevel");
    }
    return true;
}

void resetTimers(Player& player) {
    const Uint32 now = SDL_GetTicks();
    player.lastAgainTick = now;
    player.lastRealtimeTick = now;
}

void finishStartGame(Player& player, bool continued) {
    if (!continued) {
        clearSave(player);
    }
    const int level = continued ? player.savedLevel : 0;
    if (loadLevel(player, level)) {
        player.title = false;
        player.titleSelected = false;
        resetTimers(player);
    }
}

void afterStep(Player& player, const ps_step_result& result) {
    playEvents(player, result);
    ps_session_status_info status{};
    ps_session_status(player.session, &status);
    if (result.won) {
        playNamed(player, "endlevel");
    }
    if (status.title_screen) {
        clearSave(player);
        player.title = true;
        playNamed(player, "endgame");
    } else if (result.transitioned) {
        saveProgress(player);
        if (status.mode == PS_SESSION_MODE_MESSAGE) {
            playNamed(player, "showmessage");
        } else if (status.mode == PS_SESSION_MODE_LEVEL) {
            playNamed(player, "startlevel");
        }
    }
    resetTimers(player);
}

void beginTitleStart(Player& player, bool continued) {
    if (player.titleSelected) {
        return;
    }
    player.titleSelected = true;
    player.titleContinue = continued;
    player.titleSelectedAt = SDL_GetTicks();
    playNamed(player, "startgame");
}

std::optional<ps_input> keyToInput(SDL_Keycode key) {
    switch (key) {
        case SDLK_UP:
        case SDLK_w:
            return PS_INPUT_UP;
        case SDLK_LEFT:
        case SDLK_a:
            return PS_INPUT_LEFT;
        case SDLK_DOWN:
        case SDLK_s:
            return PS_INPUT_DOWN;
        case SDLK_RIGHT:
        case SDLK_d:
            return PS_INPUT_RIGHT;
        case SDLK_x:
        case SDLK_c:
        case SDLK_SPACE:
        case SDLK_RETURN:
        case SDLK_KP_ENTER:
            return PS_INPUT_ACTION;
        default:
            return std::nullopt;
    }
}

bool isActionKey(SDL_Keycode key) {
    const auto input = keyToInput(key);
    return input.has_value() && *input == PS_INPUT_ACTION;
}

void processAutomaticTicks(Player& player, Uint32 againIntervalMs, Uint32 realtimeIntervalMs) {
    if (player.title) {
        if (player.titleSelected && SDL_GetTicks() - player.titleSelectedAt > 300) {
            finishStartGame(player, player.titleContinue);
        }
        return;
    }

    ps_session_status_info status{};
    ps_session_status(player.session, &status);
    if (status.mode != PS_SESSION_MODE_LEVEL) {
        return;
    }

    const Uint32 now = SDL_GetTicks();
    if (ps_session_pending_again(player.session)) {
        if (now - player.lastAgainTick >= againIntervalMs) {
            const ps_step_result result = ps_session_tick(player.session);
            afterStep(player, result);
        }
        return;
    }

    if (realtimeIntervalMs > 0 && now - player.lastRealtimeTick >= realtimeIntervalMs) {
        const ps_step_result result = ps_session_tick(player.session);
        afterStep(player, result);
    }
}

int runPlayer(ps_game* game, const std::string& saveKey) {
    ps_error* error = nullptr;
    ps_session* session = nullptr;
    if (!ps_session_create(game, &session, &error)) {
        SDL_Log("%s", ps_error_message(error));
        ps_free_error(error);
        return 1;
    }

    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO) != 0) {
        SDL_Log("SDL_Init failed: %s", SDL_GetError());
        ps_session_destroy(session);
        return 1;
    }

    Player player;
    player.game = game;
    player.session = session;
    player.saveKey = saveKey;
    player.savePath = makeSavePath(saveKey);
    player.font = loadFont();
    refreshSave(player);
    (void)player.audio.init();
    resetTimers(player);

    SDL_Window* window = SDL_CreateWindow("PuzzleScript Native Player", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 800, 600, SDL_WINDOW_RESIZABLE);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!renderer && window) {
        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_SOFTWARE);
    }
    if (!window || !renderer) {
        SDL_Log("SDL window/renderer failed: %s", SDL_GetError());
        player.audio.close();
        if (renderer) SDL_DestroyRenderer(renderer);
        if (window) SDL_DestroyWindow(window);
        SDL_Quit();
        ps_session_destroy(session);
        return 1;
    }

    playNamed(player, "titlescreen");
    const Uint32 againIntervalMs = parseIntervalMs(ps_game_metadata_value(player.game, "again_interval"), 150);
    const Uint32 realtimeIntervalMs = parseIntervalMs(ps_game_metadata_value(player.game, "realtime_interval"), 0);
    int frameLimit = 0;
    if (const char* value = std::getenv("PS_PLAYER_FRAME_LIMIT")) {
        frameLimit = std::max(0, std::atoi(value));
    }
    int frames = 0;
    bool running = true;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            } else if (event.type == SDL_KEYDOWN && event.key.repeat == 0) {
                const SDL_Keycode key = event.key.keysym.sym;
                if (key == SDLK_ESCAPE) {
                    if (player.title) {
                        running = false;
                    } else {
                        player.title = true;
                        player.titleSelected = false;
                        refreshSave(player);
                        playNamed(player, "titlescreen");
                    }
                    continue;
                }
                if (player.title) {
                    if (player.titleSelected) {
                        continue;
                    }
                    if (player.hasSave && (key == SDLK_UP || key == SDLK_w)) {
                        player.titleSelection = 0;
                    } else if (player.hasSave && (key == SDLK_DOWN || key == SDLK_s)) {
                        player.titleSelection = 1;
                    } else if (isActionKey(key)) {
                        beginTitleStart(player, player.hasSave && player.titleSelection == 1);
                    }
                    continue;
                }
                if ((key == SDLK_z || key == SDLK_u) && !ps_game_has_metadata(player.game, "noundo")) {
                    if (ps_session_undo(player.session)) {
                        playNamed(player, "undo");
                    }
                    continue;
                }
                if (key == SDLK_r && !ps_game_has_metadata(player.game, "norestart")) {
                    if (ps_session_restart(player.session)) {
                        playNamed(player, "restart");
                    }
                    continue;
                }
                const auto input = keyToInput(key);
                if (input.has_value()) {
                    ps_session_status_info status{};
                    ps_session_status(player.session, &status);
                    if (*input == PS_INPUT_ACTION && status.mode == PS_SESSION_MODE_LEVEL && ps_game_has_metadata(player.game, "noaction")) {
                        continue;
                    }
                    const ps_step_result result = ps_session_step(player.session, *input);
                    afterStep(player, result);
                }
            }
        }

        processAutomaticTicks(player, againIntervalMs, realtimeIntervalMs);

        int winW = 0;
        int winH = 0;
        SDL_GetWindowSize(window, &winW, &winH);
        const Color fg = parseColor(ps_game_foreground_color(player.game));
        const Color bg = parseColor(ps_game_background_color(player.game));
        ps_session_status_info status{};
        ps_session_status(player.session, &status);
        if (player.title) {
            drawTextRows(renderer, player, generateTitleRows(player, SDL_GetTicks()), fg, bg, winW, winH);
        } else if (status.mode == PS_SESSION_MODE_MESSAGE) {
            drawTextRows(renderer, player, generateMessageRows(player), fg, bg, winW, winH);
        } else {
            drawLevel(renderer, player, bg, winW, winH);
        }
        SDL_RenderPresent(renderer);
        if (frameLimit > 0 && ++frames >= frameLimit) {
            running = false;
        }
    }

    player.audio.close();
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    ps_session_destroy(session);
    return 0;
}

} // namespace

int puzzlescript_cpp_run_player_for_game(ps_game* game, const std::string& saveKey) {
    return runPlayer(game, saveKey.empty() ? "native-game" : saveKey);
}

int puzzlescript_cpp_run_player_for_ir(const std::string& irPath) {
    const std::string json = readFile(irPath);
    ps_game* game = nullptr;
    ps_error* error = nullptr;
    if (!ps_load_ir_json(json.data(), json.size(), &game, &error)) {
        SDL_Log("%s", ps_error_message(error));
        ps_free_error(error);
        return 1;
    }

    const int result = puzzlescript_cpp_run_player_for_game(game, irPath);
    ps_free_game(game);
    return result;
}
