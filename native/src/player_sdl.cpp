#include <string>
#include <stdexcept>

#include <SDL.h>

#include "puzzlescript/puzzlescript.h"

namespace {

std::string readFile(const std::string& path) {
    FILE* file = fopen(path.c_str(), "rb");
    if (!file) {
        throw std::runtime_error("Failed to open IR file");
    }
    fseek(file, 0, SEEK_END);
    const long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    std::string data(static_cast<size_t>(size), '\0');
    fread(data.data(), 1, static_cast<size_t>(size), file);
    fclose(file);
    return data;
}

void drawChecker(SDL_Renderer* renderer, int tileX, int tileY, int tileSize) {
    SDL_Rect rect{tileX * tileSize, tileY * tileSize, tileSize, tileSize};
    SDL_SetRenderDrawColor(renderer, 16, 16, 16, 255);
    SDL_RenderFillRect(renderer, &rect);
    SDL_SetRenderDrawColor(renderer, 48, 48, 48, 255);
    SDL_RenderDrawRect(renderer, &rect);
}

} // namespace

int ps_cli_run_player_for_game(ps_game* game) {
    ps_error* error = nullptr;
    ps_session* session = nullptr;
    if (!ps_session_create(game, &session, &error)) {
        SDL_Log("%s", ps_error_message(error));
        ps_free_error(error);
        return 1;
    }

    ps_session_status_info status{};
    ps_session_status(session, &status);

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        SDL_Log("SDL_Init failed: %s", SDL_GetError());
        ps_session_destroy(session);
        return 1;
    }

    const int tileSize = 24;
    const int width = status.width > 0 ? status.width * tileSize : 640;
    const int height = status.height > 0 ? status.height * tileSize : 480;
    SDL_Window* window = SDL_CreateWindow("PuzzleScript Native Player", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, 0);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

    bool running = true;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            } else if (event.type == SDL_KEYDOWN) {
                if (event.key.keysym.sym == SDLK_ESCAPE) {
                    running = false;
                } else if (event.key.keysym.sym == SDLK_r) {
                    ps_session_restart(session);
                } else if (event.key.keysym.sym == SDLK_z) {
                    ps_session_undo(session);
                } else if (event.key.keysym.sym == SDLK_UP) {
                    ps_session_step(session, PS_INPUT_UP);
                } else if (event.key.keysym.sym == SDLK_LEFT) {
                    ps_session_step(session, PS_INPUT_LEFT);
                } else if (event.key.keysym.sym == SDLK_DOWN) {
                    ps_session_step(session, PS_INPUT_DOWN);
                } else if (event.key.keysym.sym == SDLK_RIGHT) {
                    ps_session_step(session, PS_INPUT_RIGHT);
                } else if (event.key.keysym.sym == SDLK_x) {
                    ps_session_step(session, PS_INPUT_ACTION);
                }
            }
        }

        ps_session_status(session, &status);
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);
        for (int y = 0; y < status.height; ++y) {
            for (int x = 0; x < status.width; ++x) {
                drawChecker(renderer, x, y, tileSize);
            }
        }
        SDL_RenderPresent(renderer);
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    ps_session_destroy(session);
    return 0;
}

int ps_cli_run_player(const std::string& irPath) {
    const std::string json = readFile(irPath);
    ps_game* game = nullptr;
    ps_error* error = nullptr;
    if (!ps_load_ir_json(json.data(), json.size(), &game, &error)) {
        SDL_Log("%s", ps_error_message(error));
        ps_free_error(error);
        return 1;
    }

    const int result = ps_cli_run_player_for_game(game);
    ps_free_game(game);
    return result;
}
