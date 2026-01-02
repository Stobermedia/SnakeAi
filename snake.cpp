#include <SDL2/SDL.h>
#include <deque>
#include <random>
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>
#include <torch/torch.h>
#include <csignal>
#include <sstream>
#include <iomanip>

volatile std::sig_atomic_t force_render_next = 0;

void signal_handler(int signum){
    force_render_next = 1;
}

using vec_t = std::vector<float>;

// Window layout constants - larger for full page feel
const int GAME_SIZE = 500;      // Snake game area
const int STATS_W = 700;        // Stats panel width (includes controls)
const int NET_VIS_H = 400;      // Network visualization height
const int WINDOW_W = GAME_SIZE + STATS_W;
const int WINDOW_H = GAME_SIZE + NET_VIS_H;

// Game constants
const int CELL = 40;  // Larger cells = smaller grid (12x12)
const int COLS = GAME_SIZE / CELL;
const int ROWS = GAME_SIZE / CELL;

// Neural network constants (fixed)
const int INPUT_SIZE = 16;
const int HIDDEN_SIZE = 128;
const int OUTPUT_SIZE = 4;

// Initial constants
const float EPSILON_START = 1.0f;
const float INITIAL_LEARNING_RATE = 0.001f;

// Adjustable parameters (defaults)
struct TrainingParams {
    float learning_rate = 0.001f;
    float gamma = 0.99f;
    float epsilon_decay = 0.998f;
    float epsilon_end = 0.01f;
    int batch_size = 128;
    int replay_buffer_size = 50000;
    float reward_food = 10.0f;
    float reward_closer = 0.1f;
    float penalty_away = -0.15f;
    float penalty_death = -10.0f;
};

enum class Dir { UP=-1, DOWN=1, LEFT=-2, RIGHT=2 };

struct Point {
    int x, y;
    bool operator==(Point const& o) const { return x==o.x && y==o.y; }
};

struct Experience {
    vec_t state;
    int action;
    float reward;
    vec_t next_state;
    bool done;
};

struct QNetImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    QNetImpl() {
        fc1 = register_module("fc1", torch::nn::Linear(INPUT_SIZE, HIDDEN_SIZE));
        fc2 = register_module("fc2", torch::nn::Linear(HIDDEN_SIZE, HIDDEN_SIZE));
        fc3 = register_module("fc3", torch::nn::Linear(HIDDEN_SIZE, OUTPUT_SIZE));
    }
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = fc3->forward(x);
        return x;
    }
};
TORCH_MODULE(QNet);

// Simple bitmap font for digits and letters (5x7 pixels each)
const uint8_t FONT_DATA[128][7] = {
    ['0'] = {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E},
    ['1'] = {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E},
    ['2'] = {0x0E,0x11,0x01,0x02,0x04,0x08,0x1F},
    ['3'] = {0x1F,0x02,0x04,0x02,0x01,0x11,0x0E},
    ['4'] = {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02},
    ['5'] = {0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E},
    ['6'] = {0x06,0x08,0x10,0x1E,0x11,0x11,0x0E},
    ['7'] = {0x1F,0x01,0x02,0x04,0x08,0x08,0x08},
    ['8'] = {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E},
    ['9'] = {0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C},
    ['.'] = {0x00,0x00,0x00,0x00,0x00,0x0C,0x0C},
    [':'] = {0x00,0x0C,0x0C,0x00,0x0C,0x0C,0x00},
    ['-'] = {0x00,0x00,0x00,0x1F,0x00,0x00,0x00},
    ['+'] = {0x00,0x04,0x04,0x1F,0x04,0x04,0x00},
    ['%'] = {0x18,0x19,0x02,0x04,0x08,0x13,0x03},
    ['/'] = {0x01,0x02,0x02,0x04,0x08,0x08,0x10},
    ['A'] = {0x0E,0x11,0x11,0x1F,0x11,0x11,0x11},
    ['B'] = {0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E},
    ['C'] = {0x0E,0x11,0x10,0x10,0x10,0x11,0x0E},
    ['D'] = {0x1C,0x12,0x11,0x11,0x11,0x12,0x1C},
    ['E'] = {0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F},
    ['F'] = {0x1F,0x10,0x10,0x1E,0x10,0x10,0x10},
    ['G'] = {0x0E,0x11,0x10,0x17,0x11,0x11,0x0F},
    ['H'] = {0x11,0x11,0x11,0x1F,0x11,0x11,0x11},
    ['I'] = {0x0E,0x04,0x04,0x04,0x04,0x04,0x0E},
    ['J'] = {0x07,0x02,0x02,0x02,0x02,0x12,0x0C},
    ['K'] = {0x11,0x12,0x14,0x18,0x14,0x12,0x11},
    ['L'] = {0x10,0x10,0x10,0x10,0x10,0x10,0x1F},
    ['M'] = {0x11,0x1B,0x15,0x15,0x11,0x11,0x11},
    ['N'] = {0x11,0x11,0x19,0x15,0x13,0x11,0x11},
    ['O'] = {0x0E,0x11,0x11,0x11,0x11,0x11,0x0E},
    ['P'] = {0x1E,0x11,0x11,0x1E,0x10,0x10,0x10},
    ['Q'] = {0x0E,0x11,0x11,0x11,0x15,0x12,0x0D},
    ['R'] = {0x1E,0x11,0x11,0x1E,0x14,0x12,0x11},
    ['S'] = {0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E},
    ['T'] = {0x1F,0x04,0x04,0x04,0x04,0x04,0x04},
    ['U'] = {0x11,0x11,0x11,0x11,0x11,0x11,0x0E},
    ['V'] = {0x11,0x11,0x11,0x11,0x11,0x0A,0x04},
    ['W'] = {0x11,0x11,0x11,0x15,0x15,0x15,0x0A},
    ['X'] = {0x11,0x11,0x0A,0x04,0x0A,0x11,0x11},
    ['Y'] = {0x11,0x11,0x11,0x0A,0x04,0x04,0x04},
    ['Z'] = {0x1F,0x01,0x02,0x04,0x08,0x10,0x1F},
    ['a'] = {0x00,0x00,0x0E,0x01,0x0F,0x11,0x0F},
    ['b'] = {0x10,0x10,0x16,0x19,0x11,0x11,0x1E},
    ['c'] = {0x00,0x00,0x0E,0x10,0x10,0x11,0x0E},
    ['d'] = {0x01,0x01,0x0D,0x13,0x11,0x11,0x0F},
    ['e'] = {0x00,0x00,0x0E,0x11,0x1F,0x10,0x0E},
    ['f'] = {0x06,0x09,0x08,0x1C,0x08,0x08,0x08},
    ['g'] = {0x00,0x0F,0x11,0x11,0x0F,0x01,0x0E},
    ['h'] = {0x10,0x10,0x16,0x19,0x11,0x11,0x11},
    ['i'] = {0x04,0x00,0x0C,0x04,0x04,0x04,0x0E},
    ['j'] = {0x02,0x00,0x06,0x02,0x02,0x12,0x0C},
    ['k'] = {0x10,0x10,0x12,0x14,0x18,0x14,0x12},
    ['l'] = {0x0C,0x04,0x04,0x04,0x04,0x04,0x0E},
    ['m'] = {0x00,0x00,0x1A,0x15,0x15,0x11,0x11},
    ['n'] = {0x00,0x00,0x16,0x19,0x11,0x11,0x11},
    ['o'] = {0x00,0x00,0x0E,0x11,0x11,0x11,0x0E},
    ['p'] = {0x00,0x00,0x1E,0x11,0x1E,0x10,0x10},
    ['q'] = {0x00,0x00,0x0D,0x13,0x0F,0x01,0x01},
    ['r'] = {0x00,0x00,0x16,0x19,0x10,0x10,0x10},
    ['s'] = {0x00,0x00,0x0E,0x10,0x0E,0x01,0x1E},
    ['t'] = {0x08,0x08,0x1C,0x08,0x08,0x09,0x06},
    ['u'] = {0x00,0x00,0x11,0x11,0x11,0x13,0x0D},
    ['v'] = {0x00,0x00,0x11,0x11,0x11,0x0A,0x04},
    ['w'] = {0x00,0x00,0x11,0x11,0x15,0x15,0x0A},
    ['x'] = {0x00,0x00,0x11,0x0A,0x04,0x0A,0x11},
    ['y'] = {0x00,0x00,0x11,0x11,0x0F,0x01,0x0E},
    ['z'] = {0x00,0x00,0x1F,0x02,0x04,0x08,0x1F},
    [' '] = {0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    ['['] = {0x0E,0x08,0x08,0x08,0x08,0x08,0x0E},
    [']'] = {0x0E,0x02,0x02,0x02,0x02,0x02,0x0E},
    ['('] = {0x02,0x04,0x08,0x08,0x08,0x04,0x02},
    [')'] = {0x08,0x04,0x02,0x02,0x02,0x04,0x08},
    ['<'] = {0x02,0x04,0x08,0x10,0x08,0x04,0x02},
    ['>'] = {0x08,0x04,0x02,0x01,0x02,0x04,0x08},
    ['='] = {0x00,0x00,0x1F,0x00,0x1F,0x00,0x00},
    ['|'] = {0x04,0x04,0x04,0x04,0x04,0x04,0x04},
};

class SnakeGame {
public:
    SnakeGame(bool render);
    ~SnakeGame();
    bool init();
    void train(int episodes);

private:
    void reset();
    int serializeDirection(const int signedDir) const;
    int unpackDirection(const int unsignedDir) const;
    float step(int action, bool shouldPrint);
    void render_all(const vec_t& state, int episode, int ep_score);
    void render_game(int x_off, int y_off);
    void render_network(int x_off, int y_off, int w, int h);
    void render_network_dynamic(const vec_t& state, int x_off, int y_off, int w, int h);
    void render_stats(int x_off, int y_off, int w, int h, int episode, int ep_score);
    void draw_text(const std::string& text, int x, int y, SDL_Color color, int scale = 1);
    void spawn_food();
    bool inside_grid(const Point &p) const;
    bool snake_contains(const Point &p) const;
    vec_t get_state() const;
    int select_action(const vec_t& state);
    void train_step();

    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;

    std::deque<Point> snake;
    Dir dir = Dir::RIGHT;
    Point food{0,0};
    bool game_over = false;
    int score = 0;
    int steps_without_food = 0;

    std::mt19937 rng;

    QNet q_network;
    QNet target_network;
    torch::optim::Adam optimizer;

    std::deque<Experience> replay_buffer;
    float epsilon = EPSILON_START;
    bool render_mode;

    int episode_count = 0;
    int total_steps = 0;

    // Stats tracking
    std::vector<float> score_history;
    std::vector<float> avg_score_history;
    std::vector<float> epsilon_history;
    int current_max_score = 0;
    float current_avg = 0;

    // Speed control
    int game_speed = 15;  // FPS (adjustable with +/-)
    int train_speed = 1;  // 1 = normal, higher = faster (skip renders)

    // Adjustable training parameters
    TrainingParams params;
    int selected_param = 0;  // Which parameter is selected for adjustment
};

SnakeGame::SnakeGame(bool render)
: render_mode(render),
  q_network(QNet()),
  target_network(QNet()),
  optimizer(q_network->parameters(), torch::optim::AdamOptions(INITIAL_LEARNING_RATE))
{
    rng.seed((unsigned)std::chrono::system_clock::now().time_since_epoch().count());

    torch::NoGradGuard no_grad;
    for(size_t i=0;i<q_network->parameters().size();i++){
        target_network->parameters()[i].copy_(q_network->parameters()[i]);
    }
}

SnakeGame::~SnakeGame(){
    if(renderer) SDL_DestroyRenderer(renderer);
    if(window) SDL_DestroyWindow(window);
    if(render_mode) SDL_Quit();
}

bool SnakeGame::init(){
    if(render_mode){
        if(SDL_Init(SDL_INIT_VIDEO)!=0){
            std::cerr << "SDL_Init failed: " << SDL_GetError() << "\n";
            return false;
        }
        window = SDL_CreateWindow("Snake AI - Neural Network Training",
                                  SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                  WINDOW_W, WINDOW_H, SDL_WINDOW_SHOWN);
        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    }
    reset();
    return true;
}

void SnakeGame::reset(){
    snake.clear();
    int x = COLS/2, y = ROWS/2;
    snake.push_back({x,y});
    snake.push_back({x-1,y});
    snake.push_back({x-2,y});
    dir = Dir::RIGHT;
    score = 0;
    game_over = false;
    steps_without_food = 0;
    spawn_food();
}

void SnakeGame::spawn_food(){
    std::uniform_int_distribution<int> dx(0,COLS-1), dy(0,ROWS-1);
    Point p;
    do{
        p.x = dx(rng);
        p.y = dy(rng);
    } while(snake_contains(p));
    food = p;
}

int SnakeGame::serializeDirection(const int signedDir) const{
    switch(signedDir){
        case -1: return 0;
        case 1: return 1;
        case -2: return 2;
        case 2: return 3;
    }
    return 0;
}

int SnakeGame::unpackDirection(const int unsignedDir) const{
    switch(unsignedDir){
        case 0: return -1;
        case 1: return 1;
        case 2: return -2;
        case 3: return 2;
    }
    return 0;
}

bool SnakeGame::inside_grid(const Point& p) const{
    return p.x >= 0 && p.x < COLS && p.y >= 0 && p.y < ROWS;
}

bool SnakeGame::snake_contains(const Point& p) const{
    for(auto& s : snake) if(s == p) return true;
    return false;
}

vec_t SnakeGame::get_state() const{
    vec_t s(INPUT_SIZE, 0.0f);
    Point h = snake.front();

    // Danger detection (immediate) - is there wall or body in each direction?
    // UP
    Point up{h.x, h.y - 1};
    s[0] = (!inside_grid(up) || snake_contains(up)) ? 1.0f : 0.0f;
    // DOWN
    Point down{h.x, h.y + 1};
    s[1] = (!inside_grid(down) || snake_contains(down)) ? 1.0f : 0.0f;
    // LEFT
    Point left{h.x - 1, h.y};
    s[2] = (!inside_grid(left) || snake_contains(left)) ? 1.0f : 0.0f;
    // RIGHT
    Point right{h.x + 1, h.y};
    s[3] = (!inside_grid(right) || snake_contains(right)) ? 1.0f : 0.0f;

    // Food direction (relative to head) - ALWAYS visible
    s[4] = (food.y < h.y) ? 1.0f : 0.0f;  // Food is UP
    s[5] = (food.y > h.y) ? 1.0f : 0.0f;  // Food is DOWN
    s[6] = (food.x < h.x) ? 1.0f : 0.0f;  // Food is LEFT
    s[7] = (food.x > h.x) ? 1.0f : 0.0f;  // Food is RIGHT

    // Normalized distance to food
    s[8] = (h.x - food.x) / (float)COLS;  // X distance (-1 to 1)
    s[9] = (h.y - food.y) / (float)ROWS;  // Y distance (-1 to 1)

    // Current direction (one-hot)
    s[10] = (dir == Dir::UP) ? 1.0f : 0.0f;
    s[11] = (dir == Dir::DOWN) ? 1.0f : 0.0f;
    s[12] = (dir == Dir::LEFT) ? 1.0f : 0.0f;
    s[13] = (dir == Dir::RIGHT) ? 1.0f : 0.0f;

    // Snake length normalized
    s[14] = snake.size() / (float)(COLS * ROWS);

    // Steps without food (urgency)
    s[15] = std::min(1.0f, steps_without_food / 100.0f);

    return s;
}

int SnakeGame::select_action(const vec_t& state){
    std::uniform_real_distribution<float> d(0.0f, 1.0f);
    if(d(rng) < epsilon){
        std::uniform_int_distribution<int> a(0, 3);
        return a(rng);
    }
    torch::Tensor inp = torch::tensor(state).reshape({1, INPUT_SIZE});
    torch::Tensor out = q_network->forward(inp);
    return out.argmax(1).item<int>();
}

float SnakeGame::step(int action, bool shouldPrint){
    if(game_over) return 0;
    Dir old = dir;
    dir = (Dir)unpackDirection(action);
    Point h = snake.front();
    Point nh = h;

    if((int)old + (int)dir == 0) dir = old;

    if(dir == Dir::UP) nh.y--;
    else if(dir == Dir::DOWN) nh.y++;
    else if(dir == Dir::LEFT) nh.x--;
    else nh.x++;

    // Calculate distance to food before and after move
    int old_dist = std::abs(h.x - food.x) + std::abs(h.y - food.y);
    int new_dist = std::abs(nh.x - food.x) + std::abs(nh.y - food.y);

    float reward = 0.0f;
    steps_without_food++;

    if(!inside_grid(nh)){
        game_over = true;
        return params.penalty_death;
    }
    if(snake_contains(nh)){
        game_over = true;
        return params.penalty_death;
    }
    if(steps_without_food > 50 + 10 * (int)snake.size()){  // Shorter timeout for smaller grid
        game_over = true;
        return params.penalty_death * 0.5f;
    }

    bool ate = (nh == food);
    snake.push_front(nh);
    if(!ate) {
        snake.pop_back();
        // Reward for moving toward food, penalty for moving away
        if(new_dist < old_dist) {
            reward = params.reward_closer;
        } else {
            reward = params.penalty_away;
        }
    } else {
        reward = params.reward_food;
        score++;
        steps_without_food = 0;
        spawn_food();
    }
    return reward;
}

void SnakeGame::train_step(){
    if((int)replay_buffer.size() < params.batch_size) return;
    std::uniform_int_distribution<size_t> d(0, replay_buffer.size()-1);

    std::vector<torch::Tensor> states, targets;
    for(int i = 0; i < params.batch_size; i++){
        const auto& e = replay_buffer[d(rng)];
        torch::Tensor s = torch::tensor(e.state);
        torch::Tensor q = q_network->forward(s);
        torch::Tensor t = q.detach().clone();

        if(e.done) t[e.action] = e.reward;
        else {
            torch::Tensor ns = torch::tensor(e.next_state);
            auto nq = target_network->forward(ns);
            float mx = nq.max().item<float>();
            t[e.action] = e.reward + params.gamma * mx;
        }
        states.push_back(s);
        targets.push_back(t);
    }

    torch::Tensor batch_s = torch::stack(states);
    torch::Tensor batch_t = torch::stack(targets);

    optimizer.zero_grad();
    torch::Tensor out = q_network->forward(batch_s);
    torch::Tensor loss = torch::mse_loss(out, batch_t);
    loss.backward();
    optimizer.step();

    static int c = 0; c++;
    if(c % 50 == 0){
        torch::NoGradGuard no_grad;
        for(size_t i = 0; i < q_network->parameters().size(); i++){
            target_network->parameters()[i].copy_(q_network->parameters()[i]);
        }
    }
}

void SnakeGame::draw_text(const std::string& text, int x, int y, SDL_Color color, int scale){
    SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
    int cursor_x = x;
    for(char c : text){
        if(c >= 0 && c < 128){
            const uint8_t* glyph = FONT_DATA[(int)c];
            for(int row = 0; row < 7; row++){
                for(int col = 0; col < 5; col++){
                    if(glyph[row] & (0x10 >> col)){
                        SDL_Rect r = {cursor_x + col * scale, y + row * scale, scale, scale};
                        SDL_RenderFillRect(renderer, &r);
                    }
                }
            }
        }
        cursor_x += 6 * scale;
    }
}

void SnakeGame::render_game(int x_off, int y_off){
    // Background
    SDL_SetRenderDrawColor(renderer, 25, 25, 25, 255);
    SDL_Rect bg = {x_off, y_off, GAME_SIZE, GAME_SIZE};
    SDL_RenderFillRect(renderer, &bg);

    // Grid
    SDL_SetRenderDrawColor(renderer, 40, 40, 40, 255);
    for(int gx = 0; gx <= COLS; gx++){
        SDL_RenderDrawLine(renderer, x_off + gx*CELL, y_off, x_off + gx*CELL, y_off + GAME_SIZE);
    }
    for(int gy = 0; gy <= ROWS; gy++){
        SDL_RenderDrawLine(renderer, x_off, y_off + gy*CELL, x_off + GAME_SIZE, y_off + gy*CELL);
    }

    // Food
    SDL_SetRenderDrawColor(renderer, 200, 50, 50, 255);
    SDL_Rect fr = {x_off + food.x*CELL + 2, y_off + food.y*CELL + 2, CELL-4, CELL-4};
    SDL_RenderFillRect(renderer, &fr);

    // Snake
    bool first = true;
    for(auto& s : snake){
        SDL_Rect r = {x_off + s.x*CELL + 1, y_off + s.y*CELL + 1, CELL-2, CELL-2};
        if(first){
            SDL_SetRenderDrawColor(renderer, 90, 200, 90, 255);
            first = false;
        } else {
            SDL_SetRenderDrawColor(renderer, 30, 160, 30, 255);
        }
        SDL_RenderFillRect(renderer, &r);
    }

    // Score on game
    std::string score_text = "Score: " + std::to_string(score);
    draw_text(score_text, x_off + 10, y_off + 10, {255, 255, 255, 255}, 2);
}

void SnakeGame::render_stats(int x_off, int y_off, int w, int h, int episode, int ep_score){
    // Background
    SDL_SetRenderDrawColor(renderer, 30, 30, 40, 255);
    SDL_Rect bg = {x_off, y_off, w, h};
    SDL_RenderFillRect(renderer, &bg);

    int margin = 15;
    int graph_h = 65;
    int label_h = 15;

    // Title
    draw_text("TRAINING STATISTICS", x_off + margin, y_off + 8, {255, 255, 255, 255}, 2);

    // Current stats row
    int stats_y = y_off + 35;
    std::ostringstream oss;
    oss << "Episode: " << episode << "  Max: " << current_max_score << "  Avg: " << std::fixed << std::setprecision(1) << current_avg;
    draw_text(oss.str(), x_off + margin, stats_y, {200, 200, 200, 255}, 1);

    oss.str(""); oss.clear();
    oss << "Epsilon: " << std::fixed << std::setprecision(3) << epsilon << "  Speed: " << game_speed << " FPS  Train: " << train_speed << "x";
    draw_text(oss.str(), x_off + margin, stats_y + 12, {150, 150, 150, 255}, 1);

    // Parameter controls panel
    int ctrl_x = x_off + w/2 + 20;
    int ctrl_y = y_off + 8;
    draw_text("CONTROLS [Up/Down=select, Left/Right=adjust]", ctrl_x, ctrl_y, {255, 200, 100, 255}, 1);
    ctrl_y += 18;

    const char* param_names[] = {
        "Game Speed (FPS)",
        "Train Speed (x)",
        "Learning Rate",
        "Gamma (discount)",
        "Epsilon Decay",
        "Batch Size",
        "Reward: Food",
        "Reward: Closer",
        "Penalty: Away",
        "Penalty: Death"
    };
    float* param_values[] = {nullptr, nullptr, &params.learning_rate, &params.gamma, &params.epsilon_decay, nullptr, &params.reward_food, &params.reward_closer, &params.penalty_away, &params.penalty_death};
    int* param_ints[] = {&game_speed, &train_speed, nullptr, nullptr, nullptr, &params.batch_size, nullptr, nullptr, nullptr, nullptr};

    for(int i = 0; i < 10; i++){
        SDL_Color col = (i == selected_param) ? SDL_Color{100, 255, 100, 255} : SDL_Color{150, 150, 150, 255};
        std::string prefix = (i == selected_param) ? "> " : "  ";
        oss.str(""); oss.clear();
        oss << prefix << param_names[i] << ": ";
        if(param_values[i]) oss << std::fixed << std::setprecision(4) << *param_values[i];
        else if(param_ints[i]) oss << *param_ints[i];
        draw_text(oss.str(), ctrl_x, ctrl_y + i * 12, col, 1);
    }

    // Helper to draw graph
    auto draw_graph = [&](const std::vector<float>& data, int gy, SDL_Color line_color, float max_val, const std::string& label){
        int gx = x_off + margin;
        int gw = w - 2 * margin;

        // Label
        draw_text(label, gx, gy, {180, 180, 180, 255}, 1);
        gy += label_h;

        // Graph background
        SDL_SetRenderDrawColor(renderer, 40, 40, 55, 255);
        SDL_Rect gbg = {gx, gy, gw, graph_h};
        SDL_RenderFillRect(renderer, &gbg);

        // Border
        SDL_SetRenderDrawColor(renderer, 70, 70, 90, 255);
        SDL_RenderDrawRect(renderer, &gbg);

        // Grid lines
        SDL_SetRenderDrawColor(renderer, 50, 50, 65, 255);
        for(int i = 1; i < 4; i++){
            int ly = gy + (graph_h * i / 4);
            SDL_RenderDrawLine(renderer, gx, ly, gx + gw, ly);
        }

        // Data line
        if(data.size() > 1){
            SDL_SetRenderDrawColor(renderer, line_color.r, line_color.g, line_color.b, 255);
            float x_step = (float)gw / (data.size() - 1);
            for(size_t i = 1; i < data.size(); i++){
                float v1 = std::min(data[i-1] / max_val, 1.0f);
                float v2 = std::min(data[i] / max_val, 1.0f);
                int x1 = gx + (int)((i-1) * x_step);
                int x2 = gx + (int)(i * x_step);
                int y1 = gy + graph_h - (int)(v1 * (graph_h - 4)) - 2;
                int y2 = gy + graph_h - (int)(v2 * (graph_h - 4)) - 2;
                SDL_RenderDrawLine(renderer, x1, y1, x2, y2);
            }
        }

        // Current value indicator
        if(!data.empty()){
            float v = std::min(data.back() / max_val, 1.0f);
            int dot_y = gy + graph_h - (int)(v * (graph_h - 4)) - 2;
            SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
            SDL_Rect dot = {gx + gw - 4, dot_y - 2, 5, 5};
            SDL_RenderFillRect(renderer, &dot);

            // Value text
            oss.str(""); oss.clear();
            oss << std::fixed << std::setprecision(1) << data.back();
            draw_text(oss.str(), gx + gw - 40, gy + 5, {255, 255, 255, 255}, 1);
        }
    };

    // Three graphs
    float max_score_val = std::max(50.0f, (float)current_max_score * 1.2f);
    int g1_y = y_off + 75;
    int g2_y = g1_y + graph_h + label_h + 15;
    int g3_y = g2_y + graph_h + label_h + 15;

    draw_graph(score_history, g1_y, {100, 220, 100, 255}, max_score_val, "SCORE PER EPISODE (green)");
    draw_graph(avg_score_history, g2_y, {100, 150, 255, 255}, max_score_val, "AVERAGE SCORE /10 eps (blue)");
    draw_graph(epsilon_history, g3_y, {255, 200, 100, 255}, 1.0f, "EXPLORATION RATE (orange)");
}

void SnakeGame::render_network(int x_off, int y_off, int w, int h){
    // Background
    SDL_SetRenderDrawColor(renderer, 20, 20, 30, 255);
    SDL_Rect bg = {x_off, y_off, w, h};
    SDL_RenderFillRect(renderer, &bg);

    draw_text("Q-NETWORK WEIGHTS", x_off + 10, y_off + 10, {255, 255, 255, 255}, 1);

    const int padding_x = 30;
    const int padding_y = 35;
    const int layers[] = {INPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, OUTPUT_SIZE};
    const int num_layers = 4;

    std::vector<std::vector<SDL_Point>> neuron_pos(num_layers);
    for(int l = 0; l < num_layers; l++){
        int n = layers[l];
        int max_display = std::min(n, 20);  // Limit displayed neurons
        neuron_pos[l].resize(max_display);
        for(int i = 0; i < max_display; i++){
            neuron_pos[l][i] = {
                x_off + padding_x + l * (w - 2*padding_x) / (num_layers-1),
                y_off + padding_y + i * (h - 2*padding_y) / std::max(1, max_display-1)
            };
        }
    }

    auto draw_layer_connections = [&](torch::nn::Linear layer, int l){
        auto wt = layer->weight.detach();
        int in_n = std::min((int)wt.size(1), (int)neuron_pos[l].size());
        int out_n = std::min((int)wt.size(0), (int)neuron_pos[l+1].size());
        for(int i = 0; i < out_n; i++){
            for(int j = 0; j < in_n; j++){
                float val = wt[i][j].item<float>();
                int r = val > 0 ? std::min(255, int(val * 200)) : 0;
                int g = val < 0 ? std::min(255, int(-val * 200)) : 0;
                SDL_SetRenderDrawColor(renderer, r, g, 30, 100);
                SDL_RenderDrawLine(renderer,
                    neuron_pos[l][j].x, neuron_pos[l][j].y,
                    neuron_pos[l+1][i].x, neuron_pos[l+1][i].y);
            }
        }
    };

    draw_layer_connections(q_network->fc1, 0);
    draw_layer_connections(q_network->fc2, 1);
    draw_layer_connections(q_network->fc3, 2);

    // Neurons
    for(int l = 0; l < num_layers; l++){
        for(auto& p : neuron_pos[l]){
            SDL_SetRenderDrawColor(renderer, 200, 200, 200, 255);
            SDL_Rect r = {p.x - 3, p.y - 3, 6, 6};
            SDL_RenderFillRect(renderer, &r);
        }
    }

    // Layer labels
    const char* layer_names[] = {"Input", "Hidden1", "Hidden2", "Output"};
    for(int l = 0; l < num_layers; l++){
        draw_text(layer_names[l], neuron_pos[l][0].x - 15, y_off + h - 20, {150, 150, 150, 255}, 1);
    }
}

void SnakeGame::render_network_dynamic(const vec_t& state, int x_off, int y_off, int w, int h){
    // Background
    SDL_SetRenderDrawColor(renderer, 15, 15, 25, 255);
    SDL_Rect bg = {x_off, y_off, w, h};
    SDL_RenderFillRect(renderer, &bg);

    draw_text("LIVE NETWORK ACTIVITY", x_off + 10, y_off + 10, {255, 255, 255, 255}, 1);

    const int padding_x = 30;
    const int padding_y = 35;
    const int layers[] = {INPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, OUTPUT_SIZE};
    const int num_layers = 4;

    std::vector<std::vector<SDL_Point>> neuron_pos(num_layers);
    for(int l = 0; l < num_layers; l++){
        int n = layers[l];
        int max_display = std::min(n, 20);
        neuron_pos[l].resize(max_display);
        for(int i = 0; i < max_display; i++){
            neuron_pos[l][i] = {
                x_off + padding_x + l * (w - 2*padding_x) / (num_layers-1),
                y_off + padding_y + i * (h - 2*padding_y) / std::max(1, max_display-1)
            };
        }
    }

    torch::Tensor inp = torch::tensor(state).reshape({1, INPUT_SIZE});
    torch::Tensor h1 = torch::relu(q_network->fc1->forward(inp));
    torch::Tensor h2 = torch::relu(q_network->fc2->forward(h1));
    torch::Tensor out = q_network->fc3->forward(h2);
    std::vector<torch::Tensor> activations = {inp.flatten(), h1.flatten(), h2.flatten(), out.flatten()};

    // Draw connections with activation intensity
    auto draw_connections = [&](torch::nn::Linear layer, int l){
        auto wt = layer->weight.detach();
        int in_n = std::min((int)wt.size(1), (int)neuron_pos[l].size());
        int out_n = std::min((int)wt.size(0), (int)neuron_pos[l+1].size());
        for(int i = 0; i < out_n; i++){
            for(int j = 0; j < in_n; j++){
                float val = wt[i][j].item<float>() * activations[l][j].item<float>();
                float intensity = std::tanh(std::abs(val)) * 255.0f;
                SDL_SetRenderDrawColor(renderer, 0, 0, std::min(255, (int)(intensity * 2)), 150);
                SDL_RenderDrawLine(renderer,
                    neuron_pos[l][j].x, neuron_pos[l][j].y,
                    neuron_pos[l+1][i].x, neuron_pos[l+1][i].y);
            }
        }
    };

    draw_connections(q_network->fc1, 0);
    draw_connections(q_network->fc2, 1);
    draw_connections(q_network->fc3, 2);

    // Neurons with activation colors
    for(int l = 0; l < num_layers; l++){
        int max_display = std::min(layers[l], 20);
        for(int i = 0; i < max_display; i++){
            float act = activations[l][i].item<float>();
            int intensity = std::min(255, std::max(0, int(std::abs(act) * 200)));
            SDL_SetRenderDrawColor(renderer, intensity, intensity, 255, 255);
            SDL_Rect r = {neuron_pos[l][i].x - 4, neuron_pos[l][i].y - 4, 8, 8};
            SDL_RenderFillRect(renderer, &r);
        }
    }

    // Output labels
    const char* actions[] = {"UP", "DOWN", "LEFT", "RIGHT"};
    for(int i = 0; i < 4; i++){
        float q_val = out[0][i].item<float>();
        std::ostringstream oss;
        oss << actions[i] << ": " << std::fixed << std::setprecision(2) << q_val;
        int ly = neuron_pos[3][i].y - 3;
        SDL_Color col = (i == out.argmax(1).item<int>()) ? SDL_Color{100, 255, 100, 255} : SDL_Color{150, 150, 150, 255};
        draw_text(oss.str(), neuron_pos[3][i].x + 10, ly, col, 1);
    }
}

void SnakeGame::render_all(const vec_t& state, int episode, int ep_score){
    if(!render_mode) return;

    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);

    // Layout:
    // [Game 400x400] [Stats 500x400]
    // [Network 450x350] [Dynamic 450x350]

    render_game(0, 0);
    render_stats(GAME_SIZE, 0, STATS_W, GAME_SIZE, episode, ep_score);
    render_network(0, GAME_SIZE, WINDOW_W/2, NET_VIS_H);
    render_network_dynamic(state, WINDOW_W/2, GAME_SIZE, WINDOW_W/2, NET_VIS_H);

    // Border lines
    SDL_SetRenderDrawColor(renderer, 60, 60, 80, 255);
    SDL_RenderDrawLine(renderer, GAME_SIZE, 0, GAME_SIZE, GAME_SIZE);
    SDL_RenderDrawLine(renderer, 0, GAME_SIZE, WINDOW_W, GAME_SIZE);
    SDL_RenderDrawLine(renderer, WINDOW_W/2, GAME_SIZE, WINDOW_W/2, WINDOW_H);

    SDL_RenderPresent(renderer);
}

void SnakeGame::train(int episodes){
    int total_score = 0;
    int max_score = 0;

    for(int ep = 0; ep < episodes; ep++){
        episode_count = ep;
        reset();
        auto state = get_state();
        int ep_score = 0;
        int steps = 0;

        while(!game_over && steps < 1000){
            int a = select_action(state);
            float r = step(a, false);
            auto ns = get_state();

            replay_buffer.push_back({state, a, r, ns, game_over});
            if((int)replay_buffer.size() > params.replay_buffer_size) replay_buffer.pop_front();

            if((int)replay_buffer.size() >= params.batch_size)
                train_step();

            state = ns;
            ep_score = score;
            steps++;
            total_steps++;

            // Handle events
            SDL_Event e;
            while(SDL_PollEvent(&e)){
                if(e.type == SDL_QUIT) return;
                if(e.type == SDL_KEYDOWN){
                    switch(e.key.keysym.sym){
                        case SDLK_UP:
                            selected_param = (selected_param + 9) % 10;
                            break;
                        case SDLK_DOWN:
                            selected_param = (selected_param + 1) % 10;
                            break;
                        case SDLK_LEFT:
                        case SDLK_RIGHT: {
                            float mult = (e.key.keysym.sym == SDLK_RIGHT) ? 1.1f : 0.9f;
                            int delta = (e.key.keysym.sym == SDLK_RIGHT) ? 1 : -1;
                            switch(selected_param){
                                case 0: game_speed = std::clamp(game_speed + delta * 5, 5, 120); break;
                                case 1: train_speed = std::clamp(train_speed + delta, 1, 100); break;
                                case 2: params.learning_rate = std::clamp(params.learning_rate * mult, 0.00001f, 0.1f); break;
                                case 3: params.gamma = std::clamp(params.gamma + delta * 0.01f, 0.5f, 0.999f); break;
                                case 4: params.epsilon_decay = std::clamp(params.epsilon_decay + delta * 0.001f, 0.9f, 0.9999f); break;
                                case 5: params.batch_size = std::clamp(params.batch_size + delta * 16, 16, 512); break;
                                case 6: params.reward_food = std::clamp(params.reward_food + delta * 1.0f, 1.0f, 100.0f); break;
                                case 7: params.reward_closer = std::clamp(params.reward_closer + delta * 0.05f, 0.0f, 2.0f); break;
                                case 8: params.penalty_away = std::clamp(params.penalty_away + delta * 0.05f, -2.0f, 0.0f); break;
                                case 9: params.penalty_death = std::clamp(params.penalty_death + delta * 1.0f, -100.0f, -1.0f); break;
                            }
                            break;
                        }
                        case SDLK_r:
                            train_speed = 1;
                            game_speed = 15;
                            params = TrainingParams();  // Reset to defaults
                            break;
                        case SDLK_SPACE:
                            epsilon = 1.0f;  // Reset exploration
                            break;
                    }
                }
            }

            // Render based on train_speed
            bool should_render = (ep % train_speed == 0) || force_render_next;
            if(should_render && render_mode){
                Uint32 frame_start = SDL_GetTicks();
                render_all(state, ep, ep_score);
                Uint32 frame_time = SDL_GetTicks() - frame_start;
                int frame_delay = 1000 / game_speed;
                if(frame_time < (Uint32)frame_delay){
                    SDL_Delay(frame_delay - frame_time);
                }
            }
        }

        // Quick transition - no delay between episodes
        total_score += ep_score;
        max_score = std::max(max_score, ep_score);
        current_max_score = max_score;
        epsilon = std::max(params.epsilon_end, epsilon * params.epsilon_decay);

        // Track score history
        score_history.push_back((float)ep_score);
        if(score_history.size() > 200) score_history.erase(score_history.begin());

        if(ep % 10 == 0){
            float avg = total_score / 10.0f;
            current_avg = avg;
            avg_score_history.push_back(avg);
            epsilon_history.push_back(epsilon);
            if(avg_score_history.size() > 200) avg_score_history.erase(avg_score_history.begin());
            if(epsilon_history.size() > 200) epsilon_history.erase(epsilon_history.begin());

            std::cout << "Episode " << ep
                      << " | Avg: " << avg
                      << " | Max: " << max_score
                      << " | Eps: " << std::fixed << std::setprecision(3) << epsilon
                      << " | Speed: " << train_speed << "x\n";
            total_score = 0;
        }
    }
}

int main(int argc, char** argv){
    std::signal(SIGINT, signal_handler);
    bool render = true;
    SnakeGame g(render);
    if(!g.init()) return 1;
    std::cout << "Snake AI Training - Controls:\n";
    std::cout << "  Up/Down   : Select parameter\n";
    std::cout << "  Left/Right: Adjust selected parameter\n";
    std::cout << "  R         : Reset all to defaults\n";
    std::cout << "  Space     : Reset exploration (epsilon=1)\n\n";
    std::cout << "Starting training...\n";
    g.train(100000);
    return 0;
}
