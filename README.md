# Snake Neural Network

A reinforcement learning snake game that trains itself using Deep Q-Learning (DQN). Watch a neural network learn to play snake in real-time with full visualization of the training process.

<img width="1200" height="824" alt="Screenshot 2026-01-02 at 22 22 00" src="https://github.com/user-attachments/assets/ef718777-3569-47a1-8d2d-9c7cfe319f4f" />

## Features

- **Real-time Training Visualization** - Single unified window showing:
  - Snake game (12x12 grid)
  - Training statistics with live graphs
  - Neural network weights visualization
  - Live network activity with Q-values

- **Interactive Parameter Tuning** - Adjust all training parameters on-the-fly:
  - Learning rate, gamma, epsilon decay
  - Reward/penalty values
  - Batch size, game speed, training speed

- **Deep Q-Network Architecture**:
  - 16 input neurons (danger detection, food direction, current state)
  - 2 hidden layers (128 neurons each)
  - 4 output neurons (UP, DOWN, LEFT, RIGHT)

## Requirements

- C++17 compiler
- SDL2
- LibTorch (PyTorch C++)

## Installation

### macOS (Homebrew)

```bash
# Install dependencies
brew install sdl2 pytorch

# Clone and build
git clone https://github.com/Ethan-Blesch/Snake-neural-network
cd Snake-neural-network

# Compile
g++ -std=c++17 snake.cpp -o snake_rl \
  -I/opt/homebrew/opt/pytorch/include \
  -I/opt/homebrew/opt/pytorch/include/torch/csrc/api/include \
  -I/opt/homebrew/include \
  -L/opt/homebrew/opt/pytorch/lib \
  -ltorch -lc10 -ltorch_cpu -lpthread \
  -Wl,-rpath,/opt/homebrew/opt/pytorch/lib \
  $(sdl2-config --cflags --libs) -w -O3

# Run
./snake_rl
```

### Linux

```bash
# Install SDL2
sudo apt install libsdl2-dev

# Download LibTorch from https://pytorch.org/get-started/locally/
# Extract to /usr/local/libtorch

# Compile
g++ -std=c++17 snake.cpp -o snake_rl \
  -I/usr/local/libtorch/include \
  -I/usr/local/libtorch/include/torch/csrc/api/include \
  -L/usr/local/libtorch/lib \
  -D_GLIBCXX_USE_CXX11_ABI=0 \
  -ltorch -lc10 -lpthread \
  -Wl,-rpath,/usr/local/libtorch/lib \
  $(sdl2-config --cflags --libs) -w -O3

./snake_rl
```

## Controls

| Key | Action |
|-----|--------|
| Up/Down | Select parameter |
| Left/Right | Adjust selected parameter |
| R | Reset all parameters to defaults |
| Space | Reset exploration (epsilon = 1.0) |

## Adjustable Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| Game Speed | Frames per second | 15 FPS |
| Train Speed | Skip renders to train faster | 1x |
| Learning Rate | Neural network learning rate | 0.001 |
| Gamma | Discount factor for future rewards | 0.99 |
| Epsilon Decay | Exploration decay rate | 0.998 |
| Batch Size | Training batch size | 128 |
| Reward: Food | Reward for eating food | 10.0 |
| Reward: Closer | Reward for moving toward food | 0.1 |
| Penalty: Away | Penalty for moving away from food | -0.15 |
| Penalty: Death | Penalty for dying | -10.0 |

## How It Works

The snake learns through **Deep Q-Learning**:

1. **State**: The snake observes 16 inputs including danger in each direction, food direction, distance to food, current direction, snake length, and steps without food.

2. **Action**: The network outputs Q-values for each direction (UP, DOWN, LEFT, RIGHT). The action with the highest Q-value is chosen.

3. **Reward Shaping**: The snake receives positive rewards for eating food and moving closer to it, and penalties for dying or moving away from food.

4. **Experience Replay**: Past experiences are stored and randomly sampled for training to break correlation between consecutive samples.

5. **Target Network**: A separate target network is used to stabilize training by providing consistent Q-value targets.

## Credits

Inspired by [Ethan-Blesch/Snake-neural-network](https://github.com/Ethan-Blesch/Snake-neural-network)

## License

MIT
