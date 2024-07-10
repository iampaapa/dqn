# Deep Q-Network (DQN) Module for Go

[![Go Reference](https://pkg.go.dev/badge/github.com/iampaapa/dqn.svg)](https://pkg.go.dev/github.com/iampaapa/dqn)

This module provides a flexible and efficient implementation of the Deep Q-Network (DQN) algorithm in Go. It's designed for industrial applications, offering robust reinforcement learning capabilities for complex decision-making tasks.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Example: Manufacturing Process Optimization](#example-manufacturing-process-optimization)
- [Module Structure](#module-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- Efficient Deep Q-Network implementation in pure Go
- Flexible neural network architecture with customizable activation functions
- Experience replay buffer for improved learning stability
- Epsilon-greedy exploration strategy
- Easy integration with custom environments
- Comparison utilities for classical reinforcement learning methods

## Installation

To use this module in your Go project, you can install it using `go get`:

```bash
go get github.com/iampaapa/dqn
```

Make sure you have Go installed (version 1.11+ for module support).

## Usage

Here's a basic example of how to use the DQN module:

```go
import (
    "github.com/iampaapa/dqn"
)

func main() {
    // Initialize a new DQN agent
    agent := dqn.NewDQN(
        inputSize,
        hiddenSize,
        outputSize,
        bufferSize,
        gamma,
        epsilon,
        learningRate,
        dqn.ReLU,
    )

    // Training loop
    for episode := 0; episode < numEpisodes; episode++ {
        state := environment.Reset()
        for !done {
            action := agent.EpsilonGreedyPolicy(state, numActions)
            nextState, reward, done := environment.Step(action)
            agent.Train(state, nextState, action, reward, done)
            state = nextState
        }
    }

    // Use the trained agent
    action := agent.EpsilonGreedyPolicy(state, numActions)
}
```

## Example: Manufacturing Process Optimization

We've included a comprehensive example of using this DQN module for manufacturing process optimization. This example demonstrates how to:

1. Create a simulated manufacturing environment
2. Implement and train a DQN agent
3. Compare DQN performance with classical Q-learning
4. Visualize the results

You can find this example in the `examples/manufacturing_optimization` directory.

To run the example:

```bash
cd examples/manufacturing_optimization
go run main.go
```

This will train both a DQN agent and a Q-learning agent, compare their performance, and generate a performance graph.

## Module Structure

The module consists of the following main components:

- `qnetwork.go`: Implements the neural network for Q-value approximation
- `replaybuffer.go`: Provides an experience replay buffer for improved learning stability
- `train.go`: Contains the core DQN algorithm and training loop
- `utils.go`: Offers utility functions for data normalization and other helper tasks

## Contributing

Contributions to this module are welcome! Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Commit your changes
4. Push to your fork and submit a pull request

Please make sure to update tests as appropriate and adhere to the existing coding style.

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

---

For more information or if you encounter any issues, please open an issue on the GitHub repository.
