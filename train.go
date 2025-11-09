// train.go
package dqn

import (
	"math/rand"
)

// DQN represents the Deep Q-Learning algorithm.
type DQN struct {
	qNetwork     *QNetwork
	replayBuffer *ReplayBuffer
	gamma        float64
	epsilon      float64
	learningRate float64
}

// NewDQN initializes a new DQN instance.
func NewDQN(inputSize, hiddenSize, outputSize, bufferSize int, gamma, epsilon, learningRate float64, activation Activation) *DQN {
	return &DQN{
		qNetwork:     NewQNetwork(inputSize, hiddenSize, outputSize, activation),
		replayBuffer: NewReplayBuffer(bufferSize),
		gamma:        gamma,
		epsilon:      epsilon,
		learningRate: learningRate,
	}
}

// Train trains the Q-network.
func (d *DQN) Train(state, nextState []float64, action, reward int, done bool) {
	nextQValues := d.qNetwork.Predict(nextState)
	maxNextQValue := Max(nextQValues)
	target := make([]float64, len(nextQValues))
	copy(target, nextQValues)
	target[action] = float64(reward)
	if !done {
		target[action] += d.gamma * maxNextQValue
	}

	currentQValues := d.qNetwork.Predict(state)
	// loss := d.qNetwork.Loss(currentQValues, target)

	d.qNetwork.Backward(state, currentQValues, target, d.learningRate)
}

// EpsilonGreedyPolicy selects an action using epsilon-greedy strategy.
func (d *DQN) EpsilonGreedyPolicy(state []float64, numActions int) int {
	if rand.Float64() < d.epsilon {
		return rand.Intn(numActions)
	}
	qValues := d.qNetwork.Predict(state)
	return Argmax(qValues)
}

// Helper functions

// Max returns the maximum value in a slice of float64
func Max(arr []float64) float64 {
	maxVal := arr[0]
	for _, val := range arr {
		if val > maxVal {
			maxVal = val
		}
	}
	return maxVal
}

// Argmax returns the index of the maximum value in a slice of float64
func Argmax(arr []float64) int {
	maxIdx := 0
	maxVal := arr[0]
	for i, val := range arr {
		if val > maxVal {
			maxIdx = i
			maxVal = val
		}
	}
	return maxIdx
}
