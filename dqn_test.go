// dqn_test.go
package dqn

import (
	"testing"
)

func TestQNetwork(t *testing.T) {
	qnet := NewQNetwork(4, 10, 2, ReLU)
	state := []float64{1, 2, 3, 4}
	qValues := qnet.Predict(state)
	if len(qValues) != 2 {
		t.Errorf("Expected 2 Q-values, got %d", len(qValues))
	}
}

func TestReplayBuffer(t *testing.T) {
	buffer := NewReplayBuffer(2)
	exp := Experience{State: []float64{1}, NextState: []float64{2}, Action: 1, Reward: 1, Done: false}
	buffer.Add(exp)
	buffer.Add(exp)
	buffer.Add(exp) // Should replace the first experience
	if len(buffer.buffer) != 2 {
		t.Errorf("Expected buffer size 2, got %d", len(buffer.buffer))
	}
}

func TestDQN(t *testing.T) {
	dqn := NewDQN(4, 10, 2, 100, 0.9, 0.1, 0.001, ReLU)
	state := []float64{1, 2, 3, 4}
	nextState := []float64{2, 3, 4, 5}
	dqn.Train(state, nextState, 1, 1, false)
}
