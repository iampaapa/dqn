// replaybuffer.go
package dqn

import "math/rand"

// Experience represents a single experience tuple.
type Experience struct {
    State, NextState []float64
    Action, Reward   int
    Done             bool
}

// ReplayBuffer stores experiences for training.
type ReplayBuffer struct {
    buffer []Experience
    size   int
}

// NewReplayBuffer initializes a new ReplayBuffer.
func NewReplayBuffer(size int) *ReplayBuffer {
    return &ReplayBuffer{size: size}
}

// Add adds a new experience to the buffer.
func (rb *ReplayBuffer) Add(exp Experience) {
    if len(rb.buffer) >= rb.size {
        rb.buffer = rb.buffer[1:]
    }
    rb.buffer = append(rb.buffer, exp)
}

// Sample returns a batch of experiences.
func (rb *ReplayBuffer) Sample(batchSize int) []Experience {
    sample := make([]Experience, batchSize)
    for i := range sample {
        sample[i] = rb.buffer[rand.Intn(len(rb.buffer))]
    }
    return sample
}
