// qnetwork.go
package dqn

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Activation represents an activation function
type Activation func(float64) float64

// QNetwork represents a simple neural network for Q-value approximation.
type QNetwork struct {
	inputSize  int
	hiddenSize int
	outputSize int
	w1         *mat.Dense
	b1         *mat.VecDense
	w2         *mat.Dense
	b2         *mat.VecDense
	activation Activation
}

// NewQNetwork initializes a new QNetwork with random weights.
func NewQNetwork(inputSize, hiddenSize, outputSize int, activation Activation) *QNetwork {
	w1 := mat.NewDense(hiddenSize, inputSize, nil)
	b1 := mat.NewVecDense(hiddenSize, nil)
	w2 := mat.NewDense(outputSize, hiddenSize, nil)
	b2 := mat.NewVecDense(outputSize, nil)

	// Xavier initialization
	bound1 := math.Sqrt(6.0 / float64(inputSize+hiddenSize))
	bound2 := math.Sqrt(6.0 / float64(hiddenSize+outputSize))

	w1.Apply(func(_, _ int, _ float64) float64 { return rand.Float64()*2*bound1 - bound1 }, w1)
	for i := 0; i < hiddenSize; i++ {
		b1.SetVec(i, rand.Float64()*2*bound1-bound1)
	}
	w2.Apply(func(_, _ int, _ float64) float64 { return rand.Float64()*2*bound2 - bound2 }, w2)
	for i := 0; i < outputSize; i++ {
		b2.SetVec(i, rand.Float64()*2*bound2-bound2)
	}

	return &QNetwork{
		inputSize:  inputSize,
		hiddenSize: hiddenSize,
		outputSize: outputSize,
		w1:         w1,
		b1:         b1,
		w2:         w2,
		b2:         b2,
		activation: activation,
	}
}

// Predict returns Q-values for a given state.
func (q *QNetwork) Predict(state []float64) []float64 {
	if len(state) != q.inputSize {
		panic("Input state size does not match network input size")
	}

	// Convert input to matrix
	x := mat.NewVecDense(len(state), state)

	// First layer
	h := mat.NewVecDense(q.hiddenSize, nil)
	h.MulVec(q.w1, x)
	h.AddVec(h, q.b1)

	// Apply activation function element-wise
	for i := 0; i < h.Len(); i++ {
		h.SetVec(i, q.activation(h.AtVec(i)))
	}

	// Output layer
	out := mat.NewVecDense(q.outputSize, nil)
	out.MulVec(q.w2, h)
	out.AddVec(out, q.b2)

	return out.RawVector().Data
}

// Loss computes the mean squared error loss.
func (q *QNetwork) Loss(predictions, targets []float64) float64 {
	if len(predictions) != len(targets) {
		panic("Predictions and targets must have the same length")
	}

	var loss float64
	for i := range predictions {
		diff := predictions[i] - targets[i]
		loss += diff * diff
	}
	return loss / float64(len(predictions))
}

// Backward computes gradients and updates the network weights.
func (q *QNetwork) Backward(state, prediction, target []float64, learningRate float64) {
	// Convert inputs to matrices
	x := mat.NewVecDense(len(state), state)
	y := mat.NewVecDense(len(target), target)
	yHat := mat.NewVecDense(len(prediction), prediction)

	// Forward pass (recompute for gradient calculation)
	h := mat.NewVecDense(q.hiddenSize, nil)
	h.MulVec(q.w1, x)
	h.AddVec(h, q.b1)

	// Apply activation function element-wise
	for i := 0; i < h.Len(); i++ {
		h.SetVec(i, q.activation(h.AtVec(i)))
	}

	// Compute gradients
	dOut := mat.NewVecDense(q.outputSize, nil)
	dOut.SubVec(yHat, y)

	dW2 := mat.NewDense(q.outputSize, q.hiddenSize, nil)
	dW2.Outer(1, dOut, h)

	dB2 := dOut

	dH := mat.NewVecDense(q.hiddenSize, nil)
	dH.MulVec(q.w2.T(), dOut)
	dH.MulElemVec(dH, applyDerivative(h, q.activation))

	dW1 := mat.NewDense(q.hiddenSize, q.inputSize, nil)
	dW1.Outer(1, dH, x)

	dB1 := dH

	// Update weights and biases
	q.w2.Scale(-learningRate, dW2)
	q.w2.Add(q.w2, dW2)

	q.b2.AddScaledVec(q.b2, -learningRate, dB2)

	q.w1.Scale(-learningRate, dW1)
	q.w1.Add(q.w1, dW1)

	q.b1.AddScaledVec(q.b1, -learningRate, dB1)
}

// applyDerivative applies the derivative of the activation function element-wise
func applyDerivative(v *mat.VecDense, activation Activation) *mat.VecDense {
	result := mat.NewVecDense(v.Len(), nil)
	for i := 0; i < v.Len(); i++ {
		x := v.AtVec(i)
		// Approximate derivative
		h := 1e-4
		result.SetVec(i, (activation(x+h)-activation(x-h))/(2*h))
	}
	return result
}

// Common activation functions

func ReLU(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func Tanh(x float64) float64 {
	return math.Tanh(x)
}
