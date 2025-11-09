package dqn

import (
	"encoding/gob"
	"io"

	"gonum.org/v1/gonum/mat"
)

// serializableDQN is the lightweight struct we'll encode/decode
type serializableDQN struct {
	W1, W2       [][]float64
	B1, B2       []float64
	Gamma        float64
	Epsilon      float64
	LearningRate float64
}

// Save writes the model parameters to a writer (e.g. file)
func (d *DQN) Save(w io.Writer) error {
	enc := gob.NewEncoder(w)

	s := serializableDQN{
		W1:           matToSlices(d.qNetwork.w1),
		W2:           matToSlices(d.qNetwork.w2),
		B1:           vecToSlice(d.qNetwork.b1),
		B2:           vecToSlice(d.qNetwork.b2),
		Gamma:        d.gamma,
		Epsilon:      d.epsilon,
		LearningRate: d.learningRate,
	}

	return enc.Encode(s)
}

// Load restores the model parameters from a reader (e.g. file)
func (d *DQN) Load(r io.Reader) error {
	dec := gob.NewDecoder(r)
	var s serializableDQN
	if err := dec.Decode(&s); err != nil {
		return err
	}

	// rebuild network weights
	d.qNetwork.w1 = slicesToMat(s.W1)
	d.qNetwork.w2 = slicesToMat(s.W2)
	d.qNetwork.b1 = sliceToVec(s.B1)
	d.qNetwork.b2 = sliceToVec(s.B2)

	d.gamma = s.Gamma
	d.epsilon = s.Epsilon
	d.learningRate = s.LearningRate
	return nil
}

// ===== Helper conversion functions =====

func matToSlices(m *mat.Dense) [][]float64 {
	r, c := m.Dims()
	out := make([][]float64, r)
	for i := 0; i < r; i++ {
		row := make([]float64, c)
		for j := 0; j < c; j++ {
			row[j] = m.At(i, j)
		}
		out[i] = row
	}
	return out
}

func vecToSlice(v *mat.VecDense) []float64 {
	n := v.Len()
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = v.AtVec(i)
	}
	return out
}

func slicesToMat(s [][]float64) *mat.Dense {
	if len(s) == 0 {
		return nil
	}
	r := len(s)
	c := len(s[0])
	data := make([]float64, 0, r*c)
	for _, row := range s {
		data = append(data, row...)
	}
	return mat.NewDense(r, c, data)
}

func sliceToVec(s []float64) *mat.VecDense {
	if len(s) == 0 {
		return nil
	}
	return mat.NewVecDense(len(s), s)
}
