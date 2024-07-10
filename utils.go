// utils.go
package dqn

// Normalize normalizes a state vector.
func Normalize(state []float64) []float64 {
    var maxVal float64
    for _, val := range state {
        if val > maxVal {
            maxVal = val
        }
    }
    for i := range state {
        state[i] /= maxVal
    }
    return state
}
