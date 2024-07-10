package main

import (
	"fmt"
	"image/color"
	"log"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"

	"github.com/iampaapa/dqn"
)

// ManufacturingEnvironment simulates a manufacturing process
type ManufacturingEnvironment struct {
	temperature float64
	pressure    float64
	flow        float64
	stepCount   int
}

func NewManufacturingEnvironment() *ManufacturingEnvironment {
	return &ManufacturingEnvironment{
		temperature: 150.0,
		pressure:    50.0,
		flow:        10.0,
	}
}

func (env *ManufacturingEnvironment) Step(action int) ([]float64, float64, bool) {
	// Adjust manufacturing parameters based on action
	switch action {
	case 0: // Increase temperature
		env.temperature += 5
	case 1: // Decrease temperature
		env.temperature -= 5
	case 2: // Increase pressure
		env.pressure += 2
	case 3: // Decrease pressure
		env.pressure -= 2
	case 4: // Increase flow
		env.flow += 1
	case 5: // Decrease flow
		env.flow -= 1
	}

	// Calculate reward based on how close we are to optimal conditions
	reward := -math.Abs(env.temperature-180) / 10.0
	reward -= math.Abs(env.pressure-60) / 5.0
	reward -= math.Abs(env.flow-12) / 2.0

	// Increment step count
	env.stepCount++

	// Check if we've reached a terminal state (after 10 rounds)
	// done := env.stepCount >= 10
	done := reward > -50.0 // Alternative: Consider the process optimized if reward is high enough

	// Return the new state, reward, and whether we're done
	return []float64{env.temperature, env.pressure, env.flow}, reward, done
}

func (env *ManufacturingEnvironment) Reset() []float64 {
	env.temperature = 150.0 + rand.Float64()*20.0 - 10.0
	env.pressure = 50.0 + rand.Float64()*10.0 - 5.0
	env.flow = 10.0 + rand.Float64()*2.0 - 1.0
	env.stepCount = 0 // Reset step count
	return []float64{env.temperature, env.pressure, env.flow}
}

// QLearning implements a simple Q-learning algorithm for comparison
type QLearning struct {
	qTable     map[int][]float64
	alpha      float64
	gamma      float64
	epsilon    float64
	numActions int
}

func NewQLearning(numActions int, alpha, gamma, epsilon float64) *QLearning {
	return &QLearning{
		qTable:     make(map[int][]float64),
		alpha:      alpha,
		gamma:      gamma,
		epsilon:    epsilon,
		numActions: numActions,
	}
}

func (q *QLearning) GetAction(state []float64) int {
	stateKey := q.discretizeState(state)
	if _, ok := q.qTable[stateKey]; !ok {
		q.qTable[stateKey] = make([]float64, q.numActions)
	}

	if rand.Float64() < q.epsilon {
		return rand.Intn(q.numActions)
	}
	return dqn.Argmax(q.qTable[stateKey])
}

func (q *QLearning) Update(state []float64, action int, reward float64, nextState []float64) {
	stateKey := q.discretizeState(state)
	nextStateKey := q.discretizeState(nextState)

	if _, ok := q.qTable[stateKey]; !ok {
		q.qTable[stateKey] = make([]float64, q.numActions)
	}
	if _, ok := q.qTable[nextStateKey]; !ok {
		q.qTable[nextStateKey] = make([]float64, q.numActions)
	}

	currentQ := q.qTable[stateKey][action]
	maxNextQ := dqn.Max(q.qTable[nextStateKey])
	newQ := currentQ + q.alpha*(reward+q.gamma*maxNextQ-currentQ)
	q.qTable[stateKey][action] = newQ
}

func (q *QLearning) discretizeState(state []float64) int {
	// Simple discretization: round each value to nearest integer
	return int(math.Round(state[0])*10000 + math.Round(state[1])*100 + math.Round(state[2]))
}

func runExperiment(agent interface{}, env *ManufacturingEnvironment, episodes int) []float64 {
	rewards := make([]float64, episodes)

	for i := 0; i < episodes; i++ {
		if i%100 == 0 {
			fmt.Printf("Running episode %d/%d\n", i, episodes)
		}
		state := env.Reset()
		totalReward := 0.0
		done := false

		for !done {
			var action int
			switch a := agent.(type) {
			case *dqn.DQN:
				action = a.EpsilonGreedyPolicy(dqn.Normalize(state), 6)
			case *QLearning:
				action = a.GetAction(state)
			}

			nextState, reward, stepDone := env.Step(action)
			totalReward += reward

			switch a := agent.(type) {
			case *dqn.DQN:
				a.Train(dqn.Normalize(state), dqn.Normalize(nextState), action, int(reward*100), stepDone)
			case *QLearning:
				a.Update(state, action, reward, nextState)
			}

			state = nextState
			done = stepDone // Update the outer done variable
		}

		rewards[i] = totalReward
	}

	return rewards
}

func plotResults(dqnRewards, qLearningRewards []float64) {
	p := plot.New()

	p.Title.Text = "DQN vs Q-Learning Performance"
	p.X.Label.Text = "Episode"
	p.Y.Label.Text = "Total Reward"

	dqnData := make(plotter.XYs, len(dqnRewards))
	qLearningData := make(plotter.XYs, len(qLearningRewards))

	for i := range dqnRewards {
		dqnData[i].X = float64(i)
		dqnData[i].Y = dqnRewards[i]
		qLearningData[i].X = float64(i)
		qLearningData[i].Y = qLearningRewards[i]
	}

	// Create a line plotter for the DQN data
	dqnLine, err := plotter.NewLine(dqnData)
	if err != nil {
		log.Panic(err)
	}
	dqnLine.Color = color.RGBA{R: 255, G: 0, B: 0, A: 255} // Red

	// Create a line plotter for the Q-Learning data
	qLearningLine, err := plotter.NewLine(qLearningData)
	if err != nil {
		log.Panic(err)
	}
	qLearningLine.Color = color.RGBA{B: 255, A: 255} // Blue

	// Add the lines to the plot
	p.Add(dqnLine, qLearningLine)
	p.Legend.Add("DQN", dqnLine)
	p.Legend.Add("Q-Learning", qLearningLine)

	// Save the plot to a PNG file
	if err := p.Save(8*vg.Inch, 4*vg.Inch, "performance_comparison.png"); err != nil {
		log.Panic(err)
	}
}

func main() {
	env := NewManufacturingEnvironment()
	episodes := 1000

	fmt.Println("Starting DQN experiment...")
	dqnAgent := dqn.NewDQN(3, 64, 6, 10000, 0.99, 0.1, 0.001, dqn.ReLU)
	dqnRewards := runExperiment(dqnAgent, env, episodes)

	fmt.Println("Starting Q-Learning experiment...")
	qLearningAgent := NewQLearning(6, 0.1, 0.99, 0.1)
	qLearningRewards := runExperiment(qLearningAgent, env, episodes)

	fmt.Printf("DQN Average Reward: %.2f\n", stat.Mean(dqnRewards, nil))
	fmt.Printf("Q-Learning Average Reward: %.2f\n", stat.Mean(qLearningRewards, nil))

	fmt.Println("Plotting results...")
	plotResults(dqnRewards, qLearningRewards)
	fmt.Println("Done. Check 'performance_comparison.png' for the results.")
}