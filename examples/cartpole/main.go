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

// CartPoleEnvironment simulates the CartPole problem
type CartPoleEnvironment struct {
	position, velocity, angle, angularVelocity float64
	stepCount                                  int
}

func NewCartPoleEnvironment() *CartPoleEnvironment {
	return &CartPoleEnvironment{
		position:        0.0,
		velocity:        0.0,
		angle:           0.0,
		angularVelocity: 0.0,
	}
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

func (env *CartPoleEnvironment) Step(action int) ([]float64, float64, bool) {
	const gravity = 9.8
	const masscart = 1.0
	const masspole = 0.1
	const total_mass = masscart + masspole
	const length = 0.5 // actually half the pole's length
	const polemass_length = masspole * length
	const force_mag = 10.0
	const tau = 0.02 // seconds between state updates

	force := force_mag
	if action == 1 {
		force = -force_mag
	}

	temp := (force + polemass_length*env.angularVelocity*env.angularVelocity*math.Sin(env.angle)) / total_mass
	angle_acc := (gravity*math.Sin(env.angle) - math.Cos(env.angle)*temp) / (length * (4.0/3.0 - masspole*math.Cos(env.angle)*math.Cos(env.angle)/total_mass))
	acc := temp - polemass_length*angle_acc*math.Cos(env.angle)/total_mass

	env.position += tau * env.velocity
	env.velocity += tau * acc
	env.angle += tau * env.angularVelocity
	env.angularVelocity += tau * angle_acc

	env.stepCount++

	done := env.position < -2.4 || env.position > 2.4 || env.angle < -12*2*math.Pi/360 || env.angle > 12*2*math.Pi/360 || env.stepCount >= 200
	reward := 1.0
	if done {
		reward = 0.0
	}

	return []float64{env.position, env.velocity, env.angle, env.angularVelocity}, reward, done
}

func (env *CartPoleEnvironment) Reset() []float64 {
	env.position = rand.Float64()*0.08 - 0.04
	env.velocity = rand.Float64()*0.08 - 0.04
	env.angle = rand.Float64()*0.08 - 0.04
	env.angularVelocity = rand.Float64()*0.08 - 0.04
	env.stepCount = 0
	return []float64{env.position, env.velocity, env.angle, env.angularVelocity}
}

func runExperiment(agent interface{}, env *CartPoleEnvironment, episodes int) []float64 {
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
				action = a.EpsilonGreedyPolicy(dqn.Normalize(state), 2)
			case *QLearning:
				action = a.GetAction(state)
			}

			nextState, reward, stepDone := env.Step(action)
			// fmt.Println("Reward: ", reward)
			totalReward += reward

			switch a := agent.(type) {
			case *dqn.DQN:
				a.Train(dqn.Normalize(state), dqn.Normalize(nextState), action, reward, stepDone)
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
	if err := p.Save(8*vg.Inch, 4*vg.Inch, "performance_comparison_cartpole.png"); err != nil {
		log.Panic(err)
	}
}

func main() {
	env := NewCartPoleEnvironment()
	episodes := 10000

	fmt.Println("Starting DQN experiment...")
	dqnAgent := dqn.NewDQN(4, 64, 2, 10000, 0.99, 0.1, 0.001, dqn.ReLU)
	dqnRewards := runExperiment(dqnAgent, env, episodes)

	fmt.Println("Starting Q-Learning experiment...")
	qLearningAgent := NewQLearning(2, 0.1, 0.99, 0.1)
	qLearningRewards := runExperiment(qLearningAgent, env, episodes)

	fmt.Printf("DQN Average Reward: %.2f\n", stat.Mean(dqnRewards, nil))
	fmt.Printf("Q-Learning Average Reward: %.2f\n", stat.Mean(qLearningRewards, nil))

	fmt.Println("Plotting results...")
	plotResults(dqnRewards, qLearningRewards)
	fmt.Println("Done. Check 'performance_comparison_cartpole.png' for the results.")
}
