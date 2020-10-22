package structs

import (
	"math"
	"math/rand"
	"time"

	math2 "github.com/codelite7/go-neural/math"
)

type NeuralNet struct {
	Inputs          [][]float64
	ExpectedOutputs []float64
	Weights         []float64
	Bias            float64
	TrainingCycles  int
}

// Initialize to 0 Bias and random Weights
func (neuralNet *NeuralNet) Initialize() {
	rand.Seed(time.Now().UnixNano())
	neuralNet.Bias = 0.0
	neuralNet.Weights = make([]float64, len(neuralNet.Inputs[0]))
	for i := 0; i < len(neuralNet.Inputs[0]); i++ {
		neuralNet.Weights[i] = rand.Float64()
	}
}

func (neuralNet *NeuralNet) sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func (neuralNet *NeuralNet) PropagateForward(x []float64) (sum float64) {
	return neuralNet.sigmoid(math2.SumVectorProducts(neuralNet.Weights, x) + neuralNet.Bias)
}

// calculate gradients of Weights
func (neuralNet *NeuralNet) GetWeightGradients(x []float64, y float64) []float64 {
	pred := neuralNet.PropagateForward(x)
	return math2.MultiplyMatrixMembers(-(pred-y)*pred*(1-pred), x)
}

// calculate gradients of Bias
func (neuralNet *NeuralNet) GetBiasGradient(x []float64, y float64) float64 {
	pred := neuralNet.PropagateForward(x)
	return -(pred - y) * pred * (1 - pred)
}

// Train the neural network
func (neuralNet *NeuralNet) Train() {
	for i := 0; i < neuralNet.TrainingCycles; i++ {
		weights := make([]float64, len(neuralNet.Inputs[0]))
		bias := 0.0
		for index, value := range neuralNet.Inputs {
			weights = math2.SumVectors(weights, neuralNet.GetWeightGradients(value, neuralNet.ExpectedOutputs[index]))
			bias += neuralNet.GetBiasGradient(value, neuralNet.ExpectedOutputs[index])
		}
		weights = math2.MultiplyMatrixMembers(2/float64(len(neuralNet.ExpectedOutputs)), weights)
		neuralNet.Weights = math2.SumVectors(neuralNet.Weights, weights)
		neuralNet.Bias += bias * 2 / float64(len(neuralNet.ExpectedOutputs))
	}
}
