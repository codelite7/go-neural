package main

import (
	"github.com/rs/zerolog/log"

	"github.com/codelite7/go-neural/structs"
)

func main() {
	neuralNet := structs.NeuralNet{
		Inputs:          [][]float64{{0, 0, 1}, {1, 1, 1}, {1, 0, 1}, {0, 1, 0}},
		ExpectedOutputs: []float64{0, 1, 1, 0},
		TrainingCycles:  100000,
	}
	neuralNet.Initialize()
	neuralNet.Train()
	prediction := neuralNet.PropagateForward([]float64{0, 1, 0})
	log.Info().Float64("prediction", prediction).Msg("prediction")
	prediction = neuralNet.PropagateForward([]float64{1, 0, 1})
	log.Info().Float64("prediction", prediction).Msg("prediction")
}
