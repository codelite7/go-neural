package math

func SumVectorProducts(vectorOne, vectorTwo []float64) float64 {
	sum := 0.0
	for index, number := range vectorOne {
		sum += number * vectorTwo[index]
	}
	return sum
}

func SumVectors(vectorOne, vectorTwo []float64) []float64 {
	summedVectors := make([]float64, len(vectorOne))
	for index, number := range vectorOne {
		summedVectors[index] = number + vectorTwo[index]
	}
	return summedVectors
}

func MultiplyMatrixMembers(multiplier float64, matrix []float64) []float64 {
	result := make([]float64, len(matrix))
	for index := range matrix {
		result[index] += multiplier * matrix[index]
	}
	return result
}
