const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LinearRegression {
  constructor(features, labels, options) {
    this.features = features;
    this.labels = labels;

    this.options = Object.assign(
      { learningRate: 0.1, iterations: 1000 },
      options
    );
    this.weights = ts.zeros([2,1]);
  }

  gradientDescend() {
    const currentGuesses = this.features.matMul(this.weights);
    const differences = currentGuesses.sub(this.labels);

    const slopes = this.features
    .transpose()
    .matMul(differences)
    .div(this.features.shape[0])

     this.weights = this.weights.sub(slopes.mul(this.options.learningRate))
  }
 
  train() {
    for (let i = 0; i < this.options.iterations; i++) {
      this.gradientDescend();
    }
  }
}

module.exports = LinearRegression;
