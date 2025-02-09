use linear_algebra_ml::{Dense, Sparse, Vector};
use linear_algebra_ml::matrix::{matrix::Matrix, ops::{scalar_mul, add, sub, mul, transpose}};
use std::f64::consts::E;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(x: f64) -> f64 {
    let sig = sigmoid(x);
    sig * (1.0 - sig)
}

pub struct SimpleNN {
    weights: Matrix,
    bias: Matrix,
}

impl SimpleNN {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let weights = Matrix::new(output_size, input_size, vec![0.5; output_size * input_size]);
        let bias = Matrix::new(output_size, 1, vec![0.0; output_size]);
        Self { weights, bias }
    }



    pub fn forward(&self, input: &Matrix) -> Matrix {
        let weighted_sum = add(&mul(&self.weights, input).unwrap(), &self.bias).unwrap();
        let mut result = weighted_sum.clone();
        for i in 0..result.rows {
            for j in 0..result.cols {
                result.set(i, j, sigmoid(result.get(i, j)));
            }
        }
        result
    }

    pub fn train(&mut self, input: &Matrix, target: &Matrix, learning_rate: f64) {
        let output = self.forward(input);
        let error = sub(target, &output).unwrap();

        let mut gradient = output.clone();
        for i in 0..gradient.rows {
            for j in 0..gradient.cols {
                gradient.set(i, j, sigmoid_derivative(output.get(i, j)));
            }
        }
        gradient = mul(&gradient, &error).unwrap();
        gradient = scalar_mul(&gradient, learning_rate);

        let input_t = transpose(input);
        let weight_delta = mul(&gradient, &input_t).unwrap();
        self.weights = add(&self.weights, &weight_delta).unwrap();
        self.bias = add(&self.bias, &gradient).unwrap();
    }
}

fn main() {
    let mut nn = SimpleNN::new(2, 1);
    let input = Vector!(2, [0.0, 1.0]);
    let target = Vector!(1, [1.0]);
    
    for _ in 0..1000 {
        nn.train(&input, &target, 0.1);
    }
    
    let output = nn.forward(&input);
    println!("Output: {}", output);
}
