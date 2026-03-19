pub struct MultiLayerPerceptron {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    hidden_weights: Vec<f32>,
    hidden_biases: Vec<f32>,
    output_weights: Vec<f32>,
    output_biases: Vec<f32>,
}

pub struct TrainResult {
    pub loss: f32,
    pub predicted_label: usize,
}

impl MultiLayerPerceptron {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, seed: u64) -> Self {
        let mut rng = Lcg::new(seed);
        let hidden_scale = (2.0 / input_size as f32).sqrt();
        let output_scale = (1.0 / hidden_size as f32).sqrt();

        let hidden_weights = (0..input_size * hidden_size)
            .map(|_| rng.next_range(-hidden_scale, hidden_scale))
            .collect();
        let output_weights = (0..hidden_size * output_size)
            .map(|_| rng.next_range(-output_scale, output_scale))
            .collect();

        Self {
            input_size,
            hidden_size,
            output_size,
            hidden_weights,
            hidden_biases: vec![0.0; hidden_size],
            output_weights,
            output_biases: vec![0.0; output_size],
        }
    }

    pub fn train_sample(
        &mut self,
        inputs: &[f32],
        label: usize,
        learning_rate: f32,
    ) -> TrainResult {
        assert_eq!(
            inputs.len(),
            self.input_size,
            "input size does not match model"
        );
        assert!(label < self.output_size, "label out of range");

        let mut hidden_pre = vec![0.0; self.hidden_size];
        let mut hidden = vec![0.0; self.hidden_size];

        for hidden_index in 0..self.hidden_size {
            let mut sum = self.hidden_biases[hidden_index];
            let row_offset = hidden_index * self.input_size;

            for input_index in 0..self.input_size {
                sum += self.hidden_weights[row_offset + input_index] * inputs[input_index];
            }

            hidden_pre[hidden_index] = sum;
            hidden[hidden_index] = relu(sum);
        }

        let mut logits = vec![0.0; self.output_size];
        for output_index in 0..self.output_size {
            let mut sum = self.output_biases[output_index];
            let row_offset = output_index * self.hidden_size;

            for hidden_index in 0..self.hidden_size {
                sum += self.output_weights[row_offset + hidden_index] * hidden[hidden_index];
            }

            logits[output_index] = sum;
        }

        let mut probabilities = softmax(&logits);
        let predicted_label = argmax(&probabilities);
        let loss = -probabilities[label].max(1e-7).ln();

        probabilities[label] -= 1.0;

        let mut hidden_deltas = vec![0.0; self.hidden_size];
        for hidden_index in 0..self.hidden_size {
            let mut error_signal = 0.0;

            for output_index in 0..self.output_size {
                let weight = self.output_weights[output_index * self.hidden_size + hidden_index];
                error_signal += weight * probabilities[output_index];
            }

            hidden_deltas[hidden_index] = if hidden_pre[hidden_index] > 0.0 {
                error_signal
            } else {
                0.0
            };
        }

        for output_index in 0..self.output_size {
            let row_offset = output_index * self.hidden_size;
            for hidden_index in 0..self.hidden_size {
                let gradient = probabilities[output_index] * hidden[hidden_index];
                self.output_weights[row_offset + hidden_index] -= learning_rate * gradient;
            }

            self.output_biases[output_index] -= learning_rate * probabilities[output_index];
        }

        for hidden_index in 0..self.hidden_size {
            let row_offset = hidden_index * self.input_size;
            for input_index in 0..self.input_size {
                let gradient = hidden_deltas[hidden_index] * inputs[input_index];
                self.hidden_weights[row_offset + input_index] -= learning_rate * gradient;
            }

            self.hidden_biases[hidden_index] -= learning_rate * hidden_deltas[hidden_index];
        }

        TrainResult {
            loss,
            predicted_label,
        }
    }

    pub fn predict_probabilities(&self, inputs: &[f32]) -> Vec<f32> {
        assert_eq!(
            inputs.len(),
            self.input_size,
            "input size does not match model"
        );

        let mut hidden = vec![0.0; self.hidden_size];
        for hidden_index in 0..self.hidden_size {
            let mut sum = self.hidden_biases[hidden_index];
            let row_offset = hidden_index * self.input_size;

            for input_index in 0..self.input_size {
                sum += self.hidden_weights[row_offset + input_index] * inputs[input_index];
            }

            hidden[hidden_index] = relu(sum);
        }

        let mut logits = vec![0.0; self.output_size];
        for output_index in 0..self.output_size {
            let mut sum = self.output_biases[output_index];
            let row_offset = output_index * self.hidden_size;

            for hidden_index in 0..self.hidden_size {
                sum += self.output_weights[row_offset + hidden_index] * hidden[hidden_index];
            }

            logits[output_index] = sum;
        }

        softmax(&logits)
    }

    pub fn predict_label(&self, inputs: &[f32]) -> usize {
        let probabilities = self.predict_probabilities(inputs);
        argmax(&probabilities)
    }
}

pub struct Lcg {
    state: u64,
}

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        if items.len() < 2 {
            return;
        }

        for index in (1..items.len()).rev() {
            let random_index = (self.next_u64() as usize) % (index + 1);
            items.swap(index, random_index);
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        let bits = self.next_u64() >> 40;
        let max = (1_u32 << 24) - 1;
        bits as f32 / max as f32
    }

    fn next_range(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.next_f32()
    }
}

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |left, right| left.max(right));

    let mut exp_values = Vec::with_capacity(logits.len());
    let mut total = 0.0;

    for &logit in logits {
        let value = (logit - max_logit).exp();
        exp_values.push(value);
        total += value;
    }

    exp_values.into_iter().map(|value| value / total).collect()
}

fn argmax(values: &[f32]) -> usize {
    let mut best_index = 0;
    let mut best_value = values[0];

    for (index, &value) in values.iter().enumerate().skip(1) {
        if value > best_value {
            best_value = value;
            best_index = index;
        }
    }

    best_index
}

#[cfg(test)]
mod tests {
    use super::{MultiLayerPerceptron, argmax, softmax};

    #[test]
    fn softmax_sums_to_one() {
        let probabilities = softmax(&[2.0, 1.0, 0.1]);
        let total: f32 = probabilities.iter().sum();

        assert!((total - 1.0).abs() < 1e-5);
        assert_eq!(argmax(&probabilities), 0);
    }

    #[test]
    fn learns_simple_three_class_problem() {
        let inputs = [
            [0.0, 0.0],
            [0.1, 0.2],
            [0.2, 0.1],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.1],
            [0.2, 0.9],
            [0.1, 0.8],
            [0.3, 0.7],
        ];
        let labels = [0, 0, 0, 1, 1, 1, 2, 2, 2];

        let mut model = MultiLayerPerceptron::new(2, 8, 3, 7);

        for _ in 0..1_500 {
            for (input, &label) in inputs.iter().zip(labels.iter()) {
                model.train_sample(input, label, 0.05);
            }
        }

        for (input, &label) in inputs.iter().zip(labels.iter()) {
            assert_eq!(model.predict_label(input), label);
        }
    }
}
