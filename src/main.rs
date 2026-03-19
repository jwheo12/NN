mod mlp;
mod mnist;

use std::env;
use std::error::Error;
use std::path::{Path, PathBuf};

use mlp::{Lcg, MultiLayerPerceptron};
use mnist::{CLASS_COUNT, INPUT_SIZE, load_test_set, load_training_set};

fn main() -> Result<(), Box<dyn Error>> {
    let config = Config::from_env();

    ensure_mnist_files_exist(&config.mnist_dir)?;

    let training_set = load_training_set(&config.mnist_dir, Some(config.train_limit))?;
    let test_set = load_test_set(&config.mnist_dir, Some(config.test_limit))?;

    if training_set.is_empty() || test_set.is_empty() {
        return Err("MNIST dataset is empty".into());
    }

    let mut model =
        MultiLayerPerceptron::new(INPUT_SIZE, config.hidden_size, CLASS_COUNT, config.seed);
    let mut rng = Lcg::new(config.seed ^ 0xA5A5_A5A5_A5A5_A5A5);
    let mut order: Vec<usize> = (0..training_set.len()).collect();

    println!(
        "train={} test={} hidden={} epochs={} lr={:.4}",
        training_set.len(),
        test_set.len(),
        config.hidden_size,
        config.epochs,
        config.learning_rate
    );

    for epoch in 1..=config.epochs {
        rng.shuffle(&mut order);

        let mut total_loss = 0.0;
        let mut correct = 0usize;

        for &index in &order {
            let result = model.train_sample(
                training_set.image(index),
                training_set.label(index) as usize,
                config.learning_rate,
            );
            total_loss += result.loss;

            if result.predicted_label == training_set.label(index) as usize {
                correct += 1;
            }
        }

        let train_loss = total_loss / training_set.len() as f32;
        let train_accuracy = correct as f32 / training_set.len() as f32;
        let test_accuracy = evaluate_accuracy(&model, &test_set);

        println!(
            "epoch {:>2} | loss {:.4} | train acc {:.2}% | test acc {:.2}%",
            epoch,
            train_loss,
            train_accuracy * 100.0,
            test_accuracy * 100.0
        );
    }

    println!("\nSample predictions:");
    for &index in &[0usize, 1, 2, 3, 4] {
        let image = test_set.image(index);
        let label = test_set.label(index);
        let probabilities = model.predict_probabilities(image);
        let predicted = model.predict_label(image);

        println!("label={} predicted={}", label, predicted);
        println!("{}", test_set.render_ascii(index));
        println!(
            "top probabilities: {}",
            format_top_probabilities(&probabilities, 3)
        );
    }

    Ok(())
}

fn evaluate_accuracy(model: &MultiLayerPerceptron, dataset: &mnist::MnistDataset) -> f32 {
    let mut correct = 0usize;

    for index in 0..dataset.len() {
        if model.predict_label(dataset.image(index)) == dataset.label(index) as usize {
            correct += 1;
        }
    }

    correct as f32 / dataset.len() as f32
}

fn format_top_probabilities(probabilities: &[f32], count: usize) -> String {
    let mut entries: Vec<(usize, f32)> = probabilities.iter().copied().enumerate().collect();
    entries.sort_by(|left, right| right.1.total_cmp(&left.1));

    entries
        .into_iter()
        .take(count)
        .map(|(label, probability)| format!("{label}:{:.2}%", probability * 100.0))
        .collect::<Vec<_>>()
        .join(", ")
}

fn ensure_mnist_files_exist(dir: &Path) -> Result<(), Box<dyn Error>> {
    let missing: Vec<PathBuf> = mnist::expected_paths(dir)
        .into_iter()
        .filter(|path| !path.exists())
        .collect();

    if missing.is_empty() {
        return Ok(());
    }

    let missing_list = missing
        .iter()
        .map(|path| format!("  - {}", path.display()))
        .collect::<Vec<_>>()
        .join("\n");

    Err(format!(
        "MNIST files are missing.\n{missing_list}\n\nPlace the uncompressed IDX files in {}.",
        dir.display()
    )
    .into())
}

struct Config {
    mnist_dir: PathBuf,
    train_limit: usize,
    test_limit: usize,
    hidden_size: usize,
    epochs: usize,
    learning_rate: f32,
    seed: u64,
}

impl Config {
    fn from_env() -> Self {
        Self {
            mnist_dir: env_path("MNIST_DIR", "data/mnist"),
            train_limit: env_usize("MNIST_TRAIN_LIMIT", 3_000),
            test_limit: env_usize("MNIST_TEST_LIMIT", 1_000),
            hidden_size: env_usize("MNIST_HIDDEN", 64),
            epochs: env_usize("MNIST_EPOCHS", 3),
            learning_rate: env_f32("MNIST_LR", 0.03),
            seed: env_u64("MNIST_SEED", 42),
        }
    }
}

fn env_path(key: &str, default: &str) -> PathBuf {
    env::var_os(key)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default))
}

fn env_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_u64(key: &str, default: u64) -> u64 {
    env::var(key)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default)
}

fn env_f32(key: &str, default: f32) -> f32 {
    env::var(key)
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(default)
}
