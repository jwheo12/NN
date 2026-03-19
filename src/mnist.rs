use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};

pub const IMAGE_ROWS: usize = 28;
pub const IMAGE_COLS: usize = 28;
pub const INPUT_SIZE: usize = IMAGE_ROWS * IMAGE_COLS;
pub const CLASS_COUNT: usize = 10;

const IMAGE_MAGIC: u32 = 2_051;
const LABEL_MAGIC: u32 = 2_049;

pub struct MnistDataset {
    images: Vec<f32>,
    labels: Vec<u8>,
    rows: usize,
    cols: usize,
}

impl MnistDataset {
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    pub fn image(&self, index: usize) -> &[f32] {
        let start = index * self.rows * self.cols;
        let end = start + self.rows * self.cols;
        &self.images[start..end]
    }

    pub fn label(&self, index: usize) -> u8 {
        self.labels[index]
    }

    pub fn render_ascii(&self, index: usize) -> String {
        render_ascii_image(self.image(index), self.rows, self.cols)
    }
}

pub fn load_training_set(dir: &Path, limit: Option<usize>) -> Result<MnistDataset, String> {
    load_dataset(
        dir,
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        limit,
    )
}

pub fn load_test_set(dir: &Path, limit: Option<usize>) -> Result<MnistDataset, String> {
    load_dataset(
        dir,
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
        limit,
    )
}

pub fn expected_paths(dir: &Path) -> [PathBuf; 4] {
    [
        dir.join("train-images-idx3-ubyte"),
        dir.join("train-labels-idx1-ubyte"),
        dir.join("t10k-images-idx3-ubyte"),
        dir.join("t10k-labels-idx1-ubyte"),
    ]
}

fn load_dataset(
    dir: &Path,
    image_file: &str,
    label_file: &str,
    limit: Option<usize>,
) -> Result<MnistDataset, String> {
    let image_path = dir.join(image_file);
    let label_path = dir.join(label_file);

    let image_bytes = fs::read(&image_path)
        .map_err(|error| format!("failed to read {}: {error}", image_path.display()))?;
    let label_bytes = fs::read(&label_path)
        .map_err(|error| format!("failed to read {}: {error}", label_path.display()))?;

    let (image_count, rows, cols, pixels) = parse_images(&image_bytes)?;
    let (label_count, labels) = parse_labels(&label_bytes)?;

    if image_count != label_count {
        return Err(format!(
            "image count ({image_count}) and label count ({label_count}) do not match"
        ));
    }

    if rows != IMAGE_ROWS || cols != IMAGE_COLS {
        return Err(format!(
            "expected {IMAGE_ROWS}x{IMAGE_COLS} images, got {rows}x{cols}"
        ));
    }

    let wanted = limit.unwrap_or(image_count).min(image_count);
    let image_len = rows * cols;
    let mut images = Vec::with_capacity(wanted * image_len);

    for pixel in pixels.iter().take(wanted * image_len) {
        images.push(*pixel as f32 / 255.0);
    }

    Ok(MnistDataset {
        images,
        labels: labels.into_iter().take(wanted).collect(),
        rows,
        cols,
    })
}

fn parse_images(bytes: &[u8]) -> Result<(usize, usize, usize, Vec<u8>), String> {
    if bytes.len() < 16 {
        return Err("image file is too small".to_string());
    }

    let magic = read_u32(bytes, 0)?;
    if magic != IMAGE_MAGIC {
        return Err(format!("unexpected image magic number: {magic}"));
    }

    let count = read_u32(bytes, 4)? as usize;
    let rows = read_u32(bytes, 8)? as usize;
    let cols = read_u32(bytes, 12)? as usize;
    let expected_len = 16 + count * rows * cols;

    if bytes.len() != expected_len {
        return Err(format!(
            "image file length mismatch: expected {expected_len} bytes, got {}",
            bytes.len()
        ));
    }

    Ok((count, rows, cols, bytes[16..].to_vec()))
}

fn parse_labels(bytes: &[u8]) -> Result<(usize, Vec<u8>), String> {
    if bytes.len() < 8 {
        return Err("label file is too small".to_string());
    }

    let magic = read_u32(bytes, 0)?;
    if magic != LABEL_MAGIC {
        return Err(format!("unexpected label magic number: {magic}"));
    }

    let count = read_u32(bytes, 4)? as usize;
    let expected_len = 8 + count;

    if bytes.len() != expected_len {
        return Err(format!(
            "label file length mismatch: expected {expected_len} bytes, got {}",
            bytes.len()
        ));
    }

    Ok((count, bytes[8..].to_vec()))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| "unexpected end of file while reading header".to_string())?;

    Ok(u32::from_be_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn render_ascii_image(image: &[f32], rows: usize, cols: usize) -> String {
    let palette = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];
    let mut out = String::new();

    for row in 0..rows {
        for col in 0..cols {
            let value = image[row * cols + col].clamp(0.0, 1.0);
            let index = (value * (palette.len() as f32 - 1.0)).round() as usize;
            let _ = out.write_char(palette[index]);
        }
        let _ = out.write_char('\n');
    }

    out
}

#[cfg(test)]
mod tests {
    use super::{CLASS_COUNT, IMAGE_COLS, IMAGE_ROWS, INPUT_SIZE, load_dataset};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn constants_match_mnist_shape() {
        assert_eq!(IMAGE_ROWS * IMAGE_COLS, INPUT_SIZE);
        assert_eq!(CLASS_COUNT, 10);
    }

    #[test]
    fn loads_idx_files() {
        let temp_dir = unique_temp_dir();
        fs::create_dir_all(&temp_dir).unwrap();

        let image_path = temp_dir.join("images");
        let label_path = temp_dir.join("labels");

        let mut image_bytes = Vec::new();
        image_bytes.extend_from_slice(&2_051_u32.to_be_bytes());
        image_bytes.extend_from_slice(&2_u32.to_be_bytes());
        image_bytes.extend_from_slice(&28_u32.to_be_bytes());
        image_bytes.extend_from_slice(&28_u32.to_be_bytes());
        image_bytes.extend(std::iter::repeat_n(0_u8, 28 * 28));
        image_bytes.extend(std::iter::repeat_n(255_u8, 28 * 28));

        let mut label_bytes = Vec::new();
        label_bytes.extend_from_slice(&2_049_u32.to_be_bytes());
        label_bytes.extend_from_slice(&2_u32.to_be_bytes());
        label_bytes.extend_from_slice(&[3, 7]);

        fs::write(&image_path, image_bytes).unwrap();
        fs::write(&label_path, label_bytes).unwrap();

        let dataset = load_dataset(&temp_dir, "images", "labels", Some(1)).unwrap();

        assert_eq!(dataset.len(), 1);
        assert_eq!(dataset.label(0), 3);
        assert_eq!(dataset.image(0).len(), INPUT_SIZE);
        assert!(dataset.image(0).iter().all(|&value| value == 0.0));

        fs::remove_dir_all(&temp_dir).unwrap();
    }

    fn unique_temp_dir() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        std::env::temp_dir().join(format!("mnist-test-{nanos}"))
    }
}
