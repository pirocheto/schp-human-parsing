from pathlib import Path

from datasets import load_dataset

N = 10
OUTPUT_DIR = Path("images")
SEED = 42


def download_images(n: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "detection-datasets/fashionpedia", split="val", streaming=True
    ).shuffle(seed=SEED, buffer_size=1000)

    for i, sample in enumerate(dataset.take(n)):
        img = sample["image"]
        path = output_dir / f"image_{i}.jpg"
        img.save(path)
        print(f"Image {i} saved → {path}")


if __name__ == "__main__":
    download_images(N, OUTPUT_DIR)
