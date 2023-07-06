import os
import tempfile

import torchvision
from tqdm.auto import tqdm


CLASSES = (
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
)


def main():
    out_dir = f"mnist_7"
    if os.path.exists(out_dir):
        print(f"{out_dir} already exists.")

    print("downloading...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset = torchvision.datasets.MNIST(
            root=tmp_dir, train= True , download=True
        )

    resizer = torchvision.transforms.Resize((32, 32))
    print("dumping images...")
    os.mkdir(out_dir)
    count = 0
    for i in tqdm(range(len(dataset))):
        image, label = dataset[i]
        if label == 7:
            count += 1
            image = resizer(image)
            filename = os.path.join(out_dir, f"{count:05d}.png")
            image.save(filename)


if __name__ == "__main__":
    main()