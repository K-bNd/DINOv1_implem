import argparse
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torchvision.transforms.v2 as v2_transforms
import json
import os
from models.DINO import DINO
from utils import init_dataloader
from configs.config_models import ConfigDINO, ConfigDINO_Head, ConfigDataset


def extract_model(path: str):
    """
    Extract the necessary information for evaluation run
    based on training run files.

    Args:
        path (str): The path to folder that contains config files and models

    Returns:
        DINO: complete model ready for evaluation
    """

    config_dataset: dict = json.load(open(os.path.join(path, "ConfigDataset.json")))
    config_dino_head: dict = json.load(
        open(os.path.join(path, "ConfigDINO_Head.json"))
    )
    config_dino: dict = json.load(open(os.path.join(path, "ConfigDINO.json")))

    model = DINO(
        ConfigDINO(**config_dino),
        ConfigDINO_Head(**config_dino_head),
        ConfigDataset(**config_dataset),
    )
    model.student_backbone.load_state_dict(
        torch.load(os.path.join(path, "student_backbone.pt"))
    )

    model.eval()

    return model


class LinearClassifier(nn.Module):
    def __init__(self, dim, nb_classes=10) -> None:
        super(LinearClassifier, self).__init__()
        self.dim = dim
        self.nb_classes = nb_classes
        self.model = nn.Linear(dim, nb_classes)

    def forward(self, x):
        return self.model(x)


class Evaluator:
    def __init__(self, model: DINO, config: dict, device: str) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.train_dataloader, self.test_dataloader = init_dataloader(
            model.dataset_config.name,
            model.dataset_config.root,
            config["batch_size"],
            device,
            transforms=self.get_datatransform,
        )

    def get_datatransform(self, image):
        transform = v2_transforms.Compose(
            [
                v2_transforms.ToImage(),
                v2_transforms.ToDtype(torch.float32, scale=True),
            ]
        )
        with torch.no_grad():
            res = self.model(torch.unsqueeze(transform(image), dim=0), training=False)
            _, dim = res.shape
        return res.reshape(dim)

    def eval(self, name: str) -> float | None:
        acc = None
        match name:
            case "knn":
                acc = self.eval_knn()
                print(f"kNN accuracy: {acc}")
            case "linear":
                acc = self.eval_linear()
                print(f"Linear classifier accuracy: {acc}")
            case _:
                raise None

        return acc

    def eval_knn(self, n_neighbors=3):
        """Measure accuracy with KNN classifiers"""
        print("eval_knn")
        data = {"X_train": [], "y_train": [], "X_test": [], "y_test": []}

        for name, dataloader in [
            ("train", self.train_dataloader),
            ("test", self.test_dataloader),
        ]:
            for imgs, target in dataloader:
                data[f"X_{name}"].append(imgs.detach().cpu().numpy())
                data[f"y_{name}"].append(target.detach().cpu().numpy())

        data = {key: np.concatenate(value) for key, value in data.items()}

        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(data["X_train"], data["y_train"])
        y_pred = knn.predict(data["X_test"])
        acc = accuracy_score(data["y_test"], y_pred)

        return acc

    def accuracy(self, logits, targets):
        y_pred = logits.argmax(dim=1)
        return torch.mean((y_pred == targets).float())

    def eval_linear(self, epochs=20):
        """Measure accuracy with linear classifiers"""
        print("eval_linear")

        backbone_dim = self.model.model_config.out_dim
        model = LinearClassifier(
            dim=backbone_dim, nb_classes=self.model.dataset_config.num_classes
        )
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, epochs + 1):
            loop = tqdm(
                self.train_dataloader,
                desc="Training loop for linear classifier evaluation",
                total=len(self.train_dataloader),
                ascii=True,
            )
            loop.set_description(f"Epoch [{epoch}/{epochs}]")

            for imgs, target in loop:
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
                loop.set_postfix(
                    {
                        "loss": loss.item(),
                        "accuracy": self.accuracy(outputs, target).item(),
                    }
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DINOv1 evaluation script")

    parser.add_argument(
        "--eval-name",
        type=str,
        default="linear",
        choices=["knn", "linear"],
        help="The evaluation metric to use",
    )

    parser.add_argument(
        "--model-folder",
        type=str,
        default="./training_output/CIFAR10_qvmay5sl",
        help="Folder containing both the model and config files",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")

    args = vars(parser.parse_args())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    dino_model: DINO = extract_model(args["model_folder"])
    evaluator = Evaluator(dino_model, args, device)
    evaluator.eval(args["eval_name"])
