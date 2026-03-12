from __future__ import annotations

import copy

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def flatten_images(images: torch.Tensor) -> torch.Tensor:
    return images.view(images.size(0), -1)


def to_python_table(rows: List[Dict]) -> List[Dict]:
    if pd is None:
        return rows
    return pd.DataFrame(rows)


@dataclass
class DatasetBundle:
    name: str
    train_loader: DataLoader
    test_loader: DataLoader
    input_dim: int
    input_shape: Tuple[int, int, int]
    num_classes: int
    class_names: List[str]


def _subset_dataset(dataset, subset_size: Optional[int], seed: int):
    if subset_size is None or subset_size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)


def load_image_dataloaders(
    name: str = "mnist",
    batch_size: int = 128,
    train_subset: Optional[int] = 12000,
    test_subset: Optional[int] = 2000,
    seed: int = 42,
    root: str = "../data",
    num_workers: int = 0,
) -> DatasetBundle:
    dataset_name = name.lower()
    dataset_map = {
        "mnist": (
            datasets.MNIST,
            [str(index) for index in range(10)],
        ),
        "fashion_mnist": (
            datasets.FashionMNIST,
            [
                "T-shirt/top",
                "Trouser",
                "Pullover",
                "Dress",
                "Coat",
                "Sandal",
                "Shirt",
                "Sneaker",
                "Bag",
                "Ankle boot",
            ],
        ),
    }
    if dataset_name not in dataset_map:
        raise ValueError(f"Unknown dataset '{name}'. Available options: {sorted(dataset_map)}")

    dataset_class, class_names = dataset_map[dataset_name]
    transform = transforms.ToTensor()

    train_dataset = dataset_class(root=root, train=True, transform=transform, download=True)
    test_dataset = dataset_class(root=root, train=False, transform=transform, download=True)

    train_dataset = _subset_dataset(train_dataset, train_subset, seed)
    test_dataset = _subset_dataset(test_dataset, test_subset, seed + 1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    sample_images, _ = next(iter(train_loader))
    input_shape = tuple(sample_images.shape[1:])
    input_dim = int(np.prod(input_shape))

    return DatasetBundle(
        name=dataset_name,
        train_loader=train_loader,
        test_loader=test_loader,
        input_dim=input_dim,
        input_shape=input_shape,
        num_classes=len(class_names),
        class_names=class_names,
    )


def show_dataset_samples(loader: DataLoader, num_images: int = 10, title: str = "Dataset samples") -> None:
    images, labels = next(iter(loader))
    num_images = min(num_images, len(images))
    fig, axes = plt.subplots(1, num_images, figsize=(1.6 * num_images, 2.3))
    if num_images == 1:
        axes = [axes]
    for axis, image, label in zip(axes, images[:num_images], labels[:num_images]):
        axis.imshow(image.squeeze(0), cmap="gray")
        axis.set_title(str(int(label)))
        axis.axis("off")
    fig.suptitle(title)
    plt.tight_layout()


def _apply_activation(x: torch.Tensor, name: str) -> torch.Tensor:
    activation = name.lower()
    if activation == "relu":
        return torch.relu(x)
    if activation in {"relu_clamp", "clipped_relu", "bounded_relu"}:
        return torch.relu(x).clamp(0.0, 1.0)
    if activation == "sigmoid":
        return torch.sigmoid(x)
    if activation == "tanh":
        return torch.tanh(x)
    if activation == "gelu":
        return nn.GELU()(x)
    if activation in {"identity", "linear", "none"}:
        return x
    raise ValueError(f"Unsupported activation '{name}'.")


def _activation_derivative(
    activation: str,
    preactivation: torch.Tensor,
    activated: torch.Tensor,
) -> torch.Tensor:
    activation = activation.lower()
    if activation == "sigmoid":
        return activated * (1.0 - activated)
    if activation == "tanh":
        return 1.0 - activated.pow(2)
    if activation == "relu":
        return (preactivation > 0).float()
    if activation in {"identity", "linear", "none"}:
        return torch.ones_like(preactivation)
    raise ValueError(f"Contractive derivative not implemented for '{activation}'.")


def _activation_outputs_unit_interval(name: str) -> bool:
    return name.lower() in {"sigmoid", "relu_clamp", "clipped_relu", "bounded_relu"}


def _validate_reconstruction_setup(loss_name: str, output_activation: str) -> None:
    if loss_name.lower() == "bce" and not _activation_outputs_unit_interval(output_activation):
        raise ValueError(
            "BCELoss requires decoder outputs in [0, 1]. "
            f"Received output_activation='{output_activation}'. "
            "Use 'sigmoid' for the standard setup, or 'relu_clamp' for an experimental "
            "bounded-ReLU decoder."
        )


def _resolve_stacked_pretrain_loss(
    loss_name: str,
    activation: str,
    output_activation: str,
) -> str:
    requested = loss_name.lower()
    if requested == "auto":
        if _activation_outputs_unit_interval(activation) and _activation_outputs_unit_interval(output_activation):
            return "bce"
        return "mse"

    if requested == "bce" and not _activation_outputs_unit_interval(activation):
        return "mse"

    return requested


class FlexibleAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (256,),
        latent_dim: int = 32,
        activation: str = "relu",
        output_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = tuple(hidden_dims)
        self.latent_dim = latent_dim
        self.activation = activation
        self.output_activation = output_activation

        encoder_dims = [input_dim, *self.hidden_dims, latent_dim]
        decoder_dims = [latent_dim, *reversed(self.hidden_dims), input_dim]

        self.encoder_linears = nn.ModuleList(
            nn.Linear(in_features, out_features)
            for in_features, out_features in zip(encoder_dims[:-1], encoder_dims[1:])
        )
        self.decoder_linears = nn.ModuleList(
            nn.Linear(in_features, out_features)
            for in_features, out_features in zip(decoder_dims[:-1], decoder_dims[1:])
        )

    def encode(self, x: torch.Tensor, return_preactivation: bool = False):
        hidden = x
        latent_preactivation = None
        for layer_index, linear in enumerate(self.encoder_linears):
            hidden = linear(hidden)
            if layer_index == len(self.encoder_linears) - 1:
                latent_preactivation = hidden
            hidden = _apply_activation(hidden, self.activation)
        if return_preactivation:
            return hidden, latent_preactivation
        return hidden

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        hidden = z
        for layer_index, linear in enumerate(self.decoder_linears):
            hidden = linear(hidden)
            is_last_layer = layer_index == len(self.decoder_linears) - 1
            activation_name = self.output_activation if is_last_layer else self.activation
            hidden = _apply_activation(hidden, activation_name)
        return hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def forward_with_latent(self, x: torch.Tensor):
        latent, latent_preactivation = self.encode(x, return_preactivation=True)
        reconstruction = self.decode(latent)
        return reconstruction, latent, latent_preactivation


@dataclass
class AutoencoderExperimentSpec:
    name: str
    hidden_dims: Tuple[int, ...] = (256,)
    latent_dim: int = 32
    activation: str = "relu"
    output_activation: str = "sigmoid"
    loss_name: str = "bce"
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 0.0
    corruption: str = "none"
    corruption_strength: float = 0.0
    l1_weight: float = 0.0
    kl_weight: float = 0.0
    kl_target: float = 0.05
    contractive_weight: float = 0.0


def corrupt_inputs(inputs: torch.Tensor, mode: str = "none", strength: float = 0.0) -> torch.Tensor:
    mode = mode.lower()
    if mode == "none" or strength <= 0:
        return inputs
    if mode == "gaussian":
        noisy = inputs + torch.randn_like(inputs) * strength
        return noisy.clamp(0.0, 1.0)
    if mode == "masking":
        keep_mask = (torch.rand_like(inputs) > strength).float()
        return inputs * keep_mask
    if mode == "salt_pepper":
        random_tensor = torch.rand_like(inputs)
        salt = (random_tensor < strength / 2).float()
        pepper = (random_tensor > 1 - strength / 2).float()
        neutral = 1.0 - salt - pepper
        return (inputs * neutral + salt).clamp(0.0, 1.0)
    raise ValueError(f"Unknown corruption mode '{mode}'.")


def kl_sparsity_penalty(latent: torch.Tensor, rho: float = 0.05, eps: float = 1e-6) -> torch.Tensor:
    rho_hat = latent.mean(dim=0).clamp(eps, 1.0 - eps)
    rho_tensor = torch.full_like(rho_hat, fill_value=rho)
    return (
        rho_tensor * torch.log(rho_tensor / rho_hat)
        + (1.0 - rho_tensor) * torch.log((1.0 - rho_tensor) / (1.0 - rho_hat))
    ).mean()


def contractive_penalty(
    model: FlexibleAutoencoder,
    latent: torch.Tensor,
    latent_preactivation: torch.Tensor,
) -> torch.Tensor:
    if len(model.encoder_linears) != 1:
        return torch.zeros((), device=latent.device)
    weight = model.encoder_linears[0].weight
    squared_row_norms = weight.pow(2).sum(dim=1)
    derivatives = _activation_derivative(model.activation, latent_preactivation, latent)
    penalty_per_sample = (derivatives.pow(2) * squared_row_norms.unsqueeze(0)).sum(dim=1)
    return penalty_per_sample.mean()


def _reconstruction_loss(name: str):
    loss_name = name.lower()
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "bce":
        return nn.BCELoss()
    raise ValueError(f"Unknown reconstruction loss '{name}'.")


def train_autoencoder(
    model: FlexibleAutoencoder,
    train_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    *,
    epochs: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    loss_name: str = "bce",
    corruption: str = "none",
    corruption_strength: float = 0.0,
    l1_weight: float = 0.0,
    kl_weight: float = 0.0,
    kl_target: float = 0.05,
    contractive_weight: float = 0.0,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    device = device or get_device()
    model.to(device)
    _validate_reconstruction_setup(loss_name, model.output_activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = _reconstruction_loss(loss_name)

    history = {
        "train_total": [],
        "train_reconstruction": [],
        "train_regularization": [],
        "eval_reconstruction": [],
    }

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_reconstruction = 0.0
        total_regularization = 0.0
        total_examples = 0

        for images, _ in train_loader:
            clean = flatten_images(images).to(device)
            noisy = corrupt_inputs(clean, corruption, corruption_strength)

            optimizer.zero_grad()
            reconstruction, latent, latent_preactivation = model.forward_with_latent(noisy)
            reconstruction_loss = criterion(reconstruction, clean)

            regularization = torch.zeros((), device=device)
            if l1_weight > 0:
                regularization = regularization + l1_weight * latent.abs().mean()
            if kl_weight > 0:
                latent_for_kl = latent if model.activation.lower() == "sigmoid" else torch.sigmoid(latent_preactivation)
                regularization = regularization + kl_weight * kl_sparsity_penalty(latent_for_kl, rho=kl_target)
            if contractive_weight > 0:
                regularization = regularization + contractive_weight * contractive_penalty(
                    model,
                    latent,
                    latent_preactivation,
                )

            loss = reconstruction_loss + regularization
            loss.backward()
            optimizer.step()

            batch_size = clean.size(0)
            total_examples += batch_size
            total_loss += loss.item() * batch_size
            total_reconstruction += reconstruction_loss.item() * batch_size
            total_regularization += regularization.item() * batch_size

        history["train_total"].append(total_loss / total_examples)
        history["train_reconstruction"].append(total_reconstruction / total_examples)
        history["train_regularization"].append(total_regularization / total_examples)

        if test_loader is not None:
            eval_metrics = evaluate_autoencoder(
                model,
                test_loader,
                device=device,
                loss_name=loss_name,
            )
            history["eval_reconstruction"].append(eval_metrics["loss"])

        if verbose:
            message = (
                f"Epoch {epoch + 1:02d}/{epochs:02d} | "
                f"train_total={history['train_total'][-1]:.4f} | "
                f"train_recon={history['train_reconstruction'][-1]:.4f}"
            )
            if history["eval_reconstruction"]:
                message += f" | test_recon={history['eval_reconstruction'][-1]:.4f}"
            print(message)

    return history


def evaluate_autoencoder(
    model: FlexibleAutoencoder,
    loader: DataLoader,
    *,
    device: Optional[torch.device] = None,
    loss_name: str = "bce",
    corruption: str = "none",
    corruption_strength: float = 0.0,
) -> Dict[str, float]:
    device = device or get_device()
    criterion = _reconstruction_loss(loss_name)
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_examples = 0
    latent_abs_mean = 0.0
    latent_zero_ratio = 0.0

    with torch.no_grad():
        for images, _ in loader:
            clean = flatten_images(images).to(device)
            noisy = corrupt_inputs(clean, corruption, corruption_strength)
            reconstruction, latent, _ = model.forward_with_latent(noisy)
            loss = criterion(reconstruction, clean)

            batch_size = clean.size(0)
            total_examples += batch_size
            total_loss += loss.item() * batch_size
            latent_abs_mean += latent.abs().mean().item() * batch_size
            latent_zero_ratio += (latent.abs() < 1e-3).float().mean().item() * batch_size

    return {
        "loss": total_loss / total_examples,
        "latent_abs_mean": latent_abs_mean / total_examples,
        "latent_zero_ratio": latent_zero_ratio / total_examples,
    }


def collect_encoded_features(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: Optional[torch.device] = None,
    max_points: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = device or get_device()
    model.eval()
    model.to(device)

    features = []
    labels = []
    total = 0

    with torch.no_grad():
        for images, target in loader:
            flat = flatten_images(images).to(device)
            if hasattr(model, "encode_features"):
                encoded = model.encode_features(flat)
            else:
                encoded = model.encode(flat)
            features.append(encoded.cpu())
            labels.append(target.cpu())
            total += len(images)
            if max_points is not None and total >= max_points:
                break

    features_tensor = torch.cat(features, dim=0)
    labels_tensor = torch.cat(labels, dim=0)
    if max_points is not None:
        features_tensor = features_tensor[:max_points]
        labels_tensor = labels_tensor[:max_points]
    return features_tensor, labels_tensor


def linear_probe_accuracy(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    num_classes: int,
    device: Optional[torch.device] = None,
    epochs: int = 15,
    lr: float = 0.1,
) -> float:
    device = device or get_device()
    train_features, train_labels = collect_encoded_features(model, train_loader, device=device)
    test_features, test_labels = collect_encoded_features(model, test_loader, device=device)

    probe = nn.Linear(train_features.size(1), num_classes).to(device)
    optimizer = torch.optim.SGD(probe.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    batch_size = 256
    for _ in range(epochs):
        permutation = torch.randperm(train_features.size(0))
        for start in range(0, train_features.size(0), batch_size):
            indices = permutation[start : start + batch_size]
            batch_x = train_features[indices].to(device)
            batch_y = train_labels[indices].to(device)

            optimizer.zero_grad()
            logits = probe(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    probe.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, test_features.size(0), batch_size):
            batch_x = test_features[start : start + batch_size].to(device)
            batch_y = test_labels[start : start + batch_size].to(device)
            logits = probe(batch_x)
            predictions = logits.argmax(dim=1)
            correct += (predictions == batch_y).sum().item()
            total += batch_y.numel()
    return correct / total


def run_autoencoder_suite(
    bundle: DatasetBundle,
    specs: Sequence[AutoencoderExperimentSpec],
    *,
    probe_epochs: int = 15,
    probe_lr: float = 0.1,
    noisy_eval_mode: str = "gaussian",
    noisy_eval_strength: float = 0.3,
    device: Optional[torch.device] = None,
    verbose: bool = True,
):
    device = device or get_device()
    rows = []
    artifacts = {}

    for spec in specs:
        model = FlexibleAutoencoder(
            input_dim=bundle.input_dim,
            hidden_dims=spec.hidden_dims,
            latent_dim=spec.latent_dim,
            activation=spec.activation,
            output_activation=spec.output_activation,
        )
        history = train_autoencoder(
            model,
            bundle.train_loader,
            bundle.test_loader,
            epochs=spec.epochs,
            lr=spec.lr,
            weight_decay=spec.weight_decay,
            loss_name=spec.loss_name,
            corruption=spec.corruption,
            corruption_strength=spec.corruption_strength,
            l1_weight=spec.l1_weight,
            kl_weight=spec.kl_weight,
            kl_target=spec.kl_target,
            contractive_weight=spec.contractive_weight,
            device=device,
            verbose=verbose,
        )
        clean_metrics = evaluate_autoencoder(model, bundle.test_loader, device=device, loss_name=spec.loss_name)
        noisy_metrics = evaluate_autoencoder(
            model,
            bundle.test_loader,
            device=device,
            loss_name=spec.loss_name,
            corruption=noisy_eval_mode,
            corruption_strength=noisy_eval_strength,
        )
        probe_accuracy = linear_probe_accuracy(
            model,
            bundle.train_loader,
            bundle.test_loader,
            num_classes=bundle.num_classes,
            device=device,
            epochs=probe_epochs,
            lr=probe_lr,
        )

        row = asdict(spec)
        row.update(
            {
                "parameters": count_parameters(model),
                "test_reconstruction": clean_metrics["loss"],
                "noisy_test_reconstruction": noisy_metrics["loss"],
                "latent_abs_mean": clean_metrics["latent_abs_mean"],
                "latent_zero_ratio": clean_metrics["latent_zero_ratio"],
                "linear_probe_accuracy": probe_accuracy,
            }
        )
        rows.append(row)
        artifacts[spec.name] = {
            "model": model,
            "history": history,
            "clean_metrics": clean_metrics,
            "noisy_metrics": noisy_metrics,
            "probe_accuracy": probe_accuracy,
            "spec": spec,
        }

    return to_python_table(rows), artifacts


def plot_training_curves(history: Dict[str, List[float]], title: str = "Training curves") -> None:
    plt.figure(figsize=(7, 4))
    for key, values in history.items():
        if not values:
            continue
        plt.plot(values, label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()


def plot_reconstructions(
    model: FlexibleAutoencoder,
    loader: DataLoader,
    *,
    device: Optional[torch.device] = None,
    corruption: str = "none",
    corruption_strength: float = 0.0,
    num_images: int = 10,
    title: str = "Reconstructions",
) -> None:
    device = device or get_device()
    model.eval()
    model.to(device)

    images, _ = next(iter(loader))
    images = images[:num_images]
    flat = flatten_images(images).to(device)
    corrupted = corrupt_inputs(flat, corruption, corruption_strength)

    with torch.no_grad():
        reconstructions = model(corrupted).cpu()

    rows = 3 if corruption != "none" and corruption_strength > 0 else 2
    fig, axes = plt.subplots(rows, num_images, figsize=(1.5 * num_images, 1.8 * rows))
    if rows == 2:
        axes[0, 0].set_ylabel("input")
        axes[1, 0].set_ylabel("recon")
    else:
        axes[0, 0].set_ylabel("clean")
        axes[1, 0].set_ylabel("corrupt")
        axes[2, 0].set_ylabel("recon")

    for column in range(num_images):
        clean_image = images[column].squeeze(0).numpy()
        axes[0, column].imshow(clean_image, cmap="gray")
        axes[0, column].axis("off")

        if rows == 3:
            axes[1, column].imshow(corrupted[column].view_as(images[column]).cpu().squeeze(0), cmap="gray")
            axes[1, column].axis("off")
            axes[2, column].imshow(reconstructions[column].view_as(images[column]).squeeze(0), cmap="gray")
            axes[2, column].axis("off")
        else:
            axes[1, column].imshow(reconstructions[column].view_as(images[column]).squeeze(0), cmap="gray")
            axes[1, column].axis("off")

    fig.suptitle(title)
    plt.tight_layout()


def plot_latent_pca(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: Optional[torch.device] = None,
    max_points: int = 1500,
    title: str = "Latent PCA projection",
) -> None:
    features, labels = collect_encoded_features(model, loader, device=device, max_points=max_points)
    features = features - features.mean(dim=0, keepdim=True)
    _, _, principal_components = torch.pca_lowrank(features, q=2)
    projected = features @ principal_components[:, :2]

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(projected[:, 0], projected[:, 1], c=labels, cmap="tab10", s=10, alpha=0.8)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title(title)
    plt.colorbar(scatter)
    plt.tight_layout()


class GreedyAutoencoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation: str = "sigmoid",
        output_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.output_activation = output_activation
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return _apply_activation(self.encoder(x), self.activation)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(x)
        return _apply_activation(self.decoder(hidden), self.output_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reconstruct(x)


def _encode_through_blocks(blocks: Sequence[GreedyAutoencoderBlock], x: torch.Tensor) -> torch.Tensor:
    hidden = x
    for block in blocks:
        hidden = block.encode(hidden)
    return hidden


def pretrain_stacked_autoencoder(
    train_loader: DataLoader,
    input_dim: int,
    hidden_dims: Sequence[int],
    *,
    activation: str = "sigmoid",
    output_activation: str = "sigmoid",
    loss_name: str = "auto",
    epochs_per_layer: int = 10,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    verbose: bool = True,
):
    device = device or get_device()
    effective_loss_name = _resolve_stacked_pretrain_loss(loss_name, activation, output_activation)
    _validate_reconstruction_setup(effective_loss_name, output_activation)
    blocks: List[GreedyAutoencoderBlock] = []
    histories: List[List[float]] = []
    current_dim = input_dim

    for layer_index, hidden_dim in enumerate(hidden_dims):
        block = GreedyAutoencoderBlock(
            current_dim,
            hidden_dim,
            activation=activation,
            output_activation=output_activation,
        ).to(device)
        optimizer = torch.optim.Adam(block.parameters(), lr=lr)
        criterion = _reconstruction_loss(effective_loss_name)
        layer_history = []

        for epoch in range(epochs_per_layer):
            block.train()
            total_loss = 0.0
            total_examples = 0

            for images, _ in train_loader:
                clean = flatten_images(images).to(device)
                with torch.no_grad():
                    previous_representation = _encode_through_blocks(blocks, clean)

                optimizer.zero_grad()
                reconstruction = block(previous_representation)
                loss = criterion(reconstruction, previous_representation)
                loss.backward()
                optimizer.step()

                batch_size = clean.size(0)
                total_examples += batch_size
                total_loss += loss.item() * batch_size

            epoch_loss = total_loss / total_examples
            layer_history.append(epoch_loss)
            if verbose:
                print(
                    f"Pretraining layer {layer_index + 1}/{len(hidden_dims)} | "
                    f"epoch {epoch + 1}/{epochs_per_layer} | loss={epoch_loss:.4f} "
                    f"({effective_loss_name})"
                )

        blocks.append(block)
        histories.append(layer_history)
        current_dim = hidden_dim

    return blocks, histories


class SAEClassifier(nn.Module):
    def __init__(self, blocks: Sequence[GreedyAutoencoderBlock], num_classes: int = 10) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.classifier = nn.Linear(blocks[-1].hidden_dim, num_classes)

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        return _encode_through_blocks(self.blocks, x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.encode_features(x)
        for block in reversed(self.blocks):
            hidden = _apply_activation(block.decoder(hidden), block.output_activation)
        return hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encode_features(x)
        return self.classifier(features)


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        num_classes: int = 10,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.hidden_dims = tuple(hidden_dims)
        self.activation = activation
        self.layers = nn.ModuleList()
        dims = [input_dim, *hidden_dims]
        for in_features, out_features in zip(dims[:-1], dims[1:]):
            self.layers.append(nn.Linear(in_features, out_features))
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x
        for layer in self.layers:
            hidden = _apply_activation(layer(hidden), self.activation)
        return hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode_features(x))


def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    device = device or get_device()
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_examples = 0
    total_correct = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            flat = flatten_images(images).to(device)
            labels = labels.to(device)
            logits = model(flat)
            loss = criterion(logits, labels)

            batch_size = flat.size(0)
            total_examples += batch_size
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def evaluate_feature_statistics(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    features, _ = collect_encoded_features(model, loader, device=device)
    return {
        "latent_abs_mean": features.abs().mean().item(),
        "latent_zero_ratio": (features.abs() < 1e-3).float().mean().item(),
    }


def evaluate_sae_reconstruction(
    model: SAEClassifier,
    loader: DataLoader,
    *,
    device: Optional[torch.device] = None,
    loss_name: str = "bce",
) -> Dict[str, float]:
    device = device or get_device()
    criterion = _reconstruction_loss(loss_name)
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for images, _ in loader:
            flat = flatten_images(images).to(device)
            reconstruction = model.reconstruct(flat)
            loss = criterion(reconstruction, flat)

            batch_size = flat.size(0)
            total_examples += batch_size
            total_loss += loss.item() * batch_size

    return {"loss": total_loss / total_examples}


def plot_sae_reconstructions(
    model: SAEClassifier,
    loader: DataLoader,
    *,
    device: Optional[torch.device] = None,
    num_images: int = 10,
    title: str = "Stacked autoencoder reconstructions",
) -> None:
    device = device or get_device()
    model.eval()
    model.to(device)

    images, _ = next(iter(loader))
    images = images[:num_images]
    flat = flatten_images(images).to(device)

    with torch.no_grad():
        reconstructions = model.reconstruct(flat).cpu()

    fig, axes = plt.subplots(2, num_images, figsize=(1.6 * num_images, 3.6))
    for column in range(num_images):
        axes[0, column].imshow(images[column].squeeze(0), cmap="gray")
        axes[0, column].axis("off")
        axes[1, column].imshow(reconstructions[column].view_as(images[column]).squeeze(0), cmap="gray")
        axes[1, column].axis("off")

    axes[0, 0].set_ylabel("input")
    axes[1, 0].set_ylabel("recon")
    fig.suptitle(title)
    plt.tight_layout()


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    epochs: int = 15,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    freeze_feature_extractor: bool = False,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    device = device or get_device()
    model.to(device)

    if freeze_feature_extractor:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = "classifier" in name
    else:
        for parameter in model.parameters():
            parameter.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_examples = 0

        for images, labels in train_loader:
            flat = flatten_images(images).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(flat)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = flat.size(0)
            total_examples += batch_size
            total_loss += loss.item() * batch_size

        metrics = evaluate_classifier(model, test_loader, device=device)
        history["train_loss"].append(total_loss / total_examples)
        history["test_loss"].append(metrics["loss"])
        history["test_accuracy"].append(metrics["accuracy"])

        if verbose:
            print(
                f"Epoch {epoch + 1:02d}/{epochs:02d} | "
                f"train_loss={history['train_loss'][-1]:.4f} | "
                f"test_acc={history['test_accuracy'][-1]:.4f}"
            )

    return history


def compare_sae_pretraining(
    bundle: DatasetBundle,
    hidden_dims: Sequence[int],
    *,
    activation: str = "sigmoid",
    output_activation: str = "sigmoid",
    pretrain_loss: str = "auto",
    reconstruction_loss: str = "bce",
    pretrain_epochs: int = 8,
    head_epochs: int = 8,
    finetune_epochs: int = 12,
    scratch_epochs: int = 12,
    pretrain_lr: float = 1e-3,
    classifier_lr: float = 1e-3,
    device: Optional[torch.device] = None,
    verbose: bool = True,
):
    device = device or get_device()
    blocks, pretrain_histories = pretrain_stacked_autoencoder(
        bundle.train_loader,
        bundle.input_dim,
        hidden_dims,
        activation=activation,
        output_activation=output_activation,
        loss_name=pretrain_loss,
        epochs_per_layer=pretrain_epochs,
        lr=pretrain_lr,
        device=device,
        verbose=verbose,
    )

    sae_model = SAEClassifier(blocks, num_classes=bundle.num_classes)
    pretrained_snapshot = copy.deepcopy(sae_model)

    pretraining_probe_accuracy = linear_probe_accuracy(
        pretrained_snapshot,
        bundle.train_loader,
        bundle.test_loader,
        num_classes=bundle.num_classes,
        device=device,
        epochs=max(8, head_epochs),
        lr=classifier_lr,
    )
    pretraining_feature_stats = evaluate_feature_statistics(pretrained_snapshot, bundle.test_loader, device=device)
    pretraining_reconstruction = evaluate_sae_reconstruction(
        pretrained_snapshot,
        bundle.test_loader,
        device=device,
        loss_name=reconstruction_loss,
    )

    head_history = train_classifier(
        sae_model,
        bundle.train_loader,
        bundle.test_loader,
        epochs=head_epochs,
        lr=classifier_lr,
        device=device,
        freeze_feature_extractor=True,
        verbose=verbose,
    )
    head_metrics = evaluate_classifier(sae_model, bundle.test_loader, device=device)

    finetune_history = train_classifier(
        sae_model,
        bundle.train_loader,
        bundle.test_loader,
        epochs=finetune_epochs,
        lr=classifier_lr,
        device=device,
        freeze_feature_extractor=False,
        verbose=verbose,
    )
    finetune_metrics = evaluate_classifier(sae_model, bundle.test_loader, device=device)
    finetuned_probe_accuracy = linear_probe_accuracy(
        sae_model,
        bundle.train_loader,
        bundle.test_loader,
        num_classes=bundle.num_classes,
        device=device,
        epochs=max(8, finetune_epochs),
        lr=classifier_lr,
    )
    finetuned_feature_stats = evaluate_feature_statistics(sae_model, bundle.test_loader, device=device)
    finetuned_reconstruction = evaluate_sae_reconstruction(
        sae_model,
        bundle.test_loader,
        device=device,
        loss_name=reconstruction_loss,
    )

    scratch_model = MLPClassifier(
        input_dim=bundle.input_dim,
        hidden_dims=hidden_dims,
        num_classes=bundle.num_classes,
        activation=activation,
    )
    scratch_history = train_classifier(
        scratch_model,
        bundle.train_loader,
        bundle.test_loader,
        epochs=scratch_epochs,
        lr=classifier_lr,
        device=device,
        freeze_feature_extractor=False,
        verbose=verbose,
    )
    scratch_metrics = evaluate_classifier(scratch_model, bundle.test_loader, device=device)

    summary = {
        "activation": activation,
        "output_activation": output_activation,
        "pretrain_loss": _resolve_stacked_pretrain_loss(pretrain_loss, activation, output_activation),
        "reconstruction_loss": reconstruction_loss,
        "hidden_dims": tuple(hidden_dims),
        "pretrained_head_only_accuracy": head_metrics["accuracy"],
        "pretrained_finetuned_accuracy": finetune_metrics["accuracy"],
        "scratch_accuracy": scratch_metrics["accuracy"],
        "pretraining_linear_probe_accuracy": pretraining_probe_accuracy,
        "finetuned_linear_probe_accuracy": finetuned_probe_accuracy,
        "pretraining_latent_zero_ratio": pretraining_feature_stats["latent_zero_ratio"],
        "finetuned_latent_zero_ratio": finetuned_feature_stats["latent_zero_ratio"],
        "pretraining_reconstruction_loss": pretraining_reconstruction["loss"],
        "finetuned_reconstruction_loss": finetuned_reconstruction["loss"],
    }

    return {
        "summary": summary,
        "pretrained_model": pretrained_snapshot,
        "sae_model": sae_model,
        "scratch_model": scratch_model,
        "pretrain_histories": pretrain_histories,
        "head_history": head_history,
        "finetune_history": finetune_history,
        "scratch_history": scratch_history,
        "representation_transition": {
            "linear_probe_accuracy_before": pretraining_probe_accuracy,
            "linear_probe_accuracy_after": finetuned_probe_accuracy,
            "latent_abs_mean_before": pretraining_feature_stats["latent_abs_mean"],
            "latent_abs_mean_after": finetuned_feature_stats["latent_abs_mean"],
            "latent_zero_ratio_before": pretraining_feature_stats["latent_zero_ratio"],
            "latent_zero_ratio_after": finetuned_feature_stats["latent_zero_ratio"],
            "reconstruction_loss_before": pretraining_reconstruction["loss"],
            "reconstruction_loss_after": finetuned_reconstruction["loss"],
        },
    }
