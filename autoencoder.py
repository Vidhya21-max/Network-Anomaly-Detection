"""
Autoencoder model for unsupervised anomaly detection.
Built with TensorFlow/Keras: Encoder -> Bottleneck -> Decoder.
Trained only on normal data; reconstruction error is the anomaly score.
"""
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import AUTOENCODER_CONFIG, MODEL_DIR


def build_autoencoder(
    input_dim: int,
    latent_dim: int = None,
    hidden_dims: list = None,
    activation: str = "relu",
    output_activation: str = "linear",
    learning_rate: float = None,
):
    """Build and compile the autoencoder."""
    cfg = AUTOENCODER_CONFIG
    latent_dim = latent_dim or cfg["latent_dim"]
    hidden_dims = hidden_dims or cfg["hidden_dims"]
    learning_rate = learning_rate or cfg["learning_rate"]

    # Encoder
    encoder_input = keras.Input(shape=(input_dim,), name="input")
    x = encoder_input
    for i, dim in enumerate(hidden_dims):
        x = keras.layers.Dense(dim, activation=activation, name=f"enc_dense_{i}")(x)
    bottleneck = keras.layers.Dense(latent_dim, activation=activation, name="bottleneck")(x)
    encoder = keras.Model(encoder_input, bottleneck, name="encoder")

    # Decoder (mirror)
    decoder_input = keras.Input(shape=(latent_dim,), name="decoder_input")
    x = decoder_input
    for i, dim in enumerate(reversed(hidden_dims)):
        x = keras.layers.Dense(dim, activation=activation, name=f"dec_dense_{i}")(x)
    decoder_output = keras.layers.Dense(input_dim, activation=output_activation, name="output")(x)
    decoder = keras.Model(decoder_input, decoder_output, name="decoder")

    # Full model
    autoencoder = keras.Model(
        encoder_input,
        decoder(encoder(encoder_input)),
        name="autoencoder"
    )
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mse"],
    )
    return autoencoder, encoder, decoder


def train_autoencoder(
    X_train_norm: np.ndarray,
    epochs: int = None,
    batch_size: int = None,
    latent_dim: int = None,
    hidden_dims: list = None,
    learning_rate: float = None,
    validation_split: float = 0.1,
    verbose: int = 1,
):
    """Train autoencoder on normal data only."""
    cfg = AUTOENCODER_CONFIG
    input_dim = X_train_norm.shape[1]
    autoencoder, encoder, decoder = build_autoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim or cfg["latent_dim"],
        hidden_dims=hidden_dims or cfg["hidden_dims"],
        learning_rate=learning_rate or cfg["learning_rate"],
    )
    autoencoder.fit(
        X_train_norm,
        X_train_norm,
        epochs=epochs or cfg["epochs"],
        batch_size=batch_size or cfg["batch_size"],
        validation_split=validation_split,
        verbose=verbose,
    )
    return autoencoder, encoder, decoder


def reconstruction_errors(model: keras.Model, X: np.ndarray) -> np.ndarray:
    """Compute per-sample MSE reconstruction error."""
    X_recon = model.predict(X, verbose=0)
    return np.mean((X - X_recon) ** 2, axis=1)


def save_model(autoencoder: keras.Model, path: Path = None):
    """Save autoencoder to disk."""
    path = path or MODEL_DIR / "autoencoder"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    autoencoder.save(path)
    return path


def load_model(path: Path = None) -> keras.Model:
    """Load autoencoder from disk."""
    path = path or MODEL_DIR / "autoencoder"
    return keras.models.load_model(path)
