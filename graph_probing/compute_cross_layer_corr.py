"""
Compute average cross-layer correlation matrices across a dataset.

For each pair of layers (i, j) where i < j, this script computes the
Pearson correlation between neurons in layer i and neurons in layer j,
averaged over the first N sentences of the dataset using Fisher r-to-z
aggregation: each per-sentence correlation matrix is transformed via
z = arctanh(r) before accumulation, and the final average is mapped back
to r via r = tanh(z_mean).  This yields a statistically correct mean
correlation rather than a naive arithmetic mean of r values.

The result for pair (j, i) can be obtained by transposing the (i, j) matrix.
Total matrices saved: num_layers * (num_layers - 1) / 2
"""

from absl import app, flags
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch

from utils.constants import hf_model_name_map
from utils.model_utils import load_tokenizer_and_model

flags.DEFINE_string("dataset", "openwebtext", "The name of the dataset.")
flags.DEFINE_string("llm_model_name", "gpt2", "The name of the LLM model.")
flags.DEFINE_integer("ckpt_step", -1, "The checkpoint step.")
flags.DEFINE_integer("batch_size", 2, "Batch size.")
flags.DEFINE_integer("gpu_id", 0, "The GPU ID.")
flags.DEFINE_string("dataset_path", None, "Local path to the dataset CSV file.")
flags.DEFINE_integer("start_idx", 0, "Index of the first sentence to process (inclusive).")
flags.DEFINE_integer("num_sentences", 100, "Number of sentences to process starting from --start_idx.")
flags.DEFINE_string("output_dir", None, "Output directory. If None, uses default structure.")
flags.DEFINE_boolean("within_layer", False, "If True, compute within-layer neuron correlation (averaged over sentences).")
flags.DEFINE_boolean("full_matrix", False, "If True, compute the full (D*num_layers)x(D*num_layers) block correlation matrix across all layer pairs (includes both within-layer and cross-layer). Saves a single file: full_avg_corr.npy.")
flags.DEFINE_boolean("no_fisher", False, "If True, use plain arithmetic mean of r values instead of Fisher r-to-z averaging.")
FLAGS = flags.FLAGS


_FISHER_CLIP = 1.0 - 1e-7  # clip r before arctanh to avoid ±∞


def fisher_r_to_z(r: np.ndarray) -> np.ndarray:
    """Fisher r-to-z transformation: z = arctanh(r).

    Values are clipped to (-(1-1e-7), 1-1e-7) before applying arctanh so that
    exact ±1 entries (e.g. the diagonal of a within-layer corrcoef matrix) do
    not produce ±∞.  After averaging in z-space and inverting with tanh, the
    diagonal will still round-trip to values indistinguishable from ±1.
    """
    return np.arctanh(np.clip(r, -_FISHER_CLIP, _FISHER_CLIP))


def fisher_z_to_r(z: np.ndarray) -> np.ndarray:
    """Inverse Fisher transformation: r = tanh(z)."""
    return np.tanh(z)


def cross_corrcoef(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the cross-correlation matrix between rows of x and rows of y.

    Args:
        x: (D, L) array — D neurons in layer i, each with L token activations.
        y: (D, L) array — D neurons in layer j, each with L token activations.

    Returns:
        (D, D) matrix where entry [a, b] = Pearson corr(x[a], y[b]).
        To get corr(y[b], x[a]) for all pairs, simply transpose the result.
    """
    # Center rows
    x_c = x - x.mean(axis=1, keepdims=True)
    y_c = y - y.mean(axis=1, keepdims=True)

    # Compute standard deviations, guard against zero-variance neurons
    x_std = x_c.std(axis=1, keepdims=True)
    y_std = y_c.std(axis=1, keepdims=True)
    x_std = np.where(x_std < 1e-8, 1e-8, x_std)
    y_std = np.where(y_std < 1e-8, 1e-8, y_std)

    x_n = x_c / x_std  # (D, L)
    y_n = y_c / y_std  # (D, L)

    L = x.shape[1]
    return (x_n @ y_n.T) / L  # (D, D)


def main(_):
    model_name = FLAGS.llm_model_name
    hf_model_name = hf_model_name_map.get(model_name, model_name)

    # ------------------------------------------------------------------ #
    # Output directory
    # ------------------------------------------------------------------ #
    if FLAGS.output_dir:
        dir_name = FLAGS.output_dir
    else:
        model_dir_name = (
            os.path.basename(model_name.rstrip(os.path.sep))
            if (os.path.isabs(model_name) or os.path.sep in model_name)
            else model_name
        )
        if FLAGS.ckpt_step == -1:
            dir_name = f"data/cross_layer_corr/{model_dir_name}/{FLAGS.dataset}"
        else:
            dir_name = f"data/cross_layer_corr/{model_dir_name}_step{FLAGS.ckpt_step}/{FLAGS.dataset}"

    os.makedirs(dir_name, exist_ok=True)
    print(f"Results will be saved to: {dir_name}")

    # ------------------------------------------------------------------ #
    # Dataset
    # ------------------------------------------------------------------ #
    if FLAGS.dataset_path:
        dataset_filename = FLAGS.dataset_path
    elif FLAGS.dataset == "art":
        dataset_filename = "st_data/art.csv"
    elif FLAGS.dataset == "world_place":
        dataset_filename = "st_data/world_place.csv"
    else:
        revision = "main" if FLAGS.ckpt_step == -1 else f"step{FLAGS.ckpt_step}"
        if hf_model_name.startswith("EleutherAI") and revision != "main":
            dataset_filename = (
                f"data/graph_probing/{FLAGS.dataset}-10k-"
                f"{FLAGS.llm_model_name}-{revision}.csv"
            )
        else:
            dataset_filename = (
                f"data/graph_probing/{FLAGS.dataset}-10k-{FLAGS.llm_model_name}.csv"
            )

    data = pd.read_csv(dataset_filename)

    if FLAGS.dataset == "art":
        input_texts = [
            f"When was the release date of {row['creator']}'s {row['title']}?"
            for _, row in data.iterrows()
        ]
    elif FLAGS.dataset == "world_place":
        input_texts = [
            f"What are the lat/lon coordinates of {row['name']}?"
            for _, row in data.iterrows()
        ]
    else:
        input_texts = data["sentences"].to_list()

    start_idx = FLAGS.start_idx
    end_idx = start_idx + FLAGS.num_sentences
    input_texts = input_texts[start_idx:end_idx]
    num_sentences = len(input_texts)
    print(f"Processing sentences [{start_idx}, {start_idx + num_sentences}) from '{dataset_filename}'")

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    tokenizer, model = load_tokenizer_and_model(hf_model_name, FLAGS.ckpt_step, FLAGS.gpu_id)
    tokenizer.pad_token = tokenizer.eos_token

    if hasattr(model.config, "num_hidden_layers"):
        num_layers = model.config.num_hidden_layers
    elif hasattr(model.config, "n_layer"):
        num_layers = model.config.n_layer
    else:
        raise ValueError(f"Cannot determine the number of layers for model '{model_name}'")

    num_pairs = num_layers * (num_layers - 1) // 2
    print(f"Model: {model_name}  |  layers: {num_layers}  |  layer pairs: {num_pairs}")

    # ------------------------------------------------------------------ #
    # Choose accumulation / averaging strategy based on --no_fisher flag.
    #
    # Default (Fisher):  accumulate arctanh(r), average, then tanh back to r.
    # --no_fisher:       accumulate r directly, divide by count to get mean r.
    #
    # Accumulators store either z-values (Fisher) or r-values (arithmetic):
    #   sum_corr[(i, j)]  — cross-layer (D, D) sum, for i < j
    #   sum_within[i]     — within-layer (D, D) sum, for each layer i
    #   sum_full          — full (D*L, D*L) block matrix sum (--full_matrix)
    # All lazily initialized on first use.
    # ------------------------------------------------------------------ #
    to_accum = (lambda r: r) if FLAGS.no_fisher else fisher_r_to_z
    to_avg   = (lambda s: s / count) if FLAGS.no_fisher else (lambda s: fisher_z_to_r(s / count))
    sum_corr: dict[tuple[int, int], np.ndarray] = {}
    sum_within: dict[int, np.ndarray] = {}
    sum_full: np.ndarray | None = None
    count = 0

    batch_size = FLAGS.batch_size

    with torch.no_grad():
        for batch_start in tqdm(range(0, num_sentences, batch_size), desc="Batches"):
            batch_texts = input_texts[batch_start : batch_start + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            model_output = model(
                input_ids=inputs["input_ids"].to(model.device),
                attention_mask=inputs["attention_mask"].to(model.device),
                output_hidden_states=True,
            )

            # hidden_states: tuple of (B, L, D), index 0 = embedding layer
            # Stack layers 1..num_layers → (num_layers, B, L, D)
            hidden_states = (
                torch.stack(model_output.hidden_states[1:]).cpu().numpy()
            )
            attention_mask = inputs["attention_mask"].numpy()  # (B, L)
            actual_batch_size = hidden_states.shape[1]

            for b in range(actual_batch_size):
                mask = attention_mask[b]  # (L,)

                # Per-layer valid-token hidden states: list of (D, valid_L)
                sentence_states = [
                    hidden_states[layer_idx, b][mask == 1].T  # (D, valid_L)
                    for layer_idx in range(num_layers)
                ]

                if FLAGS.full_matrix:
                    # Build the full block matrix for this sentence (in r-space),
                    # apply Fisher r-to-z element-wise, then accumulate in z-space.
                    D = sentence_states[0].shape[0]
                    full = np.zeros((D * num_layers, D * num_layers), dtype=np.float32)
                    for i in range(num_layers):
                        si, ei = i * D, (i + 1) * D
                        # Diagonal block: within-layer correlation
                        full[si:ei, si:ei] = np.corrcoef(sentence_states[i])
                        # Off-diagonal blocks: cross-layer correlation
                        for j in range(i + 1, num_layers):
                            sj, ej = j * D, (j + 1) * D
                            cc = cross_corrcoef(sentence_states[i], sentence_states[j])
                            full[si:ei, sj:ej] = cc        # upper triangle
                            full[sj:ej, si:ei] = cc.T      # lower triangle (transpose)
                    full_accum = to_accum(full)
                    if sum_full is None:
                        sum_full = full_accum
                    else:
                        sum_full += full_accum

                elif FLAGS.within_layer:
                    # Accumulate within-layer correlation in Fisher z-space
                    for i in range(num_layers):
                        wc = np.corrcoef(sentence_states[i])  # (D, D)
                        wc_accum = to_accum(wc)
                        if i not in sum_within:
                            sum_within[i] = wc_accum.copy()
                        else:
                            sum_within[i] += wc_accum
                else:
                    # Accumulate cross-correlation in Fisher z-space for every (i, j) pair with i < j
                    for i in range(num_layers):
                        for j in range(i + 1, num_layers):
                            cc = cross_corrcoef(sentence_states[i], sentence_states[j])
                            cc_accum = to_accum(cc)
                            if (i, j) not in sum_corr:
                                sum_corr[(i, j)] = cc_accum.copy()
                            else:
                                sum_corr[(i, j)] += cc_accum

                count += 1

    total_matrices = len(sum_corr) + len(sum_within) + (1 if sum_full is not None else 0)
    print(f"\nFinished processing {count} sentences. Saving {total_matrices} matrix file(s) …")

    avg_method = "arithmetic mean" if FLAGS.no_fisher else "Fisher r-to-z mean"
    print(f"  Averaging method      : {avg_method}")

    if FLAGS.full_matrix:
        avg_full = to_avg(sum_full)
        save_path = f"{dir_name}/full_avg_corr.npy"
        np.save(save_path, avg_full)
        D = avg_full.shape[0] // num_layers
        print(f"  Full block matrix     : shape {avg_full.shape}  (D={D}, layers={num_layers})")
        print(f"  Saved to              : {save_path}")
    elif FLAGS.within_layer:
        for i, accum in tqdm(sum_within.items(), desc="Saving within-layer"):
            avg = to_avg(accum)
            np.save(f"{dir_name}/layer_{i}_avg_corr.npy", avg)
        print(f"  Within-layer matrices : {len(sum_within)}  →  layer_{{i}}_avg_corr.npy")
    else:
        for (i, j), accum in tqdm(sum_corr.items(), desc="Saving cross-layer"):
            avg = to_avg(accum)
            np.save(f"{dir_name}/cross_layer_{i}_{j}_avg_corr.npy", avg)
        print(f"  Cross-layer pairs : {len(sum_corr)}  (i < j only)  →  cross_layer_{{i}}_{{j}}_avg_corr.npy")
        print(f"  To load pair (j,i): np.load('cross_layer_i_j_avg_corr.npy').T")

    # Save metadata alongside the matrices
    np.save(
        f"{dir_name}/metadata.npy",
        {
            "num_sentences": count,
            "num_layers": num_layers,
            "model_name": model_name,
            "dataset": FLAGS.dataset,
            "within_layer": FLAGS.within_layer,
            "full_matrix": FLAGS.full_matrix,
            "averaging": avg_method,
            "note": (
                "full_avg_corr.npy: block matrix of shape (D*L, D*L). "
                "Block [i,j] (each D×D) = cross-corr(layer_i, layer_j). "
                "Diagonal blocks = within-layer corr. "
                "All values averaged using Fisher r-to-z transformation."
            ),
        },
    )

    print(f"Done. All files saved to: {dir_name}")


if __name__ == "__main__":
    app.run(main)
