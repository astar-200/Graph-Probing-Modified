from absl import app, flags
import itertools
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from tqdm import tqdm

from datasets import load_dataset, load_from_disk
from evaluate import load
perplexity_revised = load("graph_probing/perplexity_revised.py", module_type="metric")
import numpy as np
import pandas as pd
import torch.multiprocessing as mp
from transformers import AutoTokenizer

from utils.constants import hf_model_name_map

# Config for locally stored datasets: path, split name, and the field used as text.
LOCAL_DATASET_CONFIG = {
    "mbpp": {
        "path": "/141nfs/wangjunxiang/graph/llm-graph-probing/dataset/mbpp",
        "split": "train",
        "text_field": "code",
    },
    "opc": {
        "path": "/141nfs/wangjunxiang/graph/llm-graph-probing/dataset/opc",
        "split": "best_of_n",
        "text_field": "solution",
    },
    "cpp-ut-bench": {
        "path": "/141nfs/wangjunxiang/graph/llm-graph-probing/dataset/cpp-ut-bench",
        "split": "train",
        "text_field": "Code",
    },
    "nifty": {
        "path": "/141nfs/wangjunxiang/graph/llm-graph-probing/dataset/nifty",
        "split": "test",
        "text_field": "news",
    },
}

flags.DEFINE_string("dataset", "openwebtext", "The name of the dataset.")
flags.DEFINE_string("llm_model_name", "gpt2", "The name of the LLM model.")
flags.DEFINE_integer("ckpt_step", -1, "The checkpoint step.")
flags.DEFINE_integer("batch_size", 2, "Batch size.")
flags.DEFINE_multi_integer("gpu_id", [0, 1], "The GPU ID.")
flags.DEFINE_string("dataset_path", None, "Optional: Local path to the dataset directory. If provided, will load from local path instead of HuggingFace Hub.")
FLAGS = flags.FLAGS


def split_examples(examples):
    text = examples["text"]
    sentences = text.split("\n\n")

    segments = []
    current_segment = []
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        # If current_segment is not empty and adding the sentence would exceed the limit,
        # finish the current segment and start a new one.
        if current_segment and (current_word_count + sentence_word_count > 800):
            segments.append("\n\n".join(current_segment))
            current_segment = [sentence]
            current_word_count = sentence_word_count
        else:
            current_segment.append(sentence)
            current_word_count += sentence_word_count

    # Append any remaining sentences as the last segment.
    if current_segment:
        segments.append("\n\n".join(current_segment))

    return {"sentences": segments}


def print_token_stats(sentences, model_name, revision):
    """Print token-length distribution of sentences before filtering."""
    if model_name.startswith("EleutherAI"):
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    token_lengths = []
    for enc in tokenizer(sentences, padding=False, truncation=False)["input_ids"]:
        token_lengths.append(len(enc))

    bins = [
        (0,    64,   "[ 0,   64)"),
        (64,   128,  "[64,  128)"),
        (128,  256,  "[128, 256)"),
        (256,  512,  "[256, 512)  <-- kept"),
        (512,  1024, "[512, 1024) <-- kept"),
        (1024, None, "[1024, inf) "),
    ]
    total = len(token_lengths)
    print(f"\n{'='*50}")
    print(f"Token-length distribution (total: {total} sentences)")
    print(f"{'='*50}")
    for lo, hi, label in bins:
        if hi is None:
            count = sum(1 for t in token_lengths if t >= lo)
        else:
            count = sum(1 for t in token_lengths if lo <= t < hi)
        pct = count / total * 100 if total > 0 else 0
        print(f"  {label}: {count:>6} ({pct:5.1f}%)")
    kept = sum(1 for t in token_lengths if 256 <= t < 1024)
    print(f"  -> Will pass token filter [256, 1024): {kept} / {total}")
    print(f"{'='*50}\n")


def run_ppl(rank, queue, model_name, revision, gpu_id, batch_size, all_sentences, p_sentence_indices):
    sentences = [all_sentences[i] for i in p_sentence_indices[rank]]
    if model_name.startswith("EleutherAI"):
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_sentences = tokenizer(sentences, padding=False, truncation=False)
    filtered_sentences = []
    for idx, sentence in enumerate(tqdm(sentences)):
        if 256 <= len(tokenized_sentences["input_ids"][idx]) < 1024:
            filtered_sentences.append(sentence)
    results = perplexity_revised.compute(
        model_id=model_name,
        revision=revision,
        add_start_token=False,
        batch_size=batch_size,
        predictions=filtered_sentences,
        device=f"cuda:{gpu_id[rank]}",
        position=rank,
    )
    perplexities = results["perplexities"]
    queue.put((rank, filtered_sentences, perplexities))


def main(_):
    if FLAGS.dataset_path:
        # Load from local disk
        dataset_dict = load_from_disk(FLAGS.dataset_path)
        # If it's a DatasetDict, get the train split; otherwise use as is
        dataset = dataset_dict["train"] if isinstance(dataset_dict, dict) and "train" in dataset_dict else dataset_dict
    elif FLAGS.dataset == "openwebtext":
        dataset = load_dataset("sam2ai/openwebtext-10k", split="train", streaming=False)
    elif FLAGS.dataset in LOCAL_DATASET_CONFIG:
        config = LOCAL_DATASET_CONFIG[FLAGS.dataset]
        dataset_dict = load_from_disk(config["path"])
        dataset = dataset_dict[config["split"]] if config["split"] in dataset_dict else dataset_dict
        if config["text_field"] != "text":
            if "text" in dataset.column_names:
                dataset = dataset.remove_columns("text")
            dataset = dataset.rename_column(config["text_field"], "text")
    else:
        raise ValueError(f"Unknown dataset: {FLAGS.dataset}")

    dataset = dataset.filter(lambda x: x["text"] != "")
    dataset = dataset.map(split_examples, remove_columns=dataset.column_names)
    all_sentences = list(itertools.chain(*dataset["sentences"]))
    p_sentence_indices = np.array_split(np.arange(len(all_sentences)), len(FLAGS.gpu_id))

    hf_model_name = hf_model_name_map.get(FLAGS.llm_model_name, FLAGS.llm_model_name)
    revision = "main" if FLAGS.ckpt_step == -1 else f"step{FLAGS.ckpt_step}"

    print_token_stats(all_sentences, hf_model_name, revision)

    with mp.Manager() as manager:
        queue = manager.Queue()
        mp.spawn(
            run_ppl,
            args=(queue, hf_model_name, revision, FLAGS.gpu_id, FLAGS.batch_size, all_sentences, p_sentence_indices),
            nprocs=len(FLAGS.gpu_id),
            join=True,
        )
        mp_results = [queue.get() for _ in range(len(FLAGS.gpu_id))]
        print(f"Received {len(mp_results)} results.")

    mp_results.sort(key=lambda x: x[0])
    filtered_sentences = []
    perplexities = []
    for _, sentences, ppls in mp_results:
        filtered_sentences.extend(sentences)
        perplexities.extend(ppls)

    min_ppl = np.percentile(perplexities, 1)
    max_ppl = np.percentile(perplexities, 99)
    print(f"min perplexity: {min_ppl}")
    print(f"max perplexity: {max_ppl}")

    saved_sentences = []
    saved_perplexities = []
    for sentence, ppl in tqdm(zip(filtered_sentences, perplexities)):
        if min_ppl <= ppl <= max_ppl:
            saved_sentences.append(sentence)
            saved_perplexities.append(ppl)

    saved_dataset = {
        "sentences": saved_sentences,
        "perplexities": saved_perplexities
    }

    # Extract model name from path if it's a local path
    if os.path.isabs(FLAGS.llm_model_name) or os.path.sep in FLAGS.llm_model_name:
        model_save_name = os.path.basename(FLAGS.llm_model_name.rstrip(os.path.sep))
    else:
        model_save_name = FLAGS.llm_model_name

    if hf_model_name.startswith("EleutherAI") and revision != "main":
        save_path = f"data/graph_probing/{FLAGS.dataset}-10k-{model_save_name}-{revision}.csv"
    else:
        save_path = f"data/graph_probing/{FLAGS.dataset}-10k-{model_save_name}.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame(saved_dataset)
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    app.run(main)
