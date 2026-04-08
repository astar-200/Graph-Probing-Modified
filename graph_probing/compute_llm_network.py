from absl import app, flags
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from tqdm import tqdm

from multiprocessing import Process, Queue
import numpy as np
import pandas as pd
import torch

from transformers import AutoConfig
from utils.constants import hf_model_name_map
from utils.model_utils import load_tokenizer_and_model

flags.DEFINE_string("dataset", "openwebtext", "The name of the dataset.")
flags.DEFINE_string("llm_model_name", "gpt2", "The name of the LLM model.")
flags.DEFINE_integer("ckpt_step", -1, "The checkpoint step.")
flags.DEFINE_multi_integer("llm_layer", [0, 1], "Layer IDs for network construction.")
flags.DEFINE_integer("batch_size", 32, "Batch size.")
flags.DEFINE_multi_integer("gpu_id", [0, 1], "The GPU ID.")
flags.DEFINE_integer("num_workers", 30, "Number of processes for computing networks.")
flags.DEFINE_boolean("resume", False, "Resume from the last generation.")
flags.DEFINE_float("network_density", 1.0, "The density of the network.")
flags.DEFINE_string("dataset_path", None, "Optional: Local path to the dataset CSV file. If provided, will load from this path instead of auto-generated path.")
flags.DEFINE_integer("data_start", 0, "Start index of the dataset subset to process.")
flags.DEFINE_integer("data_end", None, "End index of the dataset subset to process. If None, process until the end.")
flags.DEFINE_boolean("all_layers", False, "If True, automatically process all layers of the model. Overrides --llm_layer.")
flags.DEFINE_string("output_dir", None, "Optional: Custom output directory for saving results. If None, use default directory structure.")
FLAGS = flags.FLAGS


def run_llm(
    rank,
    num_producers,
    queue,
    dataset_filename,
    model_name,
    ckpt_step,
    gpu_id,
    batch_size,
    layer_list,
    resume,
    p_save_path,
    dataset_name,
    data_start,
    data_end,
    all_layers,
):
    tokenizer, model = load_tokenizer_and_model(model_name, ckpt_step, gpu_id)
    
    # Auto-detect all layers if requested
    if all_layers:
        if hasattr(model.config, 'num_hidden_layers'):
            num_layers = model.config.num_hidden_layers
        elif hasattr(model.config, 'n_layer'):
            num_layers = model.config.n_layer
        else:
            raise ValueError(f"Cannot determine number of layers for model {model_name}")
        layer_list = list(range(num_layers))
        print(f"[Producer {rank}] Auto-detected {num_layers} layers: {layer_list}")

    data = pd.read_csv(dataset_filename)
    
    if dataset_name == "art":
        original_input_texts = []
        for _, row in data.iterrows():
            prompt = f"When was the release date of {row['creator']}'s {row['title']}?"
            original_input_texts.append(prompt)
    elif dataset_name == "world_place":
        original_input_texts = []
        for _, row in data.iterrows():
            prompt = f"What are the lat/lon coordinates of {row['name']}?"
            original_input_texts.append(prompt)
    else:
        original_input_texts = data["sentences"].to_list()
    
    # Apply data subset selection
    if data_end is not None:
        original_input_texts = original_input_texts[data_start:data_end]
        print(f"[Producer {rank}] Using data subset: [{data_start}:{data_end}] ({len(original_input_texts)} samples)")
    elif data_start > 0:
        original_input_texts = original_input_texts[data_start:]
        print(f"[Producer {rank}] Using data subset: [{data_start}:] ({len(original_input_texts)} samples)")
        
    num_sentences = len(original_input_texts)
    if not resume:
        input_texts = original_input_texts[rank::num_producers]
        # Adjust sentence indices to reflect the actual position in the original dataset
        sentence_indices = list(range(rank + data_start, num_sentences + data_start, num_producers))
    else:
        input_texts = []
        sentence_indices = []
        for local_idx in range(rank, num_sentences, num_producers):
            actual_idx = local_idx + data_start
            if not os.path.exists(f"{p_save_path}/{actual_idx}"):
                input_texts.append(original_input_texts[local_idx])
                sentence_indices.append(actual_idx)

    if len(input_texts) > 0:
        tokenizer.pad_token = tokenizer.eos_token

        with torch.no_grad():
            for i in tqdm(range(0, len(input_texts), batch_size), position=rank, desc=f"Producer {rank}"):
                inputs = tokenizer(input_texts[i:i+batch_size], padding=True, truncation=False, return_tensors="pt")
                model_output = model(
                    input_ids=inputs["input_ids"].to(model.device),
                    attention_mask=inputs["attention_mask"].to(model.device),
                    output_hidden_states=True,
                )
                batch_hidden_states = torch.stack(model_output.hidden_states[1:]).cpu().numpy()  # layer activations (num_layers, B, L, D)
                batch_hidden_states = batch_hidden_states[layer_list]
                batch_attention_mask = inputs["attention_mask"].numpy()  # (B, L)
                actual_batch_size = batch_hidden_states.shape[1]
                batch_sentence_indices = sentence_indices[i:i+actual_batch_size]
                queue.put((batch_hidden_states, batch_attention_mask, batch_sentence_indices))


def run_corr(queue, layer_list, p_save_path, worker_idx, network_density=1.0):
    from torch_geometric.utils import dense_to_sparse
    with torch.no_grad():
        while True:
            batch = queue.get(block=True)
            if batch == "STOP":
                break
            hidden_states, attention_mask, sentence_indices = batch
            for i, sentence_idx in enumerate(sentence_indices):
                p_dir_name = f"{p_save_path}/{sentence_idx}"
                os.makedirs(p_dir_name, exist_ok=True)
                sentence_attention_mask = attention_mask[i]
                for j, layer_idx in enumerate(layer_list):
                    layer_hidden_states = hidden_states[j, i]
                    sentence_hidden_states = layer_hidden_states[sentence_attention_mask == 1].T
                    activation = sentence_hidden_states[:, -1]
                    np.save(f"{p_dir_name}/layer_{layer_idx}_activation.npy", activation)
                    activation_avg = sentence_hidden_states.mean(-1)
                    np.save(f"{p_dir_name}/layer_{layer_idx}_activation_avg.npy", activation_avg)
                    corr = np.corrcoef(sentence_hidden_states)
                    if network_density < 1.0:
                        percentile_threshold = network_density * 100
                        threshold = np.percentile(np.abs(corr), 100 - percentile_threshold)
                        corr[np.abs(corr) < threshold] = 0
                        np.fill_diagonal(corr, 1.0)
                        corr = torch.from_numpy(corr)
                        edge_index, edge_attr = dense_to_sparse(corr)
                        edge_index = edge_index.numpy()
                        edge_attr = edge_attr.numpy()
                        np.save(f"{p_dir_name}/layer_{layer_idx}_sparse_{network_density}_edge_index.npy", edge_index)
                        np.save(f"{p_dir_name}/layer_{layer_idx}_sparse_{network_density}_edge_attr.npy", edge_attr)
                    else:
                        np.save(f"{p_dir_name}/layer_{layer_idx}_corr.npy", corr)

    print(f"Worker {worker_idx} finished processing.")


def main(_):
    model_name = FLAGS.llm_model_name
    hf_model_name = hf_model_name_map.get(model_name, model_name)
    
    # Determine output directory
    if FLAGS.output_dir:
        # Use custom output directory
        dir_name = FLAGS.output_dir
        print(f"Using custom output directory: {dir_name}")
    else:
        # Use default directory structure
        # Extract model name from path if it's a local path
        if os.path.isabs(model_name) or os.path.sep in model_name:
            # If it's a path, use the basename as model name for directory
            model_dir_name = os.path.basename(model_name.rstrip(os.path.sep))
        else:
            # Otherwise use the model name as is
            model_dir_name = model_name
        
        # Add network density to directory name to avoid overwriting different density data
        if FLAGS.network_density < 1.0:
            density_suffix = f"-{FLAGS.network_density}"
        else:
            density_suffix = ""
        
        if FLAGS.ckpt_step == -1:
            dir_name = f"data/graph_probing/{model_dir_name}{density_suffix}/{FLAGS.dataset}"
        else:
            dir_name = f"data/graph_probing/{model_dir_name}{density_suffix}_step{FLAGS.ckpt_step}/{FLAGS.dataset}"
    
    # Load dataset from specified path or auto-generated path
    if FLAGS.dataset_path:
        dataset_filename = FLAGS.dataset_path
    elif FLAGS.dataset == "art":
        dataset_filename = "st_data/art.csv"
    elif FLAGS.dataset == "world_place":
        dataset_filename = "st_data/world_place.csv"
    else:
        revision = "main" if FLAGS.ckpt_step == -1 else f"step{FLAGS.ckpt_step}"
        if hf_model_name.startswith("EleutherAI") and revision != "main":
            dataset_filename = f"data/graph_probing/{FLAGS.dataset}-10k-{FLAGS.llm_model_name}-{revision}.csv"
        else:
            dataset_filename = f"data/graph_probing/{FLAGS.dataset}-10k-{FLAGS.llm_model_name}.csv"

    os.makedirs(dir_name, exist_ok=True)

    # Prepare layer list
    if FLAGS.all_layers:
        # Load model config to get number of layers
        config = AutoConfig.from_pretrained(hf_model_name)
        if hasattr(config, 'num_hidden_layers'):
            num_layers = config.num_hidden_layers
        elif hasattr(config, 'n_layer'):
            num_layers = config.n_layer
        else:
            raise ValueError(f"Cannot determine number of layers for model {hf_model_name}")
        layer_list = list(range(num_layers))
        print(f"All layers mode enabled. Auto-detected {num_layers} layers: {layer_list}")
    else:
        layer_list = FLAGS.llm_layer
        print(f"Processing specified layers: {layer_list}")
    
    # Print data range info
    if FLAGS.data_end is not None:
        print(f"Processing data subset: [{FLAGS.data_start}:{FLAGS.data_end}]")
    elif FLAGS.data_start > 0:
        print(f"Processing data subset: [{FLAGS.data_start}:]")
    else:
        print(f"Processing all data")
    
    queue = Queue()
    producers = []

    for i, gpu_id in enumerate(FLAGS.gpu_id):
        p = Process(
            target=run_llm,
            args=(
                i,
                len(FLAGS.gpu_id),
                queue,
                dataset_filename,
                hf_model_name,
                FLAGS.ckpt_step,
                gpu_id,
                FLAGS.batch_size,
                layer_list,
                FLAGS.resume,
                dir_name,
                FLAGS.dataset,
                FLAGS.data_start,
                FLAGS.data_end,
                FLAGS.all_layers,
            )
        )
        p.start()
        producers.append(p)

    num_workers = FLAGS.num_workers
    consumers = []
    for worker_idx in range(num_workers):
        p = Process(
            target=run_corr,
            args=(queue, layer_list, dir_name, worker_idx, FLAGS.network_density))
        p.start()
        consumers.append(p)

    for producer in producers:
        producer.join()
    for _ in range(num_workers):
        queue.put("STOP")
    for consumer in consumers:
        consumer.join()


if __name__ == "__main__":
    app.run(main)
