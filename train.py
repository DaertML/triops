import argparse
import torch
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType # Only needed for LoRA
import os
import json
import yaml # New import for YAML parsing

# --- Utility Functions ---

def check_mps_availability():
    """
    Checks for Apple Silicon (MPS) availability and sets the device.
    Returns:
        str: "mps" if available, "cpu" otherwise.
        torch.dtype: torch.bfloat16 if supported and MPS is available, else torch.float32.
    """
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                  "and/or you do not have an MPS-enabled device.")
        print("Falling back to CPU training. WARNING: Fine-tuning on CPU will be extremely slow.")
        return "cpu", torch.float32
    else:
        device = "mps"
        print(f"Using Apple Silicon (MPS) for training. Device: {device}")
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0' # Helps with memory management on MPS

        torch_dtype = torch.float32
        try:
            if torch.backends.mps.is_bf16_supported():
                torch_dtype = torch.bfloat16
                print("MPS bfloat16 is supported and will be used.")
            else:
                print("MPS bfloat16 not supported or detected. Using float32.")
        except AttributeError:
            print("`torch.backends.mps.is_bf16_supported()` not found. "
                  "Consider updating PyTorch to nightly for bfloat16 support. Using float32.")
        return device, torch_dtype

def load_and_process_datasets(dataset_configs: list, tokenizer, max_train_samples: int = None):
    """
    Loads, formats, and tokenizes datasets based on configurations.

    Args:
        dataset_configs (list): A list of dictionaries, where each dict specifies
                                {'name': 'dataset_name', 'input_column': 'col_name', 'output_column': 'col_name'}.
                                'output_column' is optional.
        tokenizer: The Hugging Face tokenizer to use.
        max_train_samples (int, optional): If specified, limits the total number of training samples
                                           across all combined datasets. Defaults to None (use all samples).

    Returns:
        Dataset: The combined and tokenized dataset.
    """
    print(f"Loading and combining datasets based on configurations: {dataset_configs}...")
    
    processed_datasets = [] 

    for ds_config in dataset_configs:
        ds_name = ds_config['name']
        input_col = ds_config.get('input_column')
        output_col = ds_config.get('output_column')
        
        print(f"\n--- Processing dataset: '{ds_name}' (Input: '{input_col}', Output: '{output_col}') ---")
        try:
            current_dataset = load_dataset(ds_name, split="train")
            print(f"  - Loaded '{ds_name}' (train split). Num examples: {len(current_dataset)}")
        except Exception as e:
            print(f"  - Could not load 'train' split for '{ds_name}'. Trying to load entire dataset. Error: {e}")
            try:
                current_dataset = load_dataset(ds_name)
                if isinstance(current_dataset, DatasetDict):
                    if "train" in current_dataset:
                        current_dataset = current_dataset["train"]
                        print(f"    - Loaded '{ds_name}' (using 'train' split). Num examples: {len(current_dataset)}")
                    elif len(current_dataset.keys()) > 0:
                        first_split = list(current_dataset.keys())[0]
                        current_dataset = current_dataset[first_split]
                        print(f"    - Loaded '{ds_name}' (using first available split: '{first_split}'). Num examples: {len(current_dataset)}")
                    else:
                        print(f"    - '{ds_name}' is a DatasetDict but has no usable splits. Skipping.")
                        continue
            except Exception as inner_e:
                print(f"  - Failed to load dataset '{ds_name}' completely. Skipping. Error: {inner_e}")
                continue

        def format_batch_for_current_dataset(examples_batch):
            formatted_texts = []
            num_examples_in_batch = len(next(iter(examples_batch.values()))) 
            
            for i in range(num_examples_in_batch):
                example = {col: examples_batch[col][i] for col in examples_batch}

                # Prioritize user-specified columns
                if input_col and input_col in example and example[input_col] is not None:
                    if output_col and output_col in example and example[output_col] is not None:
                        # Special handling for 'messages' column, which is a list of dicts for ultrachat_200k
                        if input_col == 'messages' and isinstance(example[input_col], list):
                            formatted_message = ""
                            for message_obj in example[input_col]:
                                role = message_obj.get("role", "user") 
                                content = message_obj.get("content", "")
                                if role == "user":
                                    formatted_message += f"<s>[INST] {content} [/INST]"
                                elif role == "assistant":
                                    formatted_message += f" {content}</s>"
                            formatted_texts.append(formatted_message.strip())
                        else:
                            formatted_texts.append(f"<s>[INST] {example[input_col]} [/INST] {example[output_col]}</s>")
                    else:
                        # Pure text generation (only input column)
                        if input_col == 'messages' and isinstance(example[input_col], list):
                            formatted_message = ""
                            for message_obj in example[input_col]:
                                content = message_obj.get("content", "")
                                formatted_message += f"{content} "
                            formatted_texts.append(formatted_message.strip())
                        else:
                            formatted_texts.append(example[input_col])
                
                # Fallback to common LLM formats if user-specified columns not found or incomplete
                elif "text" in example and example["text"] is not None:
                    formatted_texts.append(example["text"])
                elif "instruction" in example and "response" in example and \
                     example["instruction"] is not None and example["response"] is not None:
                    formatted_texts.append(f"<s>[INST] {example['instruction']} [/INST] {example['response']}</s>")
                elif "prompt" in example and "completion" in example and \
                     example["prompt"] is not None and example["completion"] is not None:
                    formatted_texts.append(f"{example['prompt']}{example['completion']}")
                elif "question" in example and "answer" in example and \
                     example["question"] is not None and example["answer"] is not None:
                    formatted_texts.append(f"Question: {example['question']}\nAnswer: {example['answer']}")
                else:
                    found_string_col = False
                    for key, value in example.items():
                        if isinstance(value, str) and len(value) > 0:
                            formatted_texts.append(value)
                            found_string_col = True
                            break
                    if not found_string_col:
                        formatted_texts.append(None) 

            return {"text": formatted_texts}

        print(f"  - Applying formatting and tokenization for '{ds_name}'...")
        
        current_dataset_filtered = current_dataset.filter(
            lambda ex: (input_col and input_col in ex and ex[input_col] is not None) or \
                       ("text" in ex and ex["text"] is not None) or \
                       ("instruction" in ex and "response" in ex and ex["instruction"] is not None and ex["response"] is not None) or \
                       ("prompt" in ex and "completion" in ex and ex["prompt"] is not None and ex["completion"] is not None) or \
                       ("question" in ex and "answer" in ex and ex["question"] is not None and ex["answer"] is not None) or \
                       any(isinstance(v, str) and len(v) > 0 for v in ex.values()) or \
                       (input_col == 'messages' and input_col in ex and isinstance(ex[input_col], list) and len(ex[input_col]) > 0)
        )
        
        if len(current_dataset_filtered) == 0:
            print(f"  - Warning: Dataset '{ds_name}' became empty after initial filtering. Skipping this dataset.")
            continue

        ds_with_text = current_dataset_filtered.map(
            format_batch_for_current_dataset,
            batched=True,
            remove_columns=[col for col in current_dataset_filtered.column_names if col != 'text']
        )
        
        ds_with_text = ds_with_text.filter(lambda example: example.get("text") is not None and len(example["text"].strip()) > 0)
        
        if len(ds_with_text) == 0:
            print(f"  - Warning: Dataset '{ds_name}' became empty after text formatting and secondary filtering. Skipping this dataset.")
            continue

        processed_datasets.append(ds_with_text)
        print(f"    - Processed '{ds_name}' has {len(ds_with_text)} examples.")

    if not processed_datasets:
        raise ValueError("No datasets were successfully loaded and processed for training. Please check dataset names and column configurations.")
        
    raw_combined_dataset = concatenate_datasets(processed_datasets)
    print(f"Total examples in combined dataset before final tokenization: {len(raw_combined_dataset)}")

    # Limit the number of training samples if specified
    if max_train_samples is not None and max_train_samples < len(raw_combined_dataset):
        print(f"Limiting training dataset to {max_train_samples} samples.")
        raw_combined_dataset = raw_combined_dataset.select(range(max_train_samples))
        print(f"Total examples after limiting: {len(raw_combined_dataset)}")

    def tokenize_function(examples):
        texts = examples["text"]
        valid_texts = [text for text in texts if text is not None and len(text.strip()) > 0]
        
        if not valid_texts:
            return {"input_ids": [], "attention_mask": [], "labels": []} # Ensure labels is also empty if no valid texts

        tokenized_output = tokenizer(valid_texts, truncation=True, max_length=512, padding="max_length" if len(valid_texts) > 1 else False)
        
        # Ensure lists for single example outputs if padding is False
        if not (len(valid_texts) > 1 and "max_length" in tokenizer.padding_side):
            for key in tokenized_output:
                if not isinstance(tokenized_output[key], list):
                    tokenized_output[key] = [tokenized_output[key]]

        # Add labels which are a copy of input_ids for causal language modeling
        tokenized_output["labels"] = tokenized_output["input_ids"].copy() 
        return tokenized_output

    tokenized_dataset = raw_combined_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Remove original columns that are not needed by the model
    tokenized_dataset = tokenized_dataset.remove_columns([col for col in tokenized_dataset.column_names if col not in tokenizer.model_input_names and col != "labels"])
    
    return tokenized_dataset

# --- Fine-tuning Functions ---

def train_llm_full_finetune(config: dict):
    """
    Performs full fine-tuning of an LLM with a list of datasets on Apple Silicon.

    Args:
        config (dict): Configuration dictionary containing model_name, dataset_configs, output_dir,
                       and training_arguments.
    """
    model_name = config['training']['model_name']
    output_dir = config['training']['output_dir']
    dataset_configs = config['datasets']
    training_args_config = config['training_arguments']

    device, torch_dtype = check_mps_availability()

    # 1. Load Tokenizer and Model
    print(f"Loading tokenizer and model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    model.train()

    # 2. Load and Process Datasets
    tokenized_dataset = load_and_process_datasets(dataset_configs, tokenizer)

    # 3. Define Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_args_config.get('num_train_epochs', 3),
        per_device_train_batch_size=training_args_config.get('per_device_train_batch_size', 1),
        gradient_accumulation_steps=training_args_config.get('gradient_accumulation_steps', 16),
        # Ensure learning_rate is cast to float
        learning_rate=float(training_args_config.get('learning_rate', 5e-5)), # Default for full fine-tune
        logging_dir=f"{output_dir}/logs",
        logging_steps=training_args_config.get('logging_steps', 50),
        save_steps=training_args_config.get('save_steps', 200),
        save_total_limit=training_args_config.get('save_total_limit', 1),
        push_to_hub=False,
        report_to="none",
        bf16=torch_dtype == torch.bfloat16, 
        fp16=False,
        dataloader_num_workers=training_args_config.get('dataloader_num_workers', 0),
        dataloader_pin_memory=training_args_config.get('dataloader_pin_memory', False),
        # Removed gradient_checkpointing to fix the TypeError
        # gradient_checkpointing=training_args_config.get('gradient_checkpointing', True), 
        label_names=["labels"] # Ensure labels are recognized
    )

    # 4. Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # 5. Train the model
    print("Starting full fine-tuning...")
    print(f"Training parameters: {model.num_parameters()} total parameters.")
    print(f"Trainable parameters (all of them in full finetune): {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    trainer.train()

    # 6. Save the trained model locally
    print(f"Saving the fully fine-tuned model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Full fine-tuning complete and model saved locally.")

def train_llm_lora_finetune(config: dict):
    """
    Performs LoRA fine-tuning of an LLM with a list of datasets on Apple Silicon.

    Args:
        config (dict): Configuration dictionary containing model_name, dataset_configs, output_dir,
                       max_train_samples, lora_config, and training_arguments.
    """
    model_name = config['training']['model_name']
    output_dir = config['training']['output_dir']
    dataset_configs = config['datasets']
    max_train_samples = config['training'].get('max_train_samples')
    lora_config_params = config.get('lora_config', {})
    training_args_config = config['training_arguments']

    device, torch_dtype = check_mps_availability()

    # 1. Load Tokenizer and Model
    print(f"Loading tokenizer and model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device, 
    )
    
    # 2. Configure and Apply LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=lora_config_params.get('r', 16), 
        lora_alpha=lora_config_params.get('lora_alpha', 32), 
        target_modules=lora_config_params.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"]), 
        lora_dropout=lora_config_params.get('lora_dropout', 0.05), 
        bias=lora_config_params.get('bias', "none"), 
        task_type=TaskType.CAUSAL_LM, 
    )

    model = get_peft_model(model, lora_config)
    
    model.enable_input_require_grads() # Required for some models with LoRA
    
    model.print_trainable_parameters()
    
    model.train() 

    # 3. Load and Process Datasets
    tokenized_dataset = load_and_process_datasets(dataset_configs, tokenizer, max_train_samples)

    # 4. Define Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_args_config.get('num_train_epochs', 1),
        per_device_train_batch_size=training_args_config.get('per_device_train_batch_size', 1), 
        gradient_accumulation_steps=training_args_config.get('gradient_accumulation_steps', 16), 
        # Ensure learning_rate is cast to float
        learning_rate=float(training_args_config.get('learning_rate', 2e-4)), # Default for LoRA
        logging_dir=f"{output_dir}/logs",
        logging_steps=training_args_config.get('logging_steps', 50),
        save_steps=training_args_config.get('save_steps', 200),
        save_total_limit=training_args_config.get('save_total_limit', 1),
        push_to_hub=False,
        report_to="none",
        bf16=torch_dtype == torch.bfloat16, 
        fp16=False,
        dataloader_num_workers=training_args_config.get('dataloader_num_workers', 0), 
        dataloader_pin_memory=training_args_config.get('dataloader_pin_memory', False), 
        # Removed gradient_checkpointing to fix the TypeError
        # gradient_checkpointing=training_args_config.get('gradient_checkpointing', True),
        label_names=["labels"]
    )

    # 5. Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # 6. Train the model
    print("Starting LoRA fine-tuning...")
    print(f"Training parameters (LoRA adapters only): {sum(p.numel() for p in model.parameters() if p.requires_grad)} total trainable parameters.")

    trainer.train()

    # 7. Save the trained LoRA adapters locally
    print(f"Saving the LoRA adapters to {output_dir}...")
    trainer.model.save_pretrained(output_dir) 
    tokenizer.save_pretrained(output_dir)

    print("LoRA fine-tuning complete and adapters saved locally.")

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune an LLM on Apple Silicon using Hugging Face via a YAML configuration.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the YAML configuration file.")

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found at '{args.config}'")
        exit(1)

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration file: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while reading the config file: {e}")
        exit(1)

    # Validate top-level keys
    required_top_level_keys = ['training', 'datasets', 'training_arguments']
    for key in required_top_level_keys:
        if key not in config:
            print(f"Error: Missing required top-level key '{key}' in the configuration file.")
            exit(1)

    # Validate 'training' section
    if 'type' not in config['training'] or config['training']['type'] not in ['full', 'lora']:
        print("Error: 'training.type' must be specified as either 'full' or 'lora'.")
        exit(1)
    if 'model_name' not in config['training']:
        print("Error: 'training.model_name' must be specified.")
        exit(1)
    if 'output_dir' not in config['training']:
        print("Error: 'training.output_dir' must be specified.")
        exit(1)

    # Validate 'datasets' section
    if not isinstance(config['datasets'], list) or not config['datasets']:
        print("Error: 'datasets' must be a non-empty list of dataset configurations.")
        exit(1)
    for ds_config in config['datasets']:
        if 'name' not in ds_config:
            print(f"Error: Each dataset configuration in 'datasets' must have a 'name' field. Found: {ds_config}")
            exit(1)

    training_type = config['training']['type']
    print(f"Starting LLM fine-tuning of type: {training_type.upper()}")

    if training_type == "full":
        train_llm_full_finetune(config)
    elif training_type == "lora":
        # Ensure lora_config is present if training_type is lora
        if 'lora_config' not in config:
            print("Error: 'lora_config' section is required when 'training.type' is 'lora'.")
            exit(1)
        train_llm_lora_finetune(config)
    else:
        print(f"Unknown training type: {training_type}. Must be 'full' or 'lora'.")
        exit(1)
