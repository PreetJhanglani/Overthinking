import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import pickle as pk
from tqdm import tqdm
from datetime import datetime
import os

def log(msg, logfile):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(logfile, "a") as f:
        f.write(f"[{timestamp}] {msg}\n")
    print(msg)

def main(gpu_id, start_idx, end_idx):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    log_file = f"math500_gpu{gpu_id}_{start_idx}_{end_idx}.log"

    log(f"Starting run on GPU {gpu_id}, samples {start_idx} to {end_idx - 1}", log_file)

    # Load model and tokenizer
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        quantization_config=quantization_config,
        device_map={"": gpu_id},
        trust_remote_code=True
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    ds = load_dataset("HuggingFaceH4/MATH-500")

    temp_out = []

    for i in tqdm(range(start_idx, end_idx), desc=f"GPU {gpu_id}"):
        try:
            input_text = ds['test']['problem'][i]
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

            with torch.no_grad():
                output_dict = model.generate(
                    input_ids,
                    max_new_tokens=5000,
                    do_sample=True,
                    temperature=0.6,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_attentions=False,
                    output_hidden_states=True
                )

            hidden_states = output_dict.hidden_states
            last_10_layers_per_token = [
                torch.stack(token_layers[-10:]).to(torch.float16).cpu()
                for token_layers in hidden_states
            ]

            temp_out.append({
                'unique_id': ds['test']['unique_id'][i],
                'sequences': output_dict.sequences.cpu(),
                'scores': [s.cpu() for s in output_dict.scores],
                'hidden_states': last_10_layers_per_token
            })

            # Save every 10 samples
            if (i + 1 - start_idx) % 10 == 0:
                ckpt_file = f"Math500_gpu{gpu_id}_{i+1}.pkl"
                with open(ckpt_file, 'wb') as f:
                    pk.dump(temp_out, f)
                log(f"Checkpoint saved: {ckpt_file}", log_file)
                temp_out = []

            del output_dict, input_ids, hidden_states, last_10_layers_per_token
            torch.cuda.empty_cache()

        except Exception as e:
            log(f"Error at index {i}: {str(e)}", log_file)

    # Save remaining outputs
    if temp_out:
        final_file = f"Math500_gpu{gpu_id}_{end_idx}_final.pkl"
        with open(final_file, 'wb') as f:
            pk.dump(temp_out, f)
        log(f"Final checkpoint saved: {final_file}", log_file)

    log(f"Finished run on GPU {gpu_id}", log_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MATH-500 generation on specific GPU and sample range.")
    parser.add_argument("--gpu_id", type=int, required=True, help="GPU ID to run the model on (0, 1, 2...)")
    parser.add_argument("--start_idx", type=int, required=True, help="Start index in dataset")
    parser.add_argument("--end_idx", type=int, required=True, help="End index in dataset (exclusive)")
    args = parser.parse_args()

    main(args.gpu_id, args.start_idx, args.end_idx)
 