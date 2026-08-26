import os
import sys
import argparse
import yaml
import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from model import ModelConfig, MDTransformer


def remap_keys(state_dict: dict) -> dict:
    """Map custom MD-Deployer keys to LlamaForCausalLM format."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key == "tok_emb.weight":
            new_state_dict["model.embed_tokens.weight"] = value
        elif key.startswith("blocks.") and "norm1.weight" in key:
            layer_idx = key.split(".")[1]
            new_key = f"model.layers.{layer_idx}.input_layernorm.weight"
            new_state_dict[new_key] = value
        elif key.startswith("blocks.") and "attn.qkv.weight" in key:
            layer_idx = key.split(".")[1]
            d_model = value.shape[1]
            q, k, v = value.chunk(3, dim=0)
            new_state_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = q
            new_state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = k
            new_state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = v
        elif key.startswith("blocks.") and "attn.out.weight" in key:
            layer_idx = key.split(".")[1]
            new_key = f"model.layers.{layer_idx}.self_attn.o_proj.weight"
            new_state_dict[new_key] = value
        elif key.startswith("blocks.") and "norm2.weight" in key:
            layer_idx = key.split(".")[1]
            new_key = f"model.layers.{layer_idx}.post_attention_layernorm.weight"
            new_state_dict[new_key] = value
        elif key.startswith("blocks.") and "ff.w1.weight" in key:
            layer_idx = key.split(".")[1]
            new_key = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
            new_state_dict[new_key] = value
        elif key.startswith("blocks.") and "ff.w2.weight" in key:
            layer_idx = key.split(".")[1]
            new_key = f"model.layers.{layer_idx}.mlp.down_proj.weight"
            new_state_dict[new_key] = value
        elif key.startswith("blocks.") and "ff.w3.weight" in key:
            layer_idx = key.split(".")[1]
            new_key = f"model.layers.{layer_idx}.mlp.up_proj.weight"
            new_state_dict[new_key] = value
        elif key == "norm.weight":
            new_state_dict["model.norm.weight"] = value
        elif key == "head.weight":
            new_state_dict["lm_head.weight"] = value
    return new_state_dict


def save_hf_format(config: ModelConfig, remapped_state_dict: dict, output_dir: str):
    """Create LlamaForCausalLM, load remapped weights, and save."""
    os.makedirs(output_dir, exist_ok=True)
    llama_config = LlamaConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.d_model,
        intermediate_size=config.d_ff,
        num_hidden_layers=config.n_layers,
        num_attention_heads=config.n_heads,
        max_position_embeddings=config.block_size,
        rms_norm_eps=1e-6,
    )
    model = LlamaForCausalLM(llama_config)
    model.load_state_dict(remapped_state_dict, strict=True)
    model.save_pretrained(output_dir)
    print(f"Saved HuggingFace model to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert MD-Deployer checkpoint to HuggingFace format"
    )
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument(
        "--output_dir", type=str, default="./hf_model", help="Output directory"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to load on")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = ModelConfig(**checkpoint["config"])
    model = MDTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])

    remapped = remap_keys(model.state_dict())
    save_hf_format(config, remapped, args.output_dir)


if __name__ == "__main__":
    main()
