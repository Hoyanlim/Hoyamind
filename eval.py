import argparse
import glob
import os

# import random
import warnings

# import numpy as np
import torch
from transformers import AutoTokenizer, TextStreamer
from model.model import HoyamindConfig, HoyamindForCausalLM

# from model.model_lora import apply_lora, load_lora  # ！修正：原缺少LoRA加载支持
from trainer.trainer_utils import setup_seed

warnings.filterwarnings("ignore")


def init_model(args):
    project_root = os.path.dirname(os.path.abspath(__file__))
    load_from = (
        args.load_from
        if os.path.isabs(args.load_from)
        else os.path.join(project_root, args.load_from)
    )
    tokenizer = AutoTokenizer.from_pretrained(load_from)
    if "model" in args.load_from:
        moe_suffix = "_moe" if hasattr(args, "use_moe") and args.use_moe else ""
        save_dir = (
            args.save_dir
            if os.path.isabs(args.save_dir)
            else os.path.join(project_root, args.save_dir)
        )
        ckp = os.path.join(
            save_dir, f"{args.weight}_{args.hidden_size}{moe_suffix}.pth"
        )
        if not os.path.exists(ckp):
            candidates = sorted(
                os.path.basename(p)
                for p in glob.glob(os.path.join(save_dir, f"{args.weight}_*.pth"))
            )
            raise FileNotFoundError(
                f"未找到权重文件: {ckp}\n"
                f"可用候选: {candidates if candidates else '无'}\n"
                f"请检查 --save_dir / --weight / --hidden_size / --use_moe 参数。"
            )

        state_dict = torch.load(ckp, map_location=args.device)

        # 兼容旧版权重命名：self_attn -> self_attention；忽略已废弃的 q_norm/k_norm 键
        if any(".self_attn." in k for k in state_dict.keys()):
            remapped = {}
            for k, v in state_dict.items():
                nk = k.replace(".self_attn.", ".self_attention.")
                if ".self_attention.q_norm." in nk or ".self_attention.k_norm." in nk:
                    continue
                remapped[nk] = v
            state_dict = remapped

        inferred_hidden = state_dict.get(
            "model.embed_tokens.weight", torch.empty(0)
        ).shape
        inferred_hidden = (
            inferred_hidden[1] if len(inferred_hidden) == 2 else args.hidden_size
        )

        layer_ids = {
            int(k.split(".")[2])
            for k in state_dict.keys()
            if k.startswith("model.layers.") and k.split(".")[2].isdigit()
        }
        inferred_num_layers = (
            (max(layer_ids) + 1) if layer_ids else args.num_hidden_layers
        )

        gate_key = next(
            (k for k in state_dict.keys() if k.endswith("mlp.gate_proj.weight")), None
        )
        inferred_intermediate = (
            int(state_dict[gate_key].shape[0]) if gate_key is not None else None
        )

        q_key = next(
            (
                k
                for k in state_dict.keys()
                if k.endswith("self_attention.q_proj.weight")
            ),
            None,
        )
        k_key = next(
            (
                k
                for k in state_dict.keys()
                if k.endswith("self_attention.k_proj.weight")
            ),
            None,
        )
        num_attention_heads = 8
        inferred_num_kv_heads = 2
        if q_key is not None and k_key is not None:
            q_out = int(state_dict[q_key].shape[0])
            k_out = int(state_dict[k_key].shape[0])
            if q_out > 0:
                inferred_num_kv_heads = max(
                    1, int(round((k_out / q_out) * num_attention_heads))
                )

        model = HoyamindForCausalLM(
            HoyamindConfig(
                hidden_size=inferred_hidden,
                num_hidden_layers=inferred_num_layers,
                intermediate_size=inferred_intermediate,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=inferred_num_kv_heads,
                use_moe=bool(args.use_moe),
                inference_rope_scaling=args.inference_rope_scaling,
            )
        )

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys or unexpected_keys:
            print(
                f"权重兼容加载: missing={len(missing_keys)}, unexpected={len(unexpected_keys)}"
            )

        # ！修正：原缺少LoRA加载逻辑
    #     if args.lora_weight != "None":
    #         # apply_lora(model)
    #         # load_lora(
    #         #     model,
    #         #     f"./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth",
    #         # )
    # else:
    #     model = AutoModelForCausalLM.from_pretrained(
    #         args.load_from, trust_remote_code=True
    #     )
    print(
        f"Hoyamind模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)"  # ！修正：原残留Hoyamind命名
    )
    return model.eval().to(args.device), tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Hoyamind模型推理与对话"
    )  # ！修正：原残留Hoyamind命名
    parser.add_argument(
        "--load_from",
        default="model",
        type=str,
        help="模型加载路径（model=原生torch权重，其他路径=transformers格式）",
    )
    parser.add_argument("--save_dir", default="out", type=str, help="模型权重目录")
    parser.add_argument(
        "--weight",
        default="full_sft",
        type=str,
        help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）",
    )
    parser.add_argument(
        "--lora_weight",
        default="None",
        type=str,
        help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）",
    )
    parser.add_argument(
        "--hidden_size",
        default=768,
        type=int,
        help="隐藏层维度（512=Small-26M, 640=MoE-145M, 768=Base-104M）",
    )
    parser.add_argument(
        "--num_hidden_layers",
        default=8,
        type=int,
        help="隐藏层数量（Small/MoE=8, Base=16）",
    )
    parser.add_argument(
        "--use_moe",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用MoE架构（0=否，1=是）",
    )
    parser.add_argument(
        "--inference_rope_scaling",
        default=False,
        action="store_true",
        help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）",
    )
    parser.add_argument(
        "--max_new_tokens",
        default=8192,
        type=int,
        help="最大生成长度（注意：并非模型实际长文本能力）",
    )
    parser.add_argument(
        "--temperature",
        default=0.85,
        type=float,
        help="生成温度，控制随机性（0-1，越大越随机）",
    )
    parser.add_argument(
        "--top_p", default=0.85, type=float, help="nucleus采样阈值（0-1）"
    )
    parser.add_argument(
        "--historys",
        default=0,
        type=int,
        help="携带历史对话轮数（需为偶数，0表示不携带历史）",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        type=str,
        help="运行设备",
    )
    args = parser.parse_args()

    prompts = [
        "你有什么特长？",
        "为什么天空是蓝色的",
        "请用Python写一个计算斐波那契数列的函数",
        '解释一下"光合作用"的基本过程',
        "如果明天下雨，我应该如何出门",
        "比较一下猫和狗作为宠物的优缺点",
        "解释什么是机器学习",
        "推荐一些中国的美食",
    ]

    conversation = []
    model, tokenizer = init_model(args)
    input_mode = int(input("[0] 自动测试\n[1] 手动输入\n"))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    prompt_iter = prompts if input_mode == 0 else iter(lambda: input("👶: "), "")
    for prompt in prompt_iter:
        setup_seed(2026)  # or setup_seed(random.randint(0, 2048))
        if input_mode == 0:
            print(f"👶: {prompt}")
        conversation = conversation[-args.historys :] if args.historys else []
        conversation.append({"role": "user", "content": prompt})

        if args.weight != "pretrain":
            templates = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if args.weight == "reason":
                templates["enable_thinking"] = True  # 仅Reason模型使用
            try:
                inputs = tokenizer.apply_chat_template(
                    conversation=conversation, **templates
                )
            except TypeError:
                inputs = tokenizer.apply_chat_template(
                    messages=conversation, **templates
                )
        else:
            inputs = tokenizer.bos_token + prompt
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print("🤖️: ", end="")
        generated_ids = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=1.0,
        )
        response = tokenizer.decode(
            generated_ids[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
        )

        if not response.strip():
            fallback_text = tokenizer.bos_token + prompt
            fallback_inputs = tokenizer(
                fallback_text, return_tensors="pt", truncation=True
            ).to(args.device)
            generated_ids = model.generate(
                inputs=fallback_inputs["input_ids"],
                attention_mask=fallback_inputs["attention_mask"],
                max_new_tokens=min(args.max_new_tokens, 256),
                min_new_tokens=16,
                do_sample=True,
                streamer=streamer,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                top_p=max(args.top_p, 0.9),
                temperature=max(args.temperature, 0.9),
                repetition_penalty=1.05,
            )
            response = tokenizer.decode(
                generated_ids[0][len(fallback_inputs["input_ids"][0]) :],
                skip_special_tokens=True,
            )

        conversation.append({"role": "assistant", "content": response})
        print("\n\n")


if __name__ == "__main__":
    main()
