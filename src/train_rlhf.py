import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, BitsAndBytesConfig
from executor import CodeExecutor, load_problems
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import dagshub
import mlflow
dagshub.init(
    repo_owner="ankitmohanbiswas",
    repo_name="RLHF-CODE-GEN",
    mlflow=True
)
mlflow.set_experiment("rlhf-code-gen-ppo")
MODEL_NAME = "Salesforce/codegen-350M-mono"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_skip_modules=["lm_head", "wte", "wpe"],
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["qkv_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

config = PPOConfig(
    learning_rate=5e-6,
    batch_size=1,
    mini_batch_size=1,
    gradient_accumulation_steps=1,
    ppo_epochs=2,          # was 4 — reduce to stabilize
    cliprange=0.1,         # was 0.2 — tighter clipping
    init_kl_coef=0.8,      # higher penalty = less drift
    target_kl=0.05,
    adap_kl_ctrl=True,
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
)
model.pretrained_model = prepare_model_for_kbit_training(
    model.pretrained_model,
    use_gradient_checkpointing=True,
)
model.pretrained_model = get_peft_model(model.pretrained_model, lora_config)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

ppo_trainer = PPOTrainer(
    config=config,
    model=model,
    ref_model=None,
    tokenizer=tokenizer,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
problems = load_problems(os.path.join(BASE_DIR, "data", "problems.json"))
executor = CodeExecutor()

NUM_EPOCHS = 1
history = []

def extract_body(generated: str) -> str:
    lines = []
    for line in generated.splitlines():
        if line.startswith("def ") and lines:
            break
        lines.append(line)
    return "\n".join(lines).rstrip()

with mlflow.start_run(run_name=f"codegen350M-lr{config.learning_rate}-kl{config.init_kl_coef}"):
    mlflow.log_params({
    "model_name": MODEL_NAME,
    "learning_rate": config.learning_rate,
    "lora_r": lora_config.r,
    "lora_alpha": lora_config.lora_alpha,
    "init_kl_coef": config.init_kl_coef,
    "ppo_epochs": config.ppo_epochs,
    "num_epochs": NUM_EPOCHS,
    "total_problems": len(problems),
        })
    for epoch in range(NUM_EPOCHS):
        epoch_rewards = []
        epoch_successes = 0

        for problem in problems:
            prompt = problem["prompt"] + "\n"

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=150,
            ).to("cuda")

            query_tensor = inputs["input_ids"][0]

            response_ids = ppo_trainer.generate(
                query_tensor,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.4,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            response_tensor = response_ids[0][len(query_tensor):]

            if response_tensor.shape[0] == 0 or torch.isnan(response_tensor.float()).any():
                print(f"  Skipping problem {problem['id']} — bad response tensor")
                torch.cuda.empty_cache()
                continue

            generated_body = extract_body(
                tokenizer.decode(response_tensor, skip_special_tokens=True)
            )
            full_code = prompt + generated_body

            success, error, reward_score = executor.execute(full_code, problem["test"])

            if success:
                epoch_successes += 1

            if success:
                shaped_reward = 1.0
            else:
                shaped_reward = -1.0

            if success:
                if "return" not in generated_body:
                    shaped_reward-=1.0


            shaped_reward = max(min(shaped_reward, 1.0), -1.0)
            epoch_rewards.append(shaped_reward)

            reward_tensor = torch.tensor(shaped_reward, dtype=torch.float32, device="cuda")

            stats = ppo_trainer.step(
                queries=[query_tensor],
                responses=[response_tensor],
                scores=[reward_tensor],
            )

            kl = stats.get("objective/kl", "N/A")
            print(
                f"Epoch {epoch+1} | Problem {problem['id']:>2} "
                f"| Reward: {shaped_reward:+.2f} | Success: {success} | KL: {kl}"
            )
            print(f"  Generated: {generated_body[:120].strip()!r}")
            mlflow.log_metrics({
                    "problem_reward": shaped_reward,
                    "problem_success": int(success),
                    "problem_kl": float(kl) if isinstance(kl, float) else 0.0,
                            }, step=(epoch * len(problems)) + problem["id"])

            del inputs, response_ids, response_tensor, reward_tensor
            torch.cuda.empty_cache()

        avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0
        print(
            f"\n=== EPOCH {epoch+1} SUMMARY | "
            f"Successes: {epoch_successes}/{len(problems)} | "
            f"Avg Reward: {avg_reward:.3f} ===\n"
            )
        history.append({
                "epoch": epoch + 1,
                "successes": epoch_successes,
                "avg_reward": avg_reward,
            })
        mlflow.log_metrics({
                        "epoch_successes": epoch_successes,
                        "epoch_success_rate": epoch_successes / len(problems),
                        "epoch_avg_reward": avg_reward,
                                }, step=epoch + 1)

        ppo_trainer.save_pretrained(f"models/rlhf_epoch_{epoch+1}")
    mlflow.log_metric("final_best_successes", max(h["successes"] for h in history))
    mlflow.log_metric("final_best_avg_reward", max(h["avg_reward"] for h in history))

save_dir = "./models/rlhf_ppo_trained"
ppo_trainer.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"\nModel saved to {save_dir}")

print("\n=== TRAINING SUMMARY ===")
for h in history:
    print(
        f"Epoch {h['epoch']}: "
        f"{h['successes']}/{len(problems)} solved | "
        f"Avg Reward: {h['avg_reward']:.3f}"
    )