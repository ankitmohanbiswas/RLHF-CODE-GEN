"""
SIMPLE RLHF TRAINING 
MODULE TO TRAIN THE MODEL TO GENERATE NEW CODE AND GET REWARDED BASED ON THE REWARD
"""
import torch
from loader import model_tokenizer, gen_code
from executor import CodeExecutor, load_problems


def generate_completion_only(model, tokenizer, prompt: str, max_length: int = 100):
    """
    Generate ONLY the code completion (not the prompt).
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Starting code (e.g., "def add(a, b):")
        max_length: Max new tokens to generate
    
    Returns:
        str: Complete code (prompt + generated completion)
    """
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs['input_ids'].shape[1]
    
    # Generate new tokens
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=1  # Only one completion
        )
    
    # Decode ONLY the new tokens (skip the input)
    new_tokens = outputs[0][input_length:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Combine prompt + completion
    full_code = prompt + "\n" + completion
    
    return full_code


def simple_training(num_epochs: int = 3):
    """
    A simple training loop for the RLHF model.
    
    This evaluates baseline performance before actual training.
    Shows how many problems the model can solve without RLHF.
    
    Args:
        num_epochs: Number of training epochs (not used yet)
    
    Returns:
        tuple: (successes, total_reward)
    """
    # Header
    print("=" * 60)
    print("RLHF TRAINING - BASELINE EVALUATION")
    print("=" * 60)
    
    # Load components
    print("\n📦 Loading model...")
    model, tokenizer = model_tokenizer()
    
    print("📦 Loading executor...")
    executor = CodeExecutor(timeout=5)
    
    print("📦 Loading problems...")
    problems = load_problems()
    
    print(f"\n✅ Loaded {len(problems)} problems")
    print("\n🚀 Starting evaluation...\n")
    print("=" * 60)
    
    # Track results
    total_reward = 0
    successes = 0
    results = []
    
    # Evaluate each problem
    for i, problem in enumerate(problems):
        print(f"\n[Problem {i+1}/{len(problems)}] {problem['description']}")
        print(f"Prompt: {problem['prompt'][:50]}...")  # Show first 50 chars
        
        try:
            # Generate code completion
            generated_code = generate_completion_only(
                model, 
                tokenizer, 
                problem['prompt'], 
                max_length=150
            )
            
            # Show generated code
            print("\nGenerated Code:")
            print(generated_code[:200])  # Show first 200 chars
            if len(generated_code) > 200:
                print("...")
            print("-" * 40)
            
            # Execute and get reward
            success, error, reward = executor.execute(generated_code, problem['test'])
            
            # Update stats
            total_reward += reward
            if success:
                successes += 1
            
            # Store result
            results.append({
                'id': i + 1,
                'problem': problem['description'],
                'success': success,
                'reward': reward
            })
            
            # Print result
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"\n{status} | Reward: {reward:+.1f}")
            
            # Show error if failed
            if not success and error:
                error_preview = error[:150]
                print(f"💥 Error: {error_preview}...")
        
        except Exception as e:
            print(f"❌ Generation Error: {e}")
            total_reward += -1.0
            results.append({
                'id': i + 1,
                'problem': problem['description'],
                'success': False,
                'reward': -1.0
            })
        
        print("=" * 60)
    
    # Final Summary
    print("\n" + "=" * 60)
    print("📊 BASELINE EVALUATION RESULTS")
    print("=" * 60)
    print(f"Problems Solved: {successes}/{len(problems)} ({successes/len(problems)*100:.1f}%)")
    print(f"Average Reward: {total_reward/len(problems):.2f}")
    print("=" * 60)
    
    # Show successful problems
    if successes > 0:
        print("\n✅ Successfully Solved:")
        for r in results:
            if r['success']:
                print(f"   {r['id']}. {r['problem']}")
    
    # Show failed problems
    failed = len(problems) - successes
    if failed > 0:
        print(f"\n❌ Failed ({failed} problems):")
        for r in results:
            if not r['success']:
                print(f"   {r['id']}. {r['problem']} (reward: {r['reward']:+.1f})")
    
    print("\n" + "=" * 60)
    
    return successes, total_reward


if __name__ == "__main__":
    print("\n🎯 BASELINE PERFORMANCE TEST")
    print("This shows how well the model performs BEFORE RLHF training\n")
    
    successes, total_reward = simple_training(num_epochs=3)
    
    