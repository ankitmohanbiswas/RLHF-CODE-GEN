from loader import model_tokenizer, gen_code
from executor import CodeExecutor, load_problems
from transformers import AutoModelForCausalLM, AutoTokenizer


def evaluate_model(model_path, problems, executor):
    """
    Evaluate a model on all problems.
    
    Args:
        model_path: Path to model or HuggingFace model name
        problems: List of coding problems
        executor: CodeExecutor instance
    
    Returns:
        dict: Evaluation results
    """
    print(f"\n📊 Evaluating: {model_path}")
    print("=" * 70)
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto"
    )
    
    results = {
        'total': len(problems),
        'passed': 0,
        'failed': 0,
        'rewards': [],
        'details': []
    }
    
    for i, problem in enumerate(problems):
        # Generate code
        code = gen_code(model, tokenizer, problem['prompt'], max_length=150)
        
        # Test code
        success, error, reward = executor.execute(code, problem['test'])
        
        results['rewards'].append(reward)
        
        if success:
            results['passed'] += 1
        else:
            results['failed'] += 1
        
        results['details'].append({
            'id': i + 1,
            'problem': problem['description'],
            'success': success,
            'reward': reward,
            'error': error[:100] if error else None
        })
        
        status = "✅" if success else "❌"
        print(f"   {status} | {reward:+.1f}")
    
    # Calculate metrics
    results['success_rate'] = (results['passed'] / results['total']) * 100
    results['avg_reward'] = sum(results['rewards']) / len(results['rewards'])
    
    print("\n" + "=" * 70)
    print(f"✅ Passed: {results['passed']}/10 ({results['success_rate']:.1f}%)")
    print(f"❌ Failed: {results['failed']}/10")
    print(f"📊 Avg Reward: {results['avg_reward']:+.3f}")
    print("=" * 70)
    
    return results


def compare_models():
    """
    Compare base model vs RLHF-trained model.
    """
    print("\n" + "=" * 70)
    print("MODEL COMPARISON: Before vs After RLHF")
    print("=" * 70)
    
    # Load components
    problems = load_problems()
    executor = CodeExecutor(timeout=5)
    
    # Evaluate base model
    print("\n🔵 BEFORE RLHF (Base Model)")
    base_results = evaluate_model(
        "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        problems,
        executor
    )
    
    # Evaluate trained model
    print("\n🟢 AFTER RLHF (Trained Model)")
    trained_results = evaluate_model(
        "models/rlhf_trained",  # Your saved model
        problems,
        executor
    )
    
    # Comparison
    print("\n" + "=" * 70)
    print("📈 IMPROVEMENT ANALYSIS")
    print("=" * 70)
    
    improvement_problems = trained_results['passed'] - base_results['passed']
    improvement_reward = trained_results['avg_reward'] - base_results['avg_reward']
    improvement_rate = trained_results['success_rate'] - base_results['success_rate']
    
    print(f"\n🎯 Success Rate:")
    print(f"   Before: {base_results['success_rate']:.1f}%")
    print(f"   After:  {trained_results['success_rate']:.1f}%")
    print(f"   Change: +{improvement_rate:.1f}%")
    
    print(f"\n🏆 Problems Solved:")
    print(f"   Before: {base_results['passed']}/10")
    print(f"   After:  {trained_results['passed']}/10")
    print(f"   Gained: +{improvement_problems} problems")
    
    print(f"\n💰 Average Reward:")
    print(f"   Before: {base_results['avg_reward']:+.3f}")
    print(f"   After:  {trained_results['avg_reward']:+.3f}")
    print(f"   Change: {improvement_reward:+.3f}")
    
    # Show which problems improved
    print(f"\n✨ Newly Solved Problems:")
    for i, (base, trained) in enumerate(zip(base_results['details'], trained_results['details'])):
        if not base['success'] and trained['success']:
            print(f"   • {trained['problem']}")
    
    print("\n" + "=" * 70)
    
    return base_results, trained_results


if __name__ == "__main__":
    base, trained = compare_models()