import torch
from executor import CodeExecutor, load_problems
from loader import  model_tokenizer, gen_code

def evaluation(model, tokenizer, executor, problems):
    """
    Evaluate the models on the absis of a set of problems given
    """

    S=0
    R=0
    for  problem in problems:
        generated_code=gen_code(model, tokenizer, prompt=problem['prompt'], max_length=150)
        print(f"Extracted code \n {generated_code}")
        success, error,reward= executor.execute(code=generated_code,test_code=problem['test'])
        R+=reward
        if success:
            S+=1
    avg_reward=R/len(problems)
    return S, avg_reward

def simple_rlhf_training(num_epochs=3):
    """
    Simple training module that trains the model on the problems multiple times to get the best reward
    """
    model,tokenizer=model_tokenizer()
    executor=CodeExecutor()
    problems=load_problems()
    

    initial_success, initial_reward=evaluation(model,tokenizer,executor,problems)
    print(f"Initial Success: {initial_success}, Initial Reward: {initial_reward}")
#----------------------------------------------------------------------------------------
    LIST=[]
    print(f"✅ Baseline:")
    print(f"   Successes: {initial_success}/{len(problems)} ({initial_success/len(problems)*100:.1f}%)")
    print(f"   Avg Reward: {initial_reward:.3f}")
#-----------SUCCESSES AND REWARDS BEFORE TRAINING------------------ 
    LIST.append({
        'epoch':0,
        'successes':initial_success,
        'rewards':initial_reward
    })
#--------------------------------------------------------------

    for epoch in range(num_epochs):
        epoch_rewards=[]
        epoch_successes=0

        for i ,problem in enumerate(problems):
            best_R=-float('inf')
            best_code=None
            flag=False
            for i in range(3):
                generated_code=gen_code(model,tokenizer,problem['prompt'], max_length=150)
                print(f"DEBUG {generated_code[:200]}")
                success,error,reward=executor.execute(generated_code,problem['test'])

                print(f"  Attempt {i+1}: Reward = {reward:+.1f}")
                if reward>best_R:
                    best_R=reward
                    best_code=generated_code
                if success and not flag:
                    epoch_successes+=1
                    flag=True
                    break
            epoch_rewards.append(best_R)
    avg_rewards=sum(epoch_rewards)/len(epoch_rewards)
    print(f"\n{'='*70}")
    print(f"EPOCH {epoch + 1} RESULTS")
    print(f"{'='*70}")
    print(f"Successes: {epoch_successes}/{len(problems)} ({epoch_successes/len(problems)*100:.1f}%)")
    print(f"Avg Reward: {avg_rewards:.3f}")

    LIST.append({
        'epoch':epoch +1,
        'successes':epoch_successes,
        'rewards':avg_rewards
    })

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - PROGRESS SUMMARY")
    print("=" * 70)
    
    for h in LIST:
        print(f"Epoch {h['epoch']}: {h['successes']}/10 solved | Avg Reward: {h['rewards']:.3f}")
    
    improvement = LIST[-1]['successes'] - LIST[0]['successes']
    print(f"\n📈 Improvement: +{improvement} problems solved")
    
    return model, tokenizer, LIST
if __name__=='__main__':
    model,tokenizer,LIST=simple_rlhf_training(num_epochs=3)
    print("\n✅ Training demonstration complete!")
    print("\nNote: This is a SIMPLIFIED version.")
    print("Real RLHF with PPO would actually update model weights.")
#----------------------SAVING THE MODEL------------------------------
    save_dir = "./models/rlhf_trained"
    model.save_pretrained(save_dir,safe_serialization=True)
    tokenizer.save_pretrained(save_dir)












