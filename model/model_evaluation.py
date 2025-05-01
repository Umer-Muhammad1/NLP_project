import torch
import json
import os
from model_loader import load_model_and_tokenizer
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import evaluate  # HuggingFace's evaluate library

def evaluate_model(model, tokenizer, test_examples, device="cuda", max_new_tokens=100, 
                   temperature=0.7, batch_size=1, progress_bar=True):
    """
    Evaluates a model on a list of test examples with multiple metrics.

    Args:
        model: The model to evaluate
        tokenizer: The tokenizer to use
        test_examples: List of dictionaries with 'instruction', 'input', and 'reference_output' keys
        device: Device to run inference on
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (set to 0 for deterministic generation)
        batch_size: Batch size for evaluation (use >1 for faster evaluation if memory allows)
        progress_bar: Whether to show progress bar

    Returns:
        dict: Dictionary with generated responses and metrics
    """
    model.eval()
    results = []
    
    # Initialize metric calculators
    rouge = Rouge()
    bertscore = evaluate.load("bertscore")
    
    iterator = tqdm(range(0, len(test_examples), batch_size)) if progress_bar else range(0, len(test_examples), batch_size)
    
    for i in iterator:
        batch = test_examples[i:i+batch_size]
        batch_results = []
        
        for example in batch:
            instruction = example['instruction']
            input_text = example.get('input', '')
            reference = example.get('reference_output', '')

            # Format the prompt based on whether input is provided
            if input_text:
                prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
            else:
                prompt = f"Instruction: {instruction}\nOutput:"

            # Tokenize and move to device
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Decode and extract only the response part
            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_output[len(prompt):].strip()
            
            # Calculate metrics if reference is available
            metrics = {}
            if reference:
                # BLEU score (simplified)
                try:
                    bleu_score = sentence_bleu([reference.split()], response.split())
                    metrics['bleu'] = bleu_score
                except Exception as e:
                    metrics['bleu'] = 0
                    metrics['bleu_error'] = str(e)
                
                # ROUGE scores
                try:
                    rouge_scores = rouge.get_scores(response, reference)[0]
                    metrics['rouge'] = {
                        'rouge-1': rouge_scores['rouge-1']['f'],
                        'rouge-2': rouge_scores['rouge-2']['f'],
                        'rouge-l': rouge_scores['rouge-l']['f']
                    }
                except Exception as e:
                    metrics['rouge'] = {'error': str(e)}
                
                # BERTScore (semantic similarity)
                try:
                    results_bert = bertscore.compute(
                        predictions=[response], 
                        references=[reference], 
                        lang="en"
                    )
                    metrics['bertscore'] = results_bert['f1'][0]
                except Exception as e:
                    metrics['bertscore'] = {'error': str(e)}
            
            result = {
                'instruction': instruction,
                'input': input_text,
                'generated_output': response,
                'reference_output': reference,
                'metrics': metrics
            }
            batch_results.append(result)
        
        results.extend(batch_results)

    # Calculate aggregate metrics
    if len(results) > 0 and 'metrics' in results[0] and results[0]['metrics']:
        aggregated_metrics = {
            'avg_bleu': np.mean([r['metrics'].get('bleu', 0) for r in results if 'bleu' in r['metrics']]),
        }
        
        rouge_keys = ['rouge-1', 'rouge-2', 'rouge-l']
        for key in rouge_keys:
            aggregated_metrics[f'avg_{key}'] = np.mean([
                r['metrics'].get('rouge', {}).get(key, 0) 
                for r in results if 'rouge' in r['metrics'] and key in r['metrics']['rouge']
            ])
        
        aggregated_metrics['avg_bertscore'] = np.mean([
            r['metrics'].get('bertscore', 0) 
            for r in results if 'bertscore' in r['metrics'] and not isinstance(r['metrics']['bertscore'], dict)
        ])
        
        # Add aggregated metrics to results
        return {"results": results, "aggregated_metrics": aggregated_metrics}
    
    return {"results": results}

def run_model_evaluation(model_name="Base Model", test_examples=None, 
                         save_results=False, display_examples=5, 
                         load_model_fn=None, device="cuda"):
    """
    Run complete model evaluation and display/save results.
    
    Args:
        model_name: Name of the model for logging
        test_examples: List of test examples (if None, will attempt to load)
        save_results: Whether to save results to disk
        display_examples: Number of examples to display (0 for none)
        load_model_fn: Function to load model and tokenizer (if None, will use default)
        device: Device to run inference on
    
    Returns:
        dict: Evaluation results including metrics
    """
    print(f"\n===== {model_name} Evaluation =====")
    
    # Load model if function provided
    if load_model_fn:
        model, tokenizer = load_model_fn()
    else:
        model, tokenizer = load_model_and_tokenizer()
    
    model.to(device)
    
    # Load test examples if not provided
    if test_examples is None:
        # Logic to load test examples - customize this
        try:
            with open("./data/test_examples.json", "r") as f:
                test_examples = json.load(f)
        except Exception as e:
            print(f"Error loading test examples: {e}")
            test_examples = []
    
    # Run evaluation
    eval_results = evaluate_model(
        model, 
        tokenizer, 
        test_examples, 
        device=device
    )
    
    # Display results
    if display_examples > 0:
        for i, result in enumerate(eval_results["results"][:display_examples]):
            print(f"\nExample {i+1}:")
            print(f"Instruction: {result['instruction']}")
            if result['input']:
                print(f"Input: {result['input']}")
            print(f"\nGenerated output: {result['generated_output']}")
            if result['reference_output']:
                print(f"Reference output: {result['reference_output']}")
                if 'metrics' in result:
                    print("\nMetrics:")
                    for metric_name, value in result['metrics'].items():
                        if isinstance(value, dict):
                            print(f"  {metric_name}: {value}")
                        else:
                            print(f"  {metric_name}: {value:.4f}")
            print("-" * 50)
    
    # Display aggregated metrics
    if "aggregated_metrics" in eval_results:
        print("\nAggregated Metrics:")
        for metric_name, value in eval_results["aggregated_metrics"].items():
            print(f"  {metric_name}: {value:.4f}")
    
    # Save results
    if save_results:
        results_dir = "./evaluation_results"
        os.makedirs(results_dir, exist_ok=True)
        
        filename = f"{results_dir}/{model_name.replace(' ', '_').lower()}_results.json"
        with open(filename, "w") as f:
            json.dump(eval_results, f, indent=2)
        print(f"Results saved to {filename}")
    
    return eval_results