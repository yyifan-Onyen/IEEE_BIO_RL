#!/usr/bin/env python3
"""
Evaluation script for LLaMA model on MIMIC-CXR dataset.
Supports multiple metrics: BLEU, ROUGE, METEOR, CIDEr, BERTScore
"""

import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml
import os
from tqdm import tqdm
import json
from datetime import datetime

# Evaluation metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider
from bert_score import score as bert_score
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class MIMICEvaluator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model and tokenizer
        self.load_model()
        
        # Initialize metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.cider_scorer = Cider()
        self.smoothing_function = SmoothingFunction().method4
        
    def load_model(self):
        """Load LLaMA model and tokenizer"""
        print(f"Loading model: {self.config['model']['name']}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['name'],
            trust_remote_code=True
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['name'],
            torch_dtype=torch.float16 if self.config['model']['use_fp16'] else torch.float32,
            device_map="auto" if self.config['model']['device_map'] == "auto" else None,
            trust_remote_code=True
        )
        
        if self.config['model']['device_map'] != "auto":
            self.model = self.model.to(self.device)
            
        self.model.eval()
        print("Model loaded successfully!")
        
    def generate_impression(self, finding_text):
        """Generate impression from finding using LLaMA"""
        if self.config['generation'].get('use_chat_template', False):
            # Use tokenizer's chat template
            messages = [
                {"role": "system", "content": "You are an expert radiologist. Given the following chest X-ray findings, generate a concise and accurate impression that summarizes the key clinical findings and their significance."},
                {"role": "user", "content": "Findings: {finding}\n\nPlease provide a clinical impression:"}
            ]
            
            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Fallback to manual prompt template
            prompt = self.config['generation']['prompt_template'].format(finding=finding_text)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.config['generation']['max_input_length'],
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config['generation']['max_new_tokens'],
                temperature=self.config['generation']['temperature'],
                do_sample=self.config['generation']['do_sample'],
                top_p=self.config['generation']['top_p'],
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the generated part (excluding input)
        generated_text = self.tokenizer.decode(
            outputs[0][len(inputs['input_ids'][0]):], 
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def compute_bleu_scores(self, reference, hypothesis):
        """Compute BLEU-1 and BLEU-4 scores"""
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        bleu1 = sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), 
                             smoothing_function=self.smoothing_function)
        bleu4 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25),
                             smoothing_function=self.smoothing_function)
        
        return bleu1, bleu4
    
    def compute_rouge_scores(self, reference, hypothesis):
        """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores"""
        scores = self.rouge_scorer.score(reference, hypothesis)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def compute_meteor_score(self, reference, hypothesis):
        """Compute METEOR score"""
        return meteor_score([reference.lower().split()], hypothesis.lower().split())
    
    def compute_cider_score(self, references, hypotheses):
        """Compute CIDEr score for all samples"""
        # Format for CIDEr scorer
        gts = {}
        res = {}
        
        for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
            gts[i] = [ref]
            res[i] = [hyp]
        
        score, _ = self.cider_scorer.compute_score(gts, res)
        return score
    
    def compute_bert_score(self, references, hypotheses):
        """Compute BERTScore"""
        P, R, F1 = bert_score(hypotheses, references, lang="en", verbose=False)
        return {
            'precision': P.mean().item(),
            'recall': R.mean().item(),
            'f1': F1.mean().item()
        }
    
    def evaluate_dataset(self, test_file):
        """Evaluate model on the test dataset"""
        print(f"Loading test data from: {test_file}")
        df = pd.read_csv(test_file)
        
        print(f"Dataset size: {len(df)} samples")
        
        # Lists to store results
        references = []
        hypotheses = []
        individual_scores = []
        
        # Process each sample
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            finding = row['finding']
            reference_impression = row['impression']
            
            # Generate impression
            try:
                generated_impression = self.generate_impression(finding)
                
                # Store for batch metrics
                references.append(reference_impression)
                hypotheses.append(generated_impression)
                
                # Compute individual scores
                bleu1, bleu4 = self.compute_bleu_scores(reference_impression, generated_impression)
                rouge_scores = self.compute_rouge_scores(reference_impression, generated_impression)
                meteor = self.compute_meteor_score(reference_impression, generated_impression)
                
                individual_scores.append({
                    'sample_id': idx,
                    'subject_id': row['subject_id'],
                    'study_id': row['study_id'],
                    'finding': finding,
                    'reference_impression': reference_impression,
                    'generated_impression': generated_impression,
                    'bleu1': bleu1,
                    'bleu4': bleu4,
                    'rouge1': rouge_scores['rouge1'],
                    'rouge2': rouge_scores['rouge2'],
                    'rougeL': rouge_scores['rougeL'],
                    'meteor': meteor
                })
                
            except Exception as e:
                print(f"Error processing sample {idx}: {str(e)}")
                continue
        
        # Compute batch metrics
        print("Computing batch metrics...")
        cider_score = self.compute_cider_score(references, hypotheses)
        bert_scores = self.compute_bert_score(references, hypotheses)
        
        # Aggregate individual scores
        metrics_summary = {
            'total_samples': len(individual_scores),
            'bleu1_mean': sum(s['bleu1'] for s in individual_scores) / len(individual_scores),
            'bleu4_mean': sum(s['bleu4'] for s in individual_scores) / len(individual_scores),
            'rouge1_mean': sum(s['rouge1'] for s in individual_scores) / len(individual_scores),
            'rouge2_mean': sum(s['rouge2'] for s in individual_scores) / len(individual_scores),
            'rougeL_mean': sum(s['rougeL'] for s in individual_scores) / len(individual_scores),
            'meteor_mean': sum(s['meteor'] for s in individual_scores) / len(individual_scores),
            'cider': cider_score,
            'bert_score_precision': bert_scores['precision'],
            'bert_score_recall': bert_scores['recall'],
            'bert_score_f1': bert_scores['f1']
        }
        
        return metrics_summary, individual_scores
    
    def save_results(self, metrics_summary, individual_scores, output_dir, run_name=None):
        """Save evaluation results"""
        # Create run-specific directory if run_name is provided
        if run_name:
            output_dir = os.path.join(output_dir, run_name)
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary metrics
        summary_file = os.path.join(output_dir, f"evaluation_summary_{timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        
        # Save individual scores
        detailed_file = os.path.join(output_dir, f"evaluation_detailed_{timestamp}.json")
        with open(detailed_file, 'w') as f:
            json.dump(individual_scores, f, indent=2)
        
        # Save as CSV for easy analysis
        csv_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.csv")
        pd.DataFrame(individual_scores).to_csv(csv_file, index=False)
        
        print(f"Results saved to {output_dir}")
        print(f"Summary: {summary_file}")
        print(f"Detailed: {detailed_file}")
        print(f"CSV: {csv_file}")
        
        return summary_file, detailed_file, csv_file

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLaMA model on MIMIC-CXR dataset")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to configuration YAML file")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Configuration loaded:")
    print(yaml.dump(config, default_flow_style=False))
    
    # Initialize evaluator
    evaluator = MIMICEvaluator(config)
    
    # Run evaluation
    print("Starting evaluation...")
    metrics_summary, individual_scores = evaluator.evaluate_dataset(config['data']['test_file'])
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION RESULTS SUMMARY")
    print("="*50)
    print(f"Total samples processed: {metrics_summary['total_samples']}")
    print(f"BLEU-1: {metrics_summary['bleu1_mean']:.4f}")
    print(f"BLEU-4: {metrics_summary['bleu4_mean']:.4f}")
    print(f"ROUGE-1: {metrics_summary['rouge1_mean']:.4f}")
    print(f"ROUGE-2: {metrics_summary['rouge2_mean']:.4f}")
    print(f"ROUGE-L: {metrics_summary['rougeL_mean']:.4f}")
    print(f"METEOR: {metrics_summary['meteor_mean']:.4f}")
    print(f"CIDEr: {metrics_summary['cider']:.4f}")
    print(f"BERTScore Precision: {metrics_summary['bert_score_precision']:.4f}")
    print(f"BERTScore Recall: {metrics_summary['bert_score_recall']:.4f}")
    print(f"BERTScore F1: {metrics_summary['bert_score_f1']:.4f}")
    print("="*50)
    
    # Save results
    run_name = config.get('run_name', None)
    evaluator.save_results(metrics_summary, individual_scores, config['output']['save_dir'], run_name)
    
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    main()
