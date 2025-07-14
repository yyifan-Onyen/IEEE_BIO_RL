import os
import sys
import random
import logging
import argparse
import torch
import datasets
import numpy as np
import transformers
import yaml
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from trl.trainer import GRPOConfig, GRPOTrainer
from transformers.trainer_utils import get_last_checkpoint
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
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


class MIMICMetrics:
    """Class to compute various medical text evaluation metrics"""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing_function = SmoothingFunction().method4
    
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
        try:
            return meteor_score([reference.lower().split()], hypothesis.lower().split())
        except:
            return 0.0
    
    def compute_bert_score(self, reference, hypothesis):
        """Compute BERTScore"""
        try:
            P, R, F1 = bert_score([hypothesis], [reference], lang="en", verbose=False)
            return F1.item()
        except:
            return 0.0
    
    def compute_all_metrics(self, reference, hypothesis):
        """Compute all metrics for a single reference-hypothesis pair"""
        bleu1, bleu4 = self.compute_bleu_scores(reference, hypothesis)
        rouge_scores = self.compute_rouge_scores(reference, hypothesis)
        meteor = self.compute_meteor_score(reference, hypothesis)
        bert_f1 = self.compute_bert_score(reference, hypothesis)
        
        return {
            'bleu1': bleu1,
            'bleu4': bleu4,
            'rouge1': rouge_scores['rouge1'],
            'rouge2': rouge_scores['rouge2'],
            'rougeL': rouge_scores['rougeL'],
            'meteor': meteor,
            'bert_f1': bert_f1
        }


class GRPOPipeline(object):
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Initialize metrics calculator
        self.metrics_calculator = MIMICMetrics()
        
        self.prepare()
        self.train()
        self.save()
        self.evaluate()

    def _process(self, example):
        """Process MIMIC dataset examples"""
        # Create medical impression generation prompt
        system_message = "You are an expert radiologist. Given the following chest X-ray findings, generate a concise and accurate impression that summarizes the key clinical findings and their significance."
        user_message = f"Findings: {example['finding']}\n\nPlease provide a clinical impression:"
        
        prompt = [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': user_message}
        ]
        
        return {
            'finding': example['finding'],
            'impression': example['impression'],
            'prompt': prompt
        }

    def _reward(self, prompts, completions, **kwargs):
        """
        Compute weighted reward based on multiple evaluation metrics
        """
        rewards = []
        records = ''
        
        # Extract findings and generated impressions
        findings = []
        for prompt in prompts:
            # Extract finding from user message
            user_content = prompt[1]['content']  # Get user message
            finding = user_content.split("Findings: ")[1].split("\n\nPlease provide")[0]
            findings.append(finding)
        
        generated_impressions = [completion[0]['content'] for completion in completions]
        
        # For now, we'll use a simple heuristic reward based on length and basic quality
        # In a real scenario, you might want to use a learned reward model
        # or implement a way to get reference impressions from the batch
        reference_impressions = kwargs.get('reference_impressions', None)
        
        # If no reference impressions available, use heuristic rewards
        if reference_impressions is None:
            # Use simple heuristic rewards based on generation quality
            for generated in generated_impressions:
                # Simple heuristic: reward based on length and basic medical terms
                reward = self._compute_heuristic_reward(generated)
                rewards.append(reward)
            return rewards
        
        # Get metric weights from config
        weights = self.config.get('reward_weights', {
            'bleu1': 0.15,
            'bleu4': 0.15,
            'rouge1': 0.15,
            'rouge2': 0.15,
            'rougeL': 0.15,
            'meteor': 0.15,
            'bert_f1': 0.10
        })
        
        for index, (finding, generated, reference) in enumerate(zip(findings, generated_impressions, reference_impressions)):
            # Compute all metrics
            metrics = self.metrics_calculator.compute_all_metrics(reference, generated)
            
            # Calculate weighted reward
            reward = 0.0
            for metric_name, metric_value in metrics.items():
                if metric_name in weights:
                    reward += weights[metric_name] * metric_value
            
            # Ensure reward is in [0, 1] range
            reward = max(0.0, min(1.0, reward))
            rewards.append(reward)
            
            # Log first example for debugging
            if index == 0:
                records += f'{" Finding ".center(80, "-")}\n{finding}\n\n'
                records += f'{" Reference Impression ".center(80, "-")}\n{reference}\n\n'
                records += f'{" Generated Impression ".center(80, "-")}\n{generated}\n\n'
                records += f'{" Metrics ".center(80, "-")}\n'
                for metric_name, metric_value in metrics.items():
                    records += f'{metric_name}: {metric_value:.4f}\n'
                records += f'{" Final Reward ".center(80, "-")}\n{reward:.4f}\n\n'
                
                # Log to console
                self.logger.info(f"Sample metrics: {metrics}")
                self.logger.info(f"Sample reward: {reward:.4f}")
        
        return rewards
    
    def _compute_heuristic_reward(self, generated_text):
        """
        Compute a simple heuristic reward for generated medical impressions
        """
        # Basic quality indicators for medical impressions
        medical_terms = [
            'normal', 'abnormal', 'findings', 'impression', 'chest', 'lung', 'heart',
            'pneumonia', 'effusion', 'consolidation', 'atelectasis', 'edema',
            'cardiomegaly', 'pneumothorax', 'pleural', 'opacity', 'clear'
        ]
        
        # Convert to lowercase for matching
        text_lower = generated_text.lower()
        
        # Length reward (prefer reasonable length)
        length_score = min(1.0, max(0.1, len(generated_text.split()) / 20.0))
        
        # Medical terminology reward
        medical_score = sum(1 for term in medical_terms if term in text_lower) / len(medical_terms)
        
        # Sentence structure reward (basic check for complete sentences)
        structure_score = 1.0 if generated_text.strip().endswith('.') else 0.5
        
        # Combine scores
        total_reward = (length_score * 0.4 + medical_score * 0.4 + structure_score * 0.2)
        
        return min(1.0, max(0.0, total_reward))

    def prepare(self):
        # setup logger
        logging.basicConfig(
            format='[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        self.logger = logging.getLogger(name='grpo_mimic')
        self.logger.setLevel(logging.INFO)
        datasets.utils.logging.set_verbosity_info()
        transformers.logging.set_verbosity_info()
        transformers.logging.enable_default_handler()
        transformers.logging.enable_explicit_format()

        # setup seed
        random.seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])
        torch.cuda.manual_seed_all(self.config['seed'])

        # start preparation
        self.logger.info('=============== Start Preparation ===============')
        for key, value in self.config.items():
            self.logger.info(f'{key}: {value}')

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_path'], trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # load MIMIC dataset
        self.logger.info(f"Loading MIMIC dataset from: {self.config['dataset_path']}")
        
        # Load train and test data
        train_df = pd.read_csv(self.config['train_file'])
        test_df = pd.read_csv(self.config['test_file'])
        
        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        # Create dataset dict
        self.dataset = datasets.DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })
        
        # Process the dataset
        self.dataset = self.dataset.map(self._process)
        self.dataset = self.dataset.remove_columns(['subject_id', 'study_id'])
        
        # Limit samples if specified
        if self.config['max_samples'] != -1:
            sample_indices = np.random.choice(
                len(self.dataset['train']),
                min(self.config['max_samples'], len(self.dataset['train'])),
                replace=False
            )
            self.dataset['train'] = self.dataset['train'].select(sample_indices)

        self.logger.info(f"Training samples: {len(self.dataset['train'])}")
        self.logger.info(f"Test samples: {len(self.dataset['test'])}")

        # setup configuration
        num_samples = self.config['num_devices'] * self.config['num_samples_per_device']
        assert num_samples % self.config['num_samples_per_prompt'] == 0
        micro_train_batch_size = num_samples // self.config['num_samples_per_prompt']
        assert self.config['train_batch_size'] % micro_train_batch_size == 0
        grad_accumulation_steps = self.config['train_batch_size'] // micro_train_batch_size
        assert self.config['eval_batch_size'] % num_samples == 0
        eval_accumulation_steps = self.config['eval_batch_size'] // num_samples

        logging_strategy = 'no' if self.config['log_interval'] == -1 else 'steps'
        eval_strategy = 'no' if self.config['eval_interval'] == -1 else 'steps'
        save_strategy = 'no' if self.config['save_interval'] == -1 else 'steps'

        self.trainer_config = GRPOConfig(
            model_init_kwargs=dict(
                trust_remote_code=True,
                attn_implementation='sdpa',
                torch_dtype=torch.bfloat16
            ),
            run_name=self.config['run_name'],
            output_dir=self.config['save_path'],
            logging_dir=self.config['save_path'],
            per_device_train_batch_size=self.config['num_samples_per_device'],
            num_generations=self.config['num_samples_per_prompt'],
            gradient_accumulation_steps=grad_accumulation_steps,
            per_device_eval_batch_size=self.config['num_samples_per_device'],
            eval_accumulation_steps=eval_accumulation_steps,
            learning_rate=self.config['learning_rate'],
            beta=self.config['kl_weight'],
            max_prompt_length=self.config['max_prompt_length'],
            max_completion_length=self.config['max_completion_length'],
            num_train_epochs=self.config['num_epochs'],
            seed=self.config['seed'],
            logging_strategy=logging_strategy,
            logging_steps=self.config['log_interval'],
            eval_strategy=eval_strategy,
            eval_steps=self.config['eval_interval'],
            save_strategy=save_strategy,
            save_steps=self.config['save_interval']
        )

        # setup trainer
        self.trainer = GRPOTrainer(
            model=self.config['model_path'],
            reward_funcs=[self._reward],
            args=self.trainer_config,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['test'],
            processing_class=self.tokenizer,
            peft_config=None
        )

    def train(self):
        # load checkpoint
        checkpoint = None
        if self.config['resume_last'] and os.path.isdir(self.config['save_path']):
            checkpoint = get_last_checkpoint(self.config['save_path'])

        # start training
        self.logger.info('=============== Start Training ===============')
        results = self.trainer.train(resume_from_checkpoint=checkpoint)
        metrics = results.metrics
        metrics['train_samples'] = len(self.dataset['train'])
        self.trainer.log_metrics('train', metrics)
        self.trainer.save_metrics('train', metrics)

    def evaluate(self):
        # start evaluation
        self.logger.info('=============== Start Evaluation ===============')
        metrics = self.trainer.evaluate()
        metrics['eval_samples'] = len(self.dataset['test'])
        self.trainer.log_metrics('eval', metrics)
        self.trainer.save_metrics('eval', metrics)

    def save(self):
        # start saving
        self.logger.info('=============== Start Saving ===============')
        self.trainer.save_model(self.config['save_path'])
        if self.trainer.accelerator.is_main_process:
            self.trainer.create_model_card(
                model_name=self.config['run_name'],
                dataset_name=self.config['dataset_path'],
                tags=['Medical-AI', 'GRPO', 'MIMIC-CXR']
            )
            self.trainer.model.config.use_cache = True
            self.trainer.model.config.save_pretrained(self.config['save_path'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML config file'
    )
    args = parser.parse_args()
    GRPOPipeline(args.config)


if __name__ == '__main__':
    main()
