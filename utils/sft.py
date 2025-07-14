#!/usr/bin/env python3
"""
SFT (Supervised Fine-Tuning) Training Script for MIMIC-CXR Dataset
Using TRL SFTTrainer for medical impression generation
"""

import os
import sys
import yaml
import random
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import datasets
import transformers
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer, SFTConfig


class SFTPipeline(object):
    def __init__(self, config_path):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.prepare()

    def _process(self, example):
        """
        Process MIMIC dataset samples for SFT training
        Convert findings -> impression format to conversational format
        """
        # Create the prompt using chat template
        messages = [
            {
                "role": "system", 
                "content": "You are a medical AI assistant. Generate a concise and accurate medical impression based on the given chest X-ray findings."
            },
            {
                "role": "user", 
                "content": f"Based on these chest X-ray findings, please provide a medical impression:\n\nFindings: {example['finding']}"
            },
            {
                "role": "assistant",
                "content": example['impression']
            }
        ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        return {"text": text}

    def prepare(self):
        # Setup logger
        logging.basicConfig(
            format='[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        self.logger = logging.getLogger(name='sft_mimic')
        self.logger.setLevel(logging.INFO)
        datasets.utils.logging.set_verbosity_info()
        transformers.logging.set_verbosity_info()
        transformers.logging.enable_default_handler()
        transformers.logging.enable_explicit_format()

        # Setup seed
        random.seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])
        torch.cuda.manual_seed_all(self.config['seed'])

        # Start preparation
        self.logger.info('=============== Start Preparation ===============')
        for key, value in self.config.items():
            self.logger.info(f'{key}: {value}')

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_path'], 
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Ensure chat template is properly set
        if self.tokenizer.chat_template is None:
            self.logger.warning("No chat template found, setting default template")
            self.tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

        # Load MIMIC dataset
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
        self.dataset = self.dataset.remove_columns(['subject_id', 'study_id', 'finding', 'impression'])
        
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

        # No PEFT configuration needed

        # Setup training configuration
        self.trainer_config = SFTConfig(
            # Model and data
            model_init_kwargs=dict(
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if self.config.get('bf16', True) else torch.float16,
                device_map="auto" if self.config.get('device_map_auto', True) else None
            ),
            
            # Output and logging
            run_name=self.config['run_name'],
            output_dir=self.config['save_path'],
            logging_dir=self.config['save_path'],
            
            # Training parameters
            per_device_train_batch_size=self.config['per_device_train_batch_size'],
            per_device_eval_batch_size=self.config['per_device_eval_batch_size'],
            gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
            learning_rate=self.config['learning_rate'],
            num_train_epochs=self.config['num_epochs'],
            max_steps=self.config.get('max_steps', -1),
            
            # Sequence lengths
            max_seq_length=self.config['max_seq_length'],
            
            # Optimization
            optim=self.config.get('optimizer', 'adamw_torch'),
            lr_scheduler_type=self.config.get('lr_scheduler_type', 'cosine'),
            warmup_ratio=self.config.get('warmup_ratio', 0.1),
            weight_decay=self.config.get('weight_decay', 0.01),
            
            # Memory optimization
            gradient_checkpointing=self.config.get('gradient_checkpointing', True),
            bf16=self.config.get('bf16', True),
            fp16=self.config.get('fp16', False),
            dataloader_num_workers=self.config.get('dataloader_num_workers', 4),
            
            # Logging and evaluation
            logging_strategy="steps",
            logging_steps=self.config['log_interval'],
            eval_strategy="steps" if self.config['eval_interval'] != -1 else "no",
            eval_steps=self.config['eval_interval'] if self.config['eval_interval'] != -1 else None,
            save_strategy="steps",
            save_steps=self.config['save_interval'],
            save_total_limit=self.config.get('save_total_limit', 3),
            
            # Other settings
            seed=self.config['seed'],
            remove_unused_columns=False,
            report_to=self.config.get('report_to', []),
            load_best_model_at_end=True if self.config['eval_interval'] != -1 else False,
            metric_for_best_model="eval_loss" if self.config['eval_interval'] != -1 else None,
            greater_is_better=False if self.config['eval_interval'] != -1 else None,
            

        )

        # Setup trainer
        self.trainer = SFTTrainer(
            model=self.config['model_path'],
            args=self.trainer_config,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['test'] if self.config['eval_interval'] != -1 else None,
        )

    def train(self):
        # Load checkpoint
        checkpoint = None
        if self.config['resume_last'] and os.path.isdir(self.config['save_path']):
            checkpoint = get_last_checkpoint(self.config['save_path'])
            if checkpoint:
                self.logger.info(f"Resuming from checkpoint: {checkpoint}")

        # Start training
        self.logger.info('=============== Start Training ===============')
        results = self.trainer.train(resume_from_checkpoint=checkpoint)
        metrics = results.metrics
        metrics['train_samples'] = len(self.dataset['train'])
        self.trainer.log_metrics('train', metrics)
        self.trainer.save_metrics('train', metrics)

    def evaluate(self):
        # Start evaluation
        self.logger.info('=============== Start Evaluation ===============')
        metrics = self.trainer.evaluate()
        metrics['eval_samples'] = len(self.dataset['test'])
        self.trainer.log_metrics('eval', metrics)
        self.trainer.save_metrics('eval', metrics)

    def save(self):
        # Start saving
        self.logger.info('=============== Start Saving ===============')
        self.trainer.save_model(self.config['save_path'])
        if self.trainer.accelerator.is_main_process:
            self.trainer.create_model_card(
                model_name=self.config['run_name'],
                dataset_name=self.config['dataset_path'],
                tags=['Medical-AI', 'SFT', 'MIMIC-CXR', 'Chest-X-Ray']
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML config file'
    )
    args = parser.parse_args()
    
    # Create pipeline and run training
    pipeline = SFTPipeline(args.config)
    pipeline.train()
    pipeline.save()
    
    # Run evaluation if enabled
    if pipeline.config['eval_interval'] != -1:
        pipeline.evaluate()


if __name__ == '__main__':
    main() 