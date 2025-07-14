```bash
conda activate radi
```

```bash
CUDA_VISIBLE_DEVICES=7 python utils/grpo.py --config configs/grpo_mimic_llama3.yaml
CUDA_VISIBLE_DEVICES=6 python utils/sft.py --config configs/sft_mimic_llama3.yaml
```



| Metric               | Model (Llama-3-8B) | Model (SFT) | Method (GRPO) |
|----------------------|--------------------|-------------|---------------|
| Total Samples        | 1603               | 1603        | 1603          |
| BLEU-1 Mean          | 0.0331             | 0.0352      | -             |
| BLEU-4 Mean          | 0.0056             | 0.0059      | -             |
| ROUGE-1 Mean         | 0.0455             | 0.0497      | -             |
| ROUGE-2 Mean         | 0.0011             | 0.0014      | -             |
| ROUGE-L Mean         | 0.0383             | 0.0403      | -             |
| METEOR Mean          | 0.0277             | 0.0284      | -             |
| CIDEr                | 0.0059             | 0.0084      | -             |
| BERTScore Precision  | 0.8300             | 0.8359      | -             |
| BERTScore Recall     | 0.8327             | 0.8294      | -             |
| BERTScore F1         | 0.8312             | 0.8325      | -             |

## TODO
- [] llama3 3b qwen2.5 3b minstral 3b
- [] mimic openi

## Story Line
- [] 分析类型 以往的prompt 或者 微调。
- [] reward** 

