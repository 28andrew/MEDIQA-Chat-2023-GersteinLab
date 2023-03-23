# Important
- Model files are available in [Google Drive](https://drive.google.com/drive/folders/1MO82aRLUuCmZVT6joaKWxskQUJRGb9ip?usp=sharing)
- After cloning this repository download these folders and files from the above link: `best-84.11`, `checkpoint-600`, `checkpoint-7776`, `final-roberta-classifier`, and `RoBERTa-base-PM-M3-Voc-distill-align-hf`
- `best-84.11` / `final-roberta-classifier`: Best RoBERTa with classification head checkpoint
- `checkpoint-600`: Best BART checkpoint trained via [CONFIT](https://github.com/XiangruTang/CONFIT) with attempt to add length as a loss
- `checkpoint-7776`: Best BART checkpoint trained via [CONFIT](https://github.com/XiangruTang/CONFIT)
- `RoBERTa-base-PM-M3-Voc-distill-align-hf`: From [bio-lm](https://github.com/facebookresearch/bio-lm)

Task A - Run 1:
- Classification: RoBERTa with classification head
- Summaries: Trained CONFIT (version 1)

Task A - Run 2:
- Classification: Fine-tuned OpenAI davinci
- Summaries: Trained CONFIT (version 1)

Task A - Run 3:
- Classification: RoBERTa with classification head
- Summaries: Trained CONFIT (version 2) with dynamic max length

Task B - Run 1:
- Fine-tuned OpenAI davinci