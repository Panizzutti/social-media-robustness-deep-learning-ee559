# Robustness of Multimodal Misogyny Detection to Social-Media Obfuscations

## Project Overview

This project studies whether social-media-style visual obfuscations can affect a meme-specialized vision-language model for misogyny detection. The original motivation was video and more general social-media obfuscation, but the implemented proof of concept uses static MAMI meme images so that emoji overlays can be tested in a controlled setting.

The base model is `QCRI/MemeLens-VLM`. The project compares the original model with a LoRA-adapted version.

## Repository Structure

Important script groups:

- `scripts/mami_eval/`: baseline MemeLens-VLM evaluation, clean and obfuscated image evaluation, unseen overlay geometry evaluation, alpha sweeps, and plotting/analysis helpers.
- `scripts/mami_finetuning/`: 4x4 emoji-grid training augmentation, LoRA fine-tuning, fine-tuned model evaluation, and final comparison/absolute metric reports.
- `results/`: generated experiment outputs. These are useful for local analysis but should generally not be treated as source code.

## Data and External Files

The MAMI dataset is not included in this repository. MAMI is the SemEval-2022 Multimedia Automatic Misogyny Identification dataset and is distributed only upon request for academic use.

The scripts expect the binary MAMI label `misogynous`, where `1` means misogynous and `0` means non-misogynous/clean. The original split is 10,000 training memes and 1,000 test memes.

## Running

Single-GPU distributed evaluations can be launched with `torchrun --standalone --nproc_per_node=1`. For example:

```bash
torchrun --standalone --nproc_per_node=1 scripts/mami_eval/eval_baseline_unseen_random_geometries.py --mode no_text
torchrun --standalone --nproc_per_node=1 scripts/mami_finetuning/eval_finetuned_unseen_random_geometries.py --mode with_text
```

For the fine-tuned unseen-random-geometry file, `eval_finetuned_unseen_random_geometries.py`, demo runs can limit the number of images and emojis. We used this option for the code demonstration.

```bash
torchrun --standalone --nproc_per_node=1 eval_finetuned_unseen_random_geometries.py --mode with_text --max_images 50 --max_emojis 5
```

## Limitations

This project is a proof of concept, not a deployable moderation system. It is limited to static images, emoji overlays, one dataset, one main VLM, and fixed opacity `alpha = 0.8` in the main unseen-geometry evaluation.

## LLM Usage Declaration

LLMs (ChatGPT/Codex) were used to support parts of this project for assistance in code writing, helping debugging and grammar checks. All experimental design choices, code changes, result interpretation, and final submitted content were reviewed and accepted by the project authors, who remain responsible for the work.

## Citation / Acknowledgements

This project uses or builds on the following external resources:

* [MAMI: Multimedia Automatic Misogyny Identification](https://github.com/MIND-Lab/SemEval2022-Task-5-Multimedia-Automatic-Misogyny-Identification-MAMI-), the dataset from [SemEval-2022 Task 5](https://aclanthology.org/2022.semeval-1.74/) used for misogynous / non-misogynous meme classification.
* [`QCRI/MemeLens-VLM`](https://huggingface.co/QCRI/MemeLens-VLM), the meme-specialized vision-language base model used in this project.
* [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), used for parameter-efficient fine-tuning of the base model.
* [Google Noto Emoji](https://github.com/googlefonts/noto-emoji), used for the emoji PNG overlay assets. According to the Noto Emoji repository, the emoji fonts are licensed under the [SIL Open Font License 1.1](https://github.com/googlefonts/noto-emoji/blob/main/LICENSE), while tools and most image resources are licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

The MAMI dataset, external model files, Hugging Face cache, generated augmented datasets, and trained adapter weights are not included in this repository.
