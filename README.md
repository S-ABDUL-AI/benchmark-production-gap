# Benchmark-Production Gap: Measuring AI Safety Evaluation Validity

A reproducible pipeline for quantifying the gap between AI safety
benchmark scores and real-world harmful output rates in production.

## What This Project Does

AI safety benchmarks (HarmBench, StrongREJECT) measure how well models
resist adversarial prompts in controlled settings. This project asks a
different question: **do those scores predict how models actually behave
with real users?**

Using a triple-source design — LMSYS-Chat-1M (1M real user conversations),
HarmBench, and StrongREJECT — we apply Oaxaca-Blinder decomposition to
separate the benchmark-production gap into two components:

- **Composition effect** — differences in what users actually ask vs.
  what benchmarks test
- **Within-category effect** — genuine safety gaps the benchmark misses
  even on comparable prompts

Production harmful-output rates are estimated using Prediction-Powered
Inference (PPI), calibrating LLM labels against human-annotated ground
truth to produce statistically valid confidence intervals.

## Data Sources

| Dataset | Description | Access |
|--------|-------------|--------|
| [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) | 1M real user–model conversations | Public, HuggingFace |
| [HarmBench](https://github.com/centerforaisafety/HarmBench) | Standardized AI safety benchmark (~400 adversarial prompts) | Public, GitHub |
| [StrongREJECT](https://github.com/alexandrasouly/strongreject) | Secondary safety benchmark with open codebase | Public, GitHub |

All data is publicly available. No proprietary data. No IRB required.

> ⚠️ LMSYS-Chat-1M requires a HuggingFace account and acceptance of the
> dataset's terms of use. Set `HF_TOKEN` in your `.env` file before running.

## Methodology

1. Sample LMSYS-Chat-1M by model and harm category
2. Run open-weight models on HarmBench and StrongREJECT to obtain
   benchmark scores
3. Label production conversations using LLM classifier + PPI correction
   against human-annotated subsample
4. Apply Oaxaca-Blinder decomposition to quantify the gap and its sources
5. Report model-level gap scores with confidence intervals

## Reproducing the Analysis

```bash
git clone https://github.com/YOUR-USERNAME/benchmark-production-gap
cd benchmark-production-gap
cp .env.example .env        # fill in your API keys and HF token
pip install -r requirements.txt
python run_analysis.py
```

Install HarmBench and StrongREJECT separately per their own instructions:
- https://github.com/centerforaisafety/HarmBench
- https://github.com/alexandrasouly/strongreject

## Status

Active development. Funded by [BlueDot Impact Rapid Grant — pending].

## Author

[Your Name] — [Your affiliation or "Independent researcher"]
[your-email@domain.com]

## License

MIT License. See LICENSE for details.
