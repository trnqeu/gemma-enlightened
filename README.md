# Gemma Enlightened

Fine-tuning **Gemma 2 2B** to preserve the language and thought of the **Italian Enlightenment** (Illuminismo italiano, 18th century).

The idea: besides archives and critical studies, can a language model become a new way to preserve a cultural context, so that a philosophical movement's words and ideas can be carried into the present and applied to today's questions?

The project was originally published on Kaggle as a submission to the [Google - Unlock Global Communication with Gemma](https://www.kaggle.com/competitions/gemma-language-tuning) competition.

- 📝 Article: [Gemma Enlightened](https://trnq.eu/en/ideas/2025-01-10-gemma-enlightened/)
- 📓 Kaggle notebook: [stefanotrinchero/gemma-enlightened](https://www.kaggle.com/code/stefanotrinchero/gemma-enlightened)

## What's inside

| File | Description |
|------|-------------|
| [`gemma-enlightened.ipynb`](gemma-enlightened.ipynb) | The full project write-up: source collection, text processing, dataset building, training and inference examples |
| [`Gemma_Enlightened_lora_tuning.ipynb`](Gemma_Enlightened_lora_tuning.ipynb) | The LoRA fine-tuning notebook used on Google Colab |

## How it works

1. **Sources**: public-domain texts by Neapolitan and Lombard Enlightenment authors (Beccaria, Pietro and Alessandro Verri, Filangieri, Pagano, Algarotti and others), mostly from [Liber Liber](https://liberliber.it/). They cover justice, economics, science, education, customs, architecture and more.
2. **Processing**: the texts are cleaned and split into sentence chunks with [spaCy](https://spacy.io/) (`it_core_news_lg`), then reviewed by hand.
3. **Dataset**: each 18th-century fragment (the *completion*) is paired with a question in contemporary Italian (the *prompt*). Some prompts were written by hand; the rest were generated with Llama 3.1 8B via Replicate and then reviewed. The final dataset has **12,379 prompt/response pairs**.
4. **Training**: LoRA fine-tuning (rank 4) of `gemma2_2b_en` with Keras NLP on a JAX backend, using AdamW (lr 5e-5, weight decay 0.01) and a sequence length of 256.
5. **Inference**: the model answers modern questions in the style and with the ideas of an 18th-century Italian Enlightenment philosopher.

## Example

**Prompt**: *Qual è il ruolo dell'educazione nella formazione di un cittadino illuminato e responsabile?*

**Response (translated)**: *"It is virtue, not fortune, that makes you illustrious [...] for education is the mother of virtue, for ignorance is the mother of slavery, and it is ignorance that has given humanity this sad condition of not being free, and of living in a continuous state of fear and disorder."*

## Model and dataset

The fine-tuned model and the dataset are published on Kaggle:

- Model: `lumigemma/keras/gemma2-enlightened`
- Dataset: `italian-enlightenment-q-and-a`

## Requirements

`keras`, `keras-nlp`, `jax`, `pandas`. To reproduce the preprocessing you also need `spacy` with the `it_core_news_lg` model.
