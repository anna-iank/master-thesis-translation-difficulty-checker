# master-thesis-translation-difficulty-checker

## Project History

This repository contains a refactored implementation of the tool originally developed for my master's thesis (2021). The refactoring improves reproducibility, compatibility, and maintainability while preserving the original methodology.

Please refer to **Master's Thesis – Appendix 2** for the original codebase submitted in 2021 (v1).

## Refactored Version (v2)

This version (`main_script_v2`) has been refactored as follows:

1. The fine-tuned model has been uploaded to a Hugging Face repository (annaiankovskaia/bert-base-uncased-phrasal-verbs) and is now loaded directly from there instead of Google Drive.
2. Compatibility issues between the model fine-tuned in 2021 and the current version of the `transformers` library in Google Colab have been resolved.
3. Google Drive mounting has been replaced with reading files directly from the Colab session.
4. The output format has been changed from `.txt` to `.json`, providing a structured format for linguistic annotations. Noun phrase patterns have also been added to the JSON schema.
5. Phrasal verb classification labels have been changed from `YES`/`NO` to `PV`/`0` for improved clarity.
6. General code clean-up and refactoring have been performed to improve readability and maintainability.

---

## Repository Structure

```text
master-thesis-translation-difficulty-checker
│── baselines/
│   ├── dummy_classifiers
│   ├── LSTM_with_characters_baseline
│   ├── main_training_script_bert
│   └── RandomForest
│── dataset/
│   ├── dataset_fine_tuning
│   └── LICENSE_DATASET
│── sample/
│   └── test_input
│── main_script_v2
│── README.md
│── LICENSE
│── THIRD_PARTY_LICENSES
│── requirements.txt
```

---

## What the Tool Does

The tool implements a three-stage pipeline for analysing the partial translation complexity of English texts intended for Russian neural machine translation.

* **Component 1** processes each sentence as a token sequence and classifies every token as either part of a phrasal verb or not. Phrasal verbs are treated as one potential source of translation complexity.
* **Component 2** applies an NLTK-based parser to extract noun phrases using a set of predefined grammatical patterns. Noun phrases represent another potential source of complexity for English–Russian NMT.
* **Component 3** combines the outputs of the previous components and calculates a complexity score for the analysed text.

---

## Running the Tool

A Google account is required to use Google Colab.

1. Open Google Colab.
2. Select **Open Notebook**.
3. Open `main_script_v2`.
4. Follow the instructions provided in the notebook:

   * upload an input text (a sample file is included);
   * select **Run All**;
   * wait for processing to complete;
   * download the generated JSON annotation.

---

## Fine-Tuning Dataset

The fine-tuning dataset (`dataset/dataset_fine_tuning`) uses the following annotation schema:

```text
id token tag
```

where:

* `id` denotes the sentence identifier;
* `tag = NO` indicates that the token is not part of a phrasal verb;
* `tag = YES` indicates that the token belongs to a phrasal verb.

The corpus contains both compositional and non-compositional English phrasal verbs collected from news commentary.

### Dataset Statistics

| Item              |   Count |
| ----------------- | ------: |
| Tokens            | 157,565 |
| Sentences         |   6,359 |
| Texts             |     170 |
| Phrasal verb tags |   2,111 |
| Other tags        | 155,454 |

---

## Baseline Models

The baseline scripts for **Component 1** located in `baselines/` have been uploaded without refactoring. The only modification is the hardcoded path to the fine-tuning dataset.

The table below compares their F1 scores with the fine-tuned BERT model used in the main pipeline.

| Model                                                  | F1 score |
| ------------------------------------------------------ | -------: |
| Dummy classifier (stratified)                          |     0.01 |
| Dummy classifier (most frequent)                       |     0.00 |
| Dummy classifier (prior)                               |     0.00 |
| Dummy classifier (uniform)                             |     0.03 |
| Dummy classifier (constant)                            |     0.03 |
| Random Forest                                          |     0.20 |
| Bidirectional LSTM                                     |     0.21 |
| BertForTokenClassification (fine-tuned, main pipeline) | **0.67** |


**Overall tool evaluation**

| Component | F1 score |
|------|---------:|
| Full translation difficulty checker pipeline (two components) | 0.18 |


---

## Requirements

* The Google Colab notebook installs all required dependencies automatically.
* To run the scripts in `baselines/`, install the dependencies listed in `requirements.txt`.
