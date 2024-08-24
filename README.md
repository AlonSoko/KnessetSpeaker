# Knesset Speaker: Identifying Speaker Names in the Knesset Corpus and Matching Them to Actual Knesset Members
Alon Sokolovski

## Overview

This project focuses on accurately identifying and correcting speaker names in the Knesset Corpus, a large dataset containing the proceedings of Israel's parliament over the past 30 years. The project aims to ensure that every mention of a Knesset member, despite variations in name presentation, is consistently linked to their contributions. This process also involves preventing other text fragments from being mistakenly identified as speaker names.

## Contents

- [Knesset Speaker: Identifying Speaker Names in the Knesset Corpus and Matching Them to Actual Knesset Members](#knesset-speaker-identifying-speaker-names-in-the-knesset-corpus-and-matching-them-to-actual-knesset-members)
  - [Overview](#overview)
  - [Contents](#contents)
  - [Introduction](#introduction)
  - [Document Parsing](#document-parsing)
    - [Reasoning](#reasoning)
    - [Implementation](#implementation)
  - [Metadata](#metadata)
  - [Speaker Matching](#speaker-matching)
  - [Sentence Splitting](#sentence-splitting)
  - [Output Format](#output-format)
    - [Why Parquet?](#why-parquet)
    - [Why PyArrow over Polars' Native write\_parquet?](#why-pyarrow-over-polars-native-write_parquet)
  - [Testing](#testing)
  - [References](#references)

## Introduction

The Knesset Corpus is a comprehensive dataset capturing both plenary and committee deliberations from the Knesset, Israel's parliament, spanning 30 years. This project involved developing methods to accurately identify and correct speaker names in these deliberations, ensuring consistent attribution of text to the correct Knesset members.

## Document Parsing

### Reasoning

Despite access to a pre-processed version of the Knesset Corpus, custom document parsing was implemented due to:

1. **Ensuring Accurate Text Parsing**: The standard `python-docx` library failed to parse files correctly, often missing words. Custom parsing solutions were developed for both Windows (using MS Word's COM object model) and Linux (using LibreOffice in headless mode) to ensure complete and accurate text capture.
2. **Leveraging Document Metadata**: Important metadata like underlines and text alignment, crucial for identifying speakers, was preserved through custom parsing.
3. **Granular Control and Error Handling**: Custom parsing allowed for tailored validation mechanisms and error handling, improving accuracy and consistency.
4. **Enhanced Speaker Identification**: Contextual understanding of the documents facilitated better handling of unique text structures, ensuring more accurate speaker identification.

### Implementation

The project involved separate parsing solutions for Windows and Linux:

- **Windows**: Utilized the COM object model to interact with Microsoft Word.
- **Linux**: Used the UNO API to interact with LibreOffice in headless mode.

Custom functions were implemented for:

- **Name Validation**: Ensuring document names followed a specific format.
- **Document Loading**: Dynamically selecting the appropriate document loading process based on the operating system.
- **Metadata Extraction**: Extracting protocol type, Knesset number, and protocol name from the document.
- **Speaker Identification**: Identifying the first and last speakers, and filtering irrelevant text.
- **Generating Consecutive Speaker Texts**: Creating a list of consecutive speaker texts by iterating over identified speaker names and indexes.

## Metadata

A CSV file containing detailed information about Knesset Members was manually curated to include alternative names or nicknames. A new table was created with columns for multiple name variations, enabling comprehensive name matching in the corpus.

## Speaker Matching

TODO: Add

## Sentence Splitting

The `split_text_to_sentences` method used the NLTK library's `PunktSentenceTokenizer` to divide each speaker's text into individual sentences. This method generated a list of dictionaries, each representing a sentence with associated metadata, making the data easier to analyze.

## Output Format

The `get_results` method converted the processed data into a Polars DataFrame, enriched with additional protocol metadata. The final DataFrame was saved in the Parquet format, ensuring efficient storage and retrieval.

### Why Parquet?

- **Efficiency**: Columnar storage format allows for efficient compression and faster query performance.
- **Compression**: Highly compressed files save storage space.
- **Schema Evolution**: Supports adding new columns or modifying the schema over time.
- **Big Data Compatibility**: Integrates well with distributed processing frameworks.

### Why PyArrow over Polars' Native write_parquet?

- **Advanced Partitioning**: PyArrow handles disk partitioning more effectively, optimizing data retrieval and storage.

## Testing

TODO: Add

## References

- [The Knesset Corpus: An Annotated Corpus of Hebrew Parliamentary Proceedings](https://arxiv.org/abs/2405.18115) – Gili Goldin, Nick Howell, Noam Ordan, Ella Rabinovich, Shuly Wintner