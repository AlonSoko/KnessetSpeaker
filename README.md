# Knesset Speaker: Identifying Speaker Names in the Knesset Corpus and Matching Them to Actual Knesset Members
Authored by: Mr. Alon Sokolovski – alon.sokolovski@gmail.com
Supervised by: Prof. Shuly Wintner – shuly@cs.haifa.ac.il, Mrs. Gili Goldin – gili.sommer@gmail.com
Department of Computer Science, University of Haifa, Israel

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
    - [Matching Speakers to Actual Knesset Members](#matching-speakers-to-actual-knesset-members)
  - [Sentence Splitting](#sentence-splitting)
  - [Output Format](#output-format)
    - [Why Parquet?](#why-parquet)
    - [Why PyArrow over Polars' Native write\_parquet?](#why-pyarrow-over-polars-native-write_parquet)
  - [Testing](#testing)
    - [Overview](#overview-1)
    - [Identifying Speaker Names in the Knesset Corpus](#identifying-speaker-names-in-the-knesset-corpus)
    - [Matching Speaker Names to Actual Knesset Members](#matching-speaker-names-to-actual-knesset-members)
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

### Matching Speakers to Actual Knesset Members

The goal of this process is to accurately match speaker names in the Knesset protocols to the correct Knesset members, even when there are variations in name presentation. The process can be summarized as follows:

1. **Identifying Distinct Speakers**: We begin by extracting all distinct speaker names from the protocol. This is crucial because, despite potentially hundreds of paragraphs in a protocol, the number of distinct speakers is usually under ten.

2. **Creating Name Combinations**: To handle variations in names (e.g., nicknames or multiple given names), we generate all possible combinations of first and last names for each Knesset member. This is done by:
   - Creating a list of name components (first name, last name, and any additional names) for each Knesset member, filtering out null values.
   - Exploding this list so that each row in our DataFrame represents a unique name combination.

   For example:
   ```
   person_id | last_name   | first_name | last_first_name    | last_first_name_word_count
   ---------------------------------------------------------------------
   2118      | אבו רוכן    | לביב       | אבו רוכן לביב      | 3
   2118      | אבו רוכן    | חוסיין      | אבו רוכן חוסיין    | 3
   ```

3. **Cross-Joining Speakers with Knesset Members**: We perform a cross-join of all distinct speakers with the Knesset members, resulting in a table where each speaker is matched with every possible Knesset member. This allows us to compare each speaker's name against every possible name combination.

4. **Handling Name Splitting**: Given that speaker names in the protocol may not be clearly split into first and last names, and might include additional words, we split these names into components and consider all possible combinations. This ensures we do not miss a match due to order or additional words in the speaker's name.

5. **Similarity Calculation**: To compare the generated combinations with the names in the protocol, we:
   - Sort the names lexicographically to avoid mismatches due to order.
   - Use Python’s `SequenceMatcher` from the `difflib` library to calculate the similarity ratio between each combination. This algorithm identifies the longest contiguous matching subsequence between the two strings and uses it to calculate an overall similarity ratio.

6. **Filtering by Threshold**: Based on the similarity ratios, we apply a threshold to filter out poor matches. Through empirical testing, a threshold of 0.9 was found to offer the best balance, ensuring minimal false positives and negatives.

7. **Handling Collisions**: 
   - **Multiple Person IDs for the Same Speaker**: In cases where a speaker name matches multiple Knesset members, we select the member with the highest person ID, which typically corresponds to the most recent Knesset member.
   - **Multiple Matches for the Same Person ID**: If multiple name combinations for the same person ID match a speaker, we retain the combination with the maximum length. If multiple combinations have the same length, the lexicographic maximum is selected.

8. **Final Mapping**: The results are converted into a mapping dictionary that associates each speaker's text with the correct Knesset member, including their person ID and any additional matches.

This thorough matching process ensures that speaker names in the Knesset protocols are accurately linked to the correct Knesset members, despite variations and inconsistencies in name presentation.

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

### Overview

This project had two primary objectives: **identifying speaker names** in the Knesset Corpus and **matching them to actual Knesset members**. Therefore, the testing phase was divided into two parts to evaluate the success of both objectives. The tests were conducted on a randomly selected sample of 10 documents.

### Identifying Speaker Names in the Knesset Corpus

To detect speaker names, the algorithm searches for underlined text between the first speaker index (typically the first mention of the chairman, assumed to start with “היו"ר”) and the last speaker index (which typically ends with “הישיבה נגמרה בשעה HH:MM”). The underlined text is then validated against a predefined regex pattern.

**Key Metrics:**
- **False Positive**: Instances where underlined text was incorrectly identified as a speaker.
  - **Results**: 0%
- **False Negative**: Instances where underlined text representing a speaker was not identified as such.
  - **Results**: 1.2%
- **True Positive**: Instances where underlined text correctly identified a speaker.
  - **Results**: 98.8%
- **True Negative**: Instances where non-speaker underlined text was correctly not identified as a speaker.
  - **Results**: 100%

**Accuracy Calculation**:
```math
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{0.988 + 1.0}{0.988 + 1.0 + 0.0 + 0.012} = \frac{1.988}{2} = 0.994
```
The identification process achieved an accuracy of 99.4%.

**Challenges**:
- An example where a speaker’s name did not match the expected pattern highlighted a limitation: the regex enforced a colon (:) at the end of the speaker’s name, which in some cases led to missed detections.
- Relaxing the regex constraints could introduce false positives, especially in cases where titles or other non-speaker text might be incorrectly identified as speakers.

**Future Improvements**:
- A more sophisticated approach could involve passing questionable cases through a Hebrew-trained language model to determine whether the text indicates a speaker and, if so, identify the speaker.

### Matching Speaker Names to Actual Knesset Members

This part of the test involved manually reviewing all distinct speaker names identified by the algorithm in the document and comparing them against known Knesset members.

**Key Metrics:**
- **False Positive**: Incorrect assignment of a Knesset member to a speaker.
  - **Results**: 0%
- **False Negative**: Instances where a speaker who should not have been assigned a Knesset member was assigned one.
  - **Results**: 0%
- **True Positive**: Correct assignment of a Knesset member to a speaker.
  - **Results**: 100%
- **True Negative**: Correctly not assigning a Knesset member to a non-member speaker.
  - **Results**: 100%

**Accuracy Calculation**:
```math
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{1.0 + 1.0}{1.0 + 1.0 + 0.0 + 0.0} = 1.0
```
The matching process achieved a perfect accuracy of 100%.

**Note**: This accuracy is based on the distinct speakers that were correctly identified in the first phase. Therefore, any undetected speakers were not included in this matching evaluation.

## References

- [The Knesset Corpus: An Annotated Corpus of Hebrew Parliamentary Proceedings](https://arxiv.org/abs/2405.18115) – Gili Goldin, Nick Howell, Noam Ordan, Ella Rabinovich, Shuly Wintner
