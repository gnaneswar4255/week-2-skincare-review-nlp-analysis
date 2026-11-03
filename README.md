# Review-Informed Skincare Product Discovery Using NLP and Data Analytics

![Domain](https://img.shields.io/badge/Domain-Beauty%20%26%20Wellness-coral)
![Language](https://img.shields.io/badge/Language-Python-blue)
![Stage](https://img.shields.io/badge/Stage-Week%202%20EDA-success)
![Status](https://img.shields.io/badge/Status-Initial%20EDA%20Completed-brightgreen)

## Project Overview

This project explores how Python-based data analysis and Natural Language Processing (NLP) can be used to examine customer reviews and star ratings for beauty and skincare-related products.

The goal is to identify patterns in customer feedback, understand rating distributions, and prepare the dataset for later review-theme analysis. The project is being developed as part of the **YuvaIntern Data Science with Python Analyst Internship**.

The Week 2 work moves the project from the Week 1 problem definition into data sourcing, dataset inspection, and initial exploratory data analysis (EDA).

> **Scope note:** The selected source contains broad beauty-product reviews. It has not yet been verified as a skincare-only dataset. Product metadata must be checked and a defensible skincare subset selected before making skincare-category claims.

## Business Problem

Average star ratings compress different customer experiences into one number. Written reviews may explain the reasons behind those ratings, but manually reading large volumes of feedback can be slow and inconsistent.

**Problem question:** How can product ratings and written reviews be organized into useful summaries of customer feedback across skincare-related product categories?

## Week 2 Objectives

1. Identify and compare publicly accessible data sources relevant to beauty and skincare review analysis.
2. Select a working dataset and document its access route, available fields, and limitations.
3. Load a manageable sample into JupyterLab using Python and Pandas.
4. Inspect the dataset structure, data types, missing values, duplicates, ratings, and review text.
5. Calculate initial descriptive statistics and create a rating-distribution visualization.
6. Connect the analysis plan to the research questions and hypotheses defined in Week 1.
7. Document likely data-cleaning requirements, risks, and limitations.

## Data Source

### Selected working dataset

**Amazon Beauty Reviews dataset — hosted on Hugging Face**

Dataset page: https://huggingface.co/datasets/jhan21/amazon-beauty-reviews-dataset

The dataset was accessed as a CSV file using the Hugging Face filesystem URI:

```python
dataset_path = "hf://datasets/jhan21/amazon-beauty-reviews-dataset/amazon_beauty_reviews_dataset.csv"
```

The notebook loaded the first 10,000 records for initial exploration. This was a manageable starting sample for local analysis.

### Alternative source considered

**Amazon Reviews'23 — All_Beauty category, McAuley Lab**

- Dataset overview: https://amazon-reviews-2023.github.io/main.html
- Dataset card: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

This is a documented large-scale review resource, but its All_Beauty category is broader than skincare. Product metadata must be inspected before treating records as skincare reviews.

### Data-use note

Technical accessibility does not automatically establish unrestricted reuse. Review the dataset card, provenance, licence, and current terms before redistribution or publishing raw data. The raw dataset is not included in this repository.

## Environment Setup: Conda and JupyterLab

The project was developed in a separate Conda environment to keep its Python packages isolated from other projects.

### 1. Create the Conda environment

Open **Anaconda Prompt** and run:

```bash
conda create --solver=classic -n skincare-ds python=3.12 pandas numpy matplotlib seaborn jupyterlab ipykernel -y
```

### 2. Activate the environment

```bash
conda activate skincare-ds
```

The prompt should begin with:

```text
(skincare-ds)
```

### 3. Install Hugging Face and filesystem support

```bash
python -m pip install huggingface_hub fsspec
```

### 4. Install Jupyter widget support

```bash
python -m pip install ipywidgets
```

This also resolves the `IProgress not found` progress-bar warning that can appear when notebook widgets are unavailable.

### 5. Register the environment as a Jupyter kernel

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name skincare-ds --display-name "Python (skincare-ds)"
```

### 6. Launch JupyterLab from the project folder

Change to the project directory (update the path if the folder is stored elsewhere):

```bash
cd /d "C:\Users\muppu\OneDrive\Desktop\Week2_SkinCare_Review"
jupyter lab
```

In JupyterLab, open the project notebook and select:

**Kernel → Change Kernel → Python (skincare-ds)**

### 7. Verify the active environment

Run this in a notebook cell:

```python
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import fsspec
import huggingface_hub

print("Python executable:", sys.executable)
print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("Environment is ready!")
```

The Python executable should point to the `skincare-ds` environment, for example:

```text
C:\Users\muppu\anaconda3\envs\skincare-ds\python.exe
```

The verified package versions during this work were:

| Package | Version / status |
|---|---|
| Python | 3.12 environment |
| Pandas | 3.0.6 |
| NumPy | 2.5.4 |
| fsspec | Import verified |
| huggingface_hub | Import verified |
| ipywidgets | Installed; import verified |

Package versions can change when the environment is recreated. Check the output of the verification cell rather than assuming versions will always match.

## Loading the Dataset in JupyterLab

After selecting the `Python (skincare-ds)` kernel, run:

```python
import pandas as pd

dataset_path = "hf://datasets/jhan21/amazon-beauty-reviews-dataset/amazon_beauty_reviews_dataset.csv"

df_raw = pd.read_csv(dataset_path, nrows=10000)
```

The `nrows=10000` argument limits the initial load to 10,000 rows.

To inspect the loaded data:

```python
df_raw.head()
```

For the Week 2 exploration, a working copy can be created while preserving the raw DataFrame:

```python
df = df_raw.copy()
```

- `df_raw` keeps the originally loaded sample unchanged.
- `df` is the working DataFrame for analysis and later cleaning.

If the `hf://` URI cannot be read, first verify that `fsspec` and `huggingface_hub` import successfully in the active notebook kernel. Restart the kernel after installing packages, then retry the loading cell.

## Dataset Structure

The inspected sample contained **10,000 rows and 10 columns**.

| Field | Observed type | Initial use |
|---|---|---|
| `rating` | int64 | Star-rating distribution and comparisons |
| `title` | string | Short review heading |
| `text` | string | Written review content and text analysis |
| `images` | string | Image metadata representation; requires source-specific interpretation |
| `asin` | string | Product identifier |
| `parent_asin` | string | Parent product identifier |
| `user_id` | string | Reviewer identifier; not needed for the initial analysis |
| `timestamp` | string | Review timestamp; requires parsing and validation |
| `helpful_vote` | int64 | Helpful-vote count, if relevant |
| `verified_purchase` | bool | Purchase-verification indicator |

These are the types observed in the loaded sample. They should be checked against the dataset's official documentation.

## Week 2 Exploratory Data Analysis

### 1. Dataset inspection

The notebook used Pandas operations to inspect the first records, columns, dimensions, data types, and dataset information.

Examples:

```python
df.head()
df.shape
df.columns
df.info()
```

### 2. Repeated review titles and text

The `value_counts()` method was used to identify exact repeated values in the title and review-text columns.

Examples:

```python
df["title"].value_counts(dropna=False).head(10)
df["text"].value_counts(dropna=False).head(10)
```

Some of the most frequent titles were:

| Review title | Count |
|---|---:|
| Five Stars | 528 |
| Four Stars | 113 |
| Three Stars | 57 |
| One Star | 53 |
| Great | 43 |

Some of the most frequent exact review texts were:

| Review text | Count |
|---|---:|
| Love it | 17 |
| Great product | 13 |
| Great | 13 |
| Perfect | 10 |
| Nice | 9 |

Exact text counts do not group different phrases with the same meaning. Theme analysis is planned for a later stage.

### 3. Rating distribution

The rating distribution was calculated using `value_counts()` and sorted by rating.

| Star rating | Review count | Percentage |
|---:|---:|---:|
| 1 star | 858 | 8.58% |
| 2 stars | 581 | 5.81% |
| 3 stars | 977 | 9.77% |
| 4 stars | 1,688 | 16.88% |
| 5 stars | 5,896 | 58.96% |
| **Total** | **10,000** | **100.00%** |

The 5-star group was the largest group in the loaded sample. The combined share of 4-star and 5-star reviews was 75.84%.

### 4. Rating summary statistics

| Measure | Result |
|---|---:|
| Mean rating | 4.1183 |
| Median rating | 5.0 |
| Mode rating | 5 |

The mean describes the average rating, the median is the middle rating, and the mode is the most frequently occurring rating.

### 5. Data-quality checks

| Check | Observed result |
|---|---:|
| Missing ratings | 0 |
| Missing titles | 1 |
| Missing review texts | 1 |
| Exact duplicate rows | 13 |
| Average review length | 297.16 characters |
| Median review length | 162.5 characters |
| Minimum review length | 0 characters |
| Maximum review length | 10,205 characters |

The duplicate check used exact row matching across all columns. The 13 exact duplicate rows should be inspected against product/review identifiers and source semantics before removal. Identical review text alone is not sufficient evidence that two records are duplicates.

Missing review text should not be treated as negative feedback. Missing values should be handled according to the analysis being performed.

### 6. Rating distribution visualization

A Matplotlib bar chart was created to show the number of reviews for each star rating. The chart makes the concentration of 5-star reviews visible and complements the rating-frequency table.

## Initial Findings and Interpretation

The first 10,000 records show a mean rating of 4.1183, with 58.96% of the sample receiving 5 stars. The median and mode are both 5. The sample also contains one missing title, one missing review text, and 13 exact duplicate rows.

These are descriptive observations about the loaded sample only. They should not be generalized to the full dataset, all beauty products, or the skincare market.

## Link to Week 1 Research Questions and Hypotheses

| Week 1 question / hypothesis | Week 2 progress | Remaining work |
|---|---|---|
| RQ1: Common review themes | Exact repeated titles and review texts inspected | Extract and validate frequent words, phrases, and themes |
| H1: Themes differ across skincare categories | Dataset sourcing and initial inspection completed | Validate skincare products and categories before comparison |
| H2: Lower- and higher-rated reviews differ | Rating distribution and summary statistics completed | Compare review text and themes across defined rating groups |
| H3: Simple sentiment may disagree with manual interpretation | Not tested in Week 2 | If sentiment analysis is used, compare predictions with a manually reviewed sample |
| H4: Ratings plus review text provide richer summaries | Rating-only descriptive statistics completed | Add validated review-theme summaries and compare the information provided |

**The Week 2 notebook does not confirm or reject H1–H4.** Category and text-theme comparisons remain future work.

## Planned Data Cleaning and Preprocessing

The next stage will consider the following tasks:

- Preserve the raw DataFrame and perform changes on a separate working copy.
- Inspect missing title and review-text values before deciding how each analysis should handle them.
- Investigate the 13 exact duplicate rows using identifiers and source documentation.
- Validate that ratings fall within the expected 1–5 range.
- Parse the timestamp field only after checking its format and validity.
- Inspect product identifiers and metadata to isolate skincare-relevant products.
- Preserve original review text and create a separate processed-text column.
- Standardize case and whitespace carefully.
- Preserve negation words such as “not,” because they can change the meaning of a review.
- Inspect empty, very short, and unusually long reviews before applying any filtering rules.

## Risks and Limitations

| Challenge | Why it matters | Planned response |
|---|---|---|
| Broad beauty dataset | May include non-skincare products | Inspect product metadata and document inclusion rules |
| First 10,000 rows only | May not represent the full dataset | Treat as an initial sample and document sampling limits |
| Missing values | Can reduce usable records | Quantify missingness and handle it by analysis type |
| Exact duplicate rows | May affect counts | Inspect identifiers before removing records |
| Unequal product review counts | Can distort comparisons | Report group sizes and avoid unsupported rankings |
| Short, mixed, or sarcastic reviews | Can mislead simple NLP methods | Manually inspect representative examples |
| Self-selection and platform bias | Reviewers may not represent all buyers | Limit conclusions to the available review sample |
| Licence and provenance | Public access does not guarantee unrestricted reuse | Check current source terms before redistribution |

## Tools and Technologies

- **Python 3.12** — programming language
- **Pandas** — loading, inspecting, and analysing tabular data
- **NumPy** — numerical operations
- **Matplotlib** — rating-distribution visualization
- **Seaborn** — available for statistical visualizations
- **JupyterLab** — interactive notebook environment
- **Hugging Face Hub and fsspec** — accessing the hosted CSV dataset
- **Conda** — isolated Python environment management

## Repository Contents

Update this list to match the files actually committed to the GitHub repository.

```text
week-2-skincare-review-analysis/
│
├── README.md
├── Week2_Skincare_Reviews.ipynb
├── report/
│   └── Week2_Data_Sourcing_and_Initial_Analysis_Report.docx
└── .gitignore
```

Do not commit the full raw dataset, private credentials, Hugging Face access tokens, or unnecessary reviewer identifiers. The notebook uses the hosted dataset URI to load a limited sample.

## Current Status

**Week 2 — Data sourcing, initial dataset inspection, and baseline EDA completed.**

Completed in this stage:

- Set up a dedicated Conda environment and JupyterLab kernel.
- Installed and verified the required Python and Hugging Face packages.
- Loaded 10,000 records from the Hugging Face-hosted Amazon Beauty Reviews CSV.
- Inspected the sample structure, fields, and data types.
- Explored repeated review titles and exact review text.
- Calculated rating counts, percentages, mean, median, and mode.
- Checked missing values and exact duplicate rows.
- Calculated review-length statistics.
- Created a customer rating distribution chart.
- Documented the initial findings, source limitations, and next analysis steps.

**Not completed in Week 2:** skincare-only product validation, category-level hypothesis testing, NLP theme extraction, sentiment modelling, or confirmation of the Week 1 hypotheses.

## Next Steps

The next stage will focus on validating skincare-related product records and developing a documented data-cleaning and preprocessing workflow. Once the relevant subset is established, the project can proceed to category comparisons, rating-group text exploration, and review-theme analysis.

## References

- Hugging Face dataset: https://huggingface.co/datasets/jhan21/amazon-beauty-reviews-dataset
- Amazon Reviews'23 documentation: https://amazon-reviews-2023.github.io/main.html
- Amazon Reviews'23 dataset card: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- Pandas documentation: https://pandas.pydata.org/docs/
- NumPy documentation: https://numpy.org/doc/
- Matplotlib documentation: https://matplotlib.org/stable/
- JupyterLab documentation: https://jupyterlab.readthedocs.io/
- Hugging Face Hub documentation: https://huggingface.co/docs/huggingface_hub/
