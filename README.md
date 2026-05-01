# ISIC FairReview

Human-in-the-loop fairness review pipeline for ISIC 2019 skin-lesion diagnosis data.

This project uses the ISIC 2019 metadata and labels to identify age/sex groups that may need additional human review before or during model training. The fairness attributes are limited to:

- `age_approx`
- `sex`
- `age_group`
- `age_x_sex`

We intentionally do not use anatomical site as a fairness attribute in this version.

## 1. Goal

The goal is to design and implement a fairness-aware training support system for ISIC 2019.

The system asks:

> Which training samples should be prioritized for human rereview because they belong to underrepresented age/sex groups, have missing age or sex metadata, or represent rare diagnosis-by-group combinations?

The project has four main outputs:

- An EDA notebook: `eda.ipynb`
- A reusable fairness pipeline module: `isic_fairness_pipeline.py`
- A local HTML/CSS/JS review UI: `index.html`, `styles.css`, `script.js`
- A local backend for review saving and training: `local_pipeline_server.py`

The UI is not meant to replace medical expertise. It is a workflow tool for surfacing cases that may need expert verification before being used in model training.

## Data Setup

The ISIC 2019 data is not committed to this repository. Put all downloaded ISIC 2019 files inside a local `data/` folder at the project root.

Expected layout:

```text
data/
  ISIC_2019_Training_Metadata.csv
  ISIC_2019_Training_GroundTruth.csv
  ISIC_2019_Test_Metadata.csv
  ISIC_2019_Test_GroundTruth.csv
  ISIC_2019_Training_Input/
    ISIC_*.jpg
  ISIC_2019_Test_Input/
    ISIC_*.jpg
```

The app, backend, and EDA notebook all read from this `data/` folder.

Download source:

```text
https://challenge.isic-archive.com/data/#2019
```

## 2. How We Select What Needs To Be Rereviewed

The rereview queue is generated from the training split only.

Each image can be flagged for one of three reasons.

### Missing Age Or Sex Metadata

If `age_approx` or `sex` is missing, the image is sent to the review queue.

Reason label:

```text
missing age or sex metadata
```

This matters because missing demographic metadata prevents subgroup fairness evaluation.

### Underrepresented Age x Sex Group

Each image is assigned to an intersectional group:

```text
age_group | sex
```

Examples:

```text
80+ | female
<20 | male
missing | female
```

For each group, we compute:

```text
group_share = group_count / total_training_count
uniform_share = 1 / number_of_age_x_sex_groups
ratio_to_uniform = group_share / uniform_share
```

If:

```text
ratio_to_uniform < 0.8
```

then the group is considered underrepresented and its samples are prioritized for rereview.

Reason label:

```text
underrepresented age x sex group
```

### Rare Diagnosis Within Age x Sex Group

Some groups may not be rare overall but may be rare for a specific diagnosis.

For example:

```text
MEL | 80+ | female
SCC | <20 | male
DF | 70-79 | female
```

The app computes:

```text
diagnosis_age_x_sex_n
```

If this count is below the threshold, currently set to `25` by default, the sample is sent to rereview.

Reason label:

```text
rare diagnosis within age x sex group
```

The thresholds can be adjusted in the backend and UI controls.

## 3. Project Pipeline

The pipeline is:

```text
ISIC 2019 metadata and labels
        ↓
Load and clean age / sex fields
        ↓
Create age groups and age x sex intersections
        ↓
Compute descriptive EDA and representation fairness metrics
        ↓
Build human rereview queue
        ↓
HTML/CSS reviewer UI
        ↓
Save human review decisions to human_reviews.csv
        ↓
Start lightweight image-plus-metadata training
        ↓
Train model with reviewed labels and group-aware sample weights
        ↓
Evaluate model by sex, age group, and age x sex
        ↓
Write post_training_rereview.csv for weak groups or low-confidence errors
```

The local app supports:

- live review queue loading
- blinded one-by-one reviewer label selection
- read-only age and sex context for fairness routing
- diagnosis and confidence selection by the reviewer
- disagreement prompts when the reviewer selection differs from ground truth
- hold-out and uncertainty decisions
- training start after the initial review batch
- training progress, status, and metrics
- one-by-one post-training rereview
- final completion message after rereview is done

The training intervention uses:

```text
image
final_diagnosis
age_group
sex_clean
age_x_sex
training_gate
group_weight
diagnosis_weight
sample_weight
```

Rows marked as `exclude from training` or `mark uncertain` receive:

```text
training_gate = hold_out
sample_weight = 0
```

Other rows receive group-aware and diagnosis-aware weights.

After training, the backend writes:

```text
training_group_metrics.csv
post_training_rereview.csv
```

`post_training_rereview.csv` contains validation cases the model missed in weak age/sex groups or low-confidence error cases.

## 4. Results Of EDA

The local ISIC 2019 files contain:

```text
Training rows: 25,331
Test rows:      8,238
Total rows:    33,569
```

### Missing Metadata

Across train and test:

```text
Missing age rows: 763
Missing sex rows: 723
```

By split:

```text
Train sex missing: 1.52%
Test sex missing:  4.12%
Train age missing: 1.73%
Test age missing:  3.96%
```

### Sex Distribution

Percentage within each split:

| Sex | Train | Test |
|---|---:|---:|
| Female | 46.03% | 45.97% |
| Male | 52.45% | 49.92% |
| Missing | 1.52% | 4.12% |

Main observation: male and female representation is reasonably close, but missing sex metadata is higher in the test split.

### Age Group Distribution

Percentage within each split:

| Age Group | Train | Test |
|---|---:|---:|
| `<20` | 2.70% | 1.51% |
| `20-29` | 4.20% | 3.25% |
| `30-39` | 11.25% | 10.06% |
| `40-49` | 19.07% | 16.38% |
| `50-59` | 18.39% | 15.73% |
| `60-69` | 16.23% | 15.84% |
| `70-79` | 15.46% | 17.41% |
| `80+` | 10.97% | 15.87% |
| Missing | 1.73% | 3.96% |

Main observation: the test set has a larger share of `80+` patients and missing age metadata than the training set.

### Training Diagnosis Distribution

Training labels:

| Diagnosis | Count |
|---|---:|
| NV | 12,875 |
| MEL | 4,522 |
| BCC | 3,323 |
| BKL | 2,624 |
| AK | 867 |
| SCC | 628 |
| VASC | 253 |
| DF | 239 |

Main observation: the diagnostic labels are highly imbalanced. `NV` dominates the training set, while `DF`, `VASC`, `SCC`, and `AK` are much smaller classes.

### Test Diagnosis Distribution

The local test ground-truth file includes:

| Diagnosis | Count |
|---|---:|
| NV | 2,495 |
| UNK | 2,047 |
| MEL | 1,327 |
| BCC | 975 |
| BKL | 660 |
| AK | 374 |
| SCC | 165 |
| VASC | 104 |
| DF | 91 |

Main observation: `UNK` is present in the local test labels, so test diagnosis summaries should be interpreted carefully.

### Representation Fairness Metrics

The most important representation metric is the minimum-to-maximum group ratio.

| Attribute | Split | Smallest Group | Largest Group | Min/Max Ratio |
|---|---|---|---|---:|
| Sex | Train | missing | male | 0.0289 |
| Sex | Test | missing | male | 0.0824 |
| Age group | Train | missing | 40-49 | 0.0905 |
| Age group | Test | `<20` | 70-79 | 0.0865 |
| Age x sex | Train | missing \| female | 70-79 \| male | 0.0102 |
| Age x sex | Test | 70-79 \| missing | 70-79 \| male | 0.0021 |

Main observation: single sex groups are not the biggest concern, but missing metadata and intersectional `age x sex` groups are much more imbalanced.

### Review Queue Results

The generated training review queue contains:

```text
3,489 candidate rows
```

Breakdown:

| Review Reason | Count |
|---|---:|
| Underrepresented age x sex group | 2,772 |
| Missing age or sex metadata | 437 |
| Rare diagnosis within age x sex group | 280 |

Smallest training `age x sex` groups include:

| Age x Sex Group | Count |
|---|---:|
| missing \| female | 26 |
| missing \| male | 27 |
| `<20` \| female | 336 |
| `<20` \| male | 348 |
| missing \| missing | 384 |
| 20-29 \| male | 448 |
| 20-29 \| female | 617 |
| 80+ \| female | 1,023 |

These are the kinds of groups the UI prioritizes for human review.

## 5. How To Run The App

The main app is now a local HTML/CSS/JS webpage with a Python backend.

Run:

```bash
python3 local_pipeline_server.py
```

Then open:

```text
http://127.0.0.1:8502
```

The backend is required for:

- loading the live review queue
- saving reviewer decisions
- starting training
- showing post-training rereview cases

You can still open `index.html` directly, but that is only a static preview.

If dependencies are missing:

```bash
python3 -m pip install -r requirements.txt
python3 local_pipeline_server.py
```

If you want to use a fresh virtual environment:

```bash
python3 -m venv .venv2
.venv2/bin/pip install -r requirements.txt
.venv2/bin/python local_pipeline_server.py
```

The app reads these files from the `data/` folder:

```text
data/ISIC_2019_Training_Metadata.csv
data/ISIC_2019_Training_GroundTruth.csv
data/ISIC_2019_Test_Metadata.csv
data/ISIC_2019_Test_GroundTruth.csv
```

If training images are available in:

```text
data/ISIC_2019_Training_Input/
```

then the review screen displays the lesion image. If the image file is not available yet, the app still allows metadata and label review.

Human review decisions are saved to:

```text
human_reviews.csv
```

The training process uses:

```text
reviewed labels
hold-out decisions
age x sex sample weights
diagnosis sample weights
real lesion image features
age and sex metadata features
```

After training, the backend writes:

```text
training_group_metrics.csv
post_training_rereview.csv
```

The webpage displays the post-training rereview queue automatically after training finishes.

## References

- ISIC Challenge data page, 2019 dataset: https://challenge.isic-archive.com/data/#2019
- ISIC 2019 Challenge: BCN_20000, HAM10000, and MSK datasets provided through the International Skin Imaging Collaboration archive.
