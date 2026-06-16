# AI-Powered Resume Screening & Job Recommendation System
End-to-end AI Job Recommendation System. Features an Adzuna API parsing pipeline, data cleaning, and complex feature vectorization. Leverages TF-IDF, Dimensionality Reduction (LSA), and Supervised/Unsupervised models (KNN, Naive Bayes) to dynamically match resumes with real-world job offers via an interactive app dashboard.


##  Project Overview & Motivation

Traditional Applicant Tracking Systems (ATS) rely heavily on exact keyword filtering, which critically fails when candidates use synonyms, submit unstructured formats, or when contextual alignment is ignored. 

**Our Solution:** An intelligent automated pipeline that leverages semantic NLP vectorization to:
* Process large volumes of resumes and job posts efficiently.
* Comprehend hidden textual meaning beyond static keywords.
* Predict specialized professional fields and dynamically surface real-world job vacancies with itemized **Matching Scores (%)**.

---

## 📊 Dataset Ingestion & Engineering

The system synthesizes data across three primary corporate registries:
1. **IT Resume Dataset (`resumes_database.csv`):** Structured technical profiles.
2. **Diverse Resume Dataset (`resumes_diverse_21_spheres.csv`):** Cross-industry profiles spanning 21 vocational domains[cite: 1].
   link to resumes database from Kaggle: https://www.kaggle.com/datasets/saugataroyarghya/resume-dataset
3. **Scraped Job Descriptions Dataset (`adzuna_diverse_jobs_with_medical.csv`):** Live vacancies harvested via the Adzuna API, capturing titles, descriptions, skills, salaries, and localized matrices.
   link to Adzuna: https://www.adzuna.com/

### Preprocessing & Standardization Pipeline
* **Feature Merging:** Consolidated granular parameters (`career objective`, `skills`, `responsibilities`, `positions`, `degree names`, `major field of studies`) into a unified text feature block.
* **List-to-Text Normalization:** Extracted and string-joined Python list objects embedded inside source data rows into space-separated textual structures].
* **Regex Sanitation:** Transformed raw attributes by executing lowercase casting, dropping punctuation/special syntax, and removing empty string placeholders.
* **Outlier Filtration:** Monitored sequence weights through Exploratory Data Analysis (EDA) and eliminated empty or abnormally truncated records.

---  
##  Machine Learning Experimentation

We benchmarked three distinct architectures to identify the optimal matching mechanism[cite: 1]:

### 1. TF-IDF + Naive Bayes (Baseline Model)
* **Method:** Text converted via Term Frequency-Inverse Document Frequency (TF-IDF) and categorized via a Multinomial NB classifier.
* **Limitation:** Constrained by syntax rigidity (treats "ML" and "Machine Learning" as unrelated terms).

### 2. TF-IDF + K-Nearest Neighbors (KNN)
* **Method:** Instantiated a stable spatial clustering model ($K=5$) to isolate nearest professional neighbors based on topological text proximity.
* **Advantage:** Highly intuitive for demographic classification and multi-label baseline profiling.

### 3. Latent Semantic Analysis (LSA) + Cosine Similarity (Final Selected Model)
* **Method:** Projected ultra-sparse high-dimensional TF-IDF matrices (5,000 components) down to 100 dense latent features using **Truncated SVD**.
* **Advantage:** Successfully uncovers semantic correlations and interprets domain synonyms (e.g., mapping `AI` $\approx$ `Artificial Intelligence`).

---
##  Hybrid Production Demo Workflow

The application interface operates on a sequential hybrid framework combining both supervised classification and unsupervised recommendation:
Step 1: Classification (KNN): Predicts Best-Fit Specialist Category
Step 2: Dimensionality Reduction (LSA): Projects inputs into dense latent space
Step 3: Cosine Similarity Matching: Measures distance against Adzuna Job Registry

Output: Top Matches & Matching % Scores ]


git clone [https://github.com/YOUR_USERNAME/resume-screening-system.git](https://github.com/YOUR_USERNAME/resume-screening-system.git)
   cd resume-screening-system

pip install -r requirements.txt
