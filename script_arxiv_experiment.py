from __future__ import annotations   # optional but future‑proof
import json
from typing import List
from collections import Counter
from tqdm import tqdm

from collections import Counter
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset
import random
import torch
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, precision_score, recall_score
import asyncio, os, re, random
from typing import List, Dict
from tqdm.asyncio import tqdm_asyncio     # ⬅ async‑friendly progress bar
from openai import AsyncOpenAI, RateLimitError
from sklearn.preprocessing import LabelEncoder
import re
#### fit PLSI and LDA on each of the k datasets
from sklearn.decomposition import LatentDirichletAllocation
import sys, os
sys.path.append("/Users/clairedonnat/Documents/LLM_topics/")
from plsi import pLSI
from utils import convert_to_dtm, align_on_confusion
from open_ai_sentence_generation import hide_and_generate_async
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd          # optional, but nice for column names
from scipy.stats import entropy   # KL divergence helper
import numpy as np
import pandas as pd
import json


from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="Seed Number")


args = parser.parse_args()


RANDOM_SEED = args.seed
NB_EXAMPLES_PER_CAT = 200
NB_ALTERNATIVES_PER_SENTENCE = 10

DATASET_PATH = "/Users/clairedonnat/Downloads/arxiv-metadata-oai-snapshot.json"

def read_raw_datalines(dataset_path):
    with open(dataset_path, "r") as f:
        for line in f.readlines():
            yield line

async def main():
    for exp in range(3):
        RANDOM_SEED = args.seed + exp
        raw_lines = read_raw_datalines(DATASET_PATH)
        # --- 1. parse the raw dataset --------------------------------------
        titles: List[str]            = []
        abstracts: List[str]         = []
        primary_cats: List[str]      = []
        secondary_cats: List[List[str]] = []
        all_categories: List[List[str]] = []
        years: List[int]             = []
        category_counts: Counter     = Counter()

        for line in tqdm(raw_lines, desc="Parsing"):
            try:
                paper_info = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"⚠️  Bad JSON line skipped: {e}")
                continue

            # title & abstract -------------------------------------------------
            title = paper_info.get("title", "").strip()
            abstract = paper_info.get("abstract", "").strip()
            titles.append(title)
            abstracts.append(f"{title} {abstract}")

            # categories -------------------------------------------------------
            cats = paper_info.get("categories", [])

            # Some datasets store categories as a space‑separated string,
            # others as a list → normalise to a list of strings.
            if isinstance(cats, str):
                cats = cats.split()

            all_categories.append(cats)
            primary_cat = cats[0] if cats else "unknown"
            primary_cats.append(primary_cat)
            secondary_cats.append(cats[1:])
            category_counts.update(cats)   # Counter has an .update(iterable) helper

            # year -------------------------------------------------------------
            years.append(paper_info.get("year"))   # keep as str or cast to int

        # --- quick sanity check ---------------------------------------------
        print("Total papers:", len(titles))
        print("Distinct categories:", len(category_counts))
        print("Most common:", category_counts.most_common(10))


        #chosen_categories = (pd.Series(category_counts) / sum(category_counts.values())).sort_values(ascending=False).head(15).index
        chosen_categories = ["cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.NE" ,
                            "hep-ph", "hep-th", "quant-ph",
                            "stat.ML", "stat.CO", "stat.ME"]
        dataset_size = 0
        for cat in chosen_categories:
            print(f"{cat} | cnt={category_counts[cat]}")
            dataset_size += category_counts[cat]
        print(f"\n\ndataset_size={dataset_size}")
        np.random.seed(RANDOM_SEED)

        chosen_categories_counts = {cat: 0 for cat in chosen_categories}
        random_idx = np.random.permutation(len(titles))

        dataset = []
        y_additional = []
        y_true = []
        y_cat = []
        
        for cat in chosen_categories:
            #### sample docs from that category
            index = np.where(np.array(primary_cats) == cat)[0]
            chosen_docs = np.random.choice(index, size=NB_EXAMPLES_PER_CAT, replace=False)
            dataset += [ abstracts[i] for i in chosen_docs]
            y_true += [cat] * NB_EXAMPLES_PER_CAT
            y_additional += [secondary_cats[i] for i in chosen_docs]
            y_cat += [all_categories[i] for i in chosen_docs]


        ### enrich dataset

        per_doc, by_k = await hide_and_generate_async(dataset, n_alts_per_sentence=NB_ALTERNATIVES_PER_SENTENCE)

        #### save enriched dataset
        with open("dataset_enriched" + str(RANDOM_SEED) + ".json", "w") as f:
            json.dump(by_k, f)
        with open("new_sentences" + str(RANDOM_SEED) + ".json", "w") as f:
            json.dump(per_doc, f)
        with open("dataset_enriched_labels.json", "w") as f:
            json.dump({"y_true": y_true, "y_additional": y_additional, "y_cat": y_cat}, f)


        # one‑hot encoding --------------------------------------------------
        mlb   = MultiLabelBinarizer()
        Y     = mlb.fit_transform(y_cat)      # NumPy array (n_samples × n_classes)
        labels = mlb.classes_                         # ordered list of category names

        # wrap in a DataFrame for readability (optional)
        Y_df = pd.DataFrame(Y, columns=labels)
        print(Y_df.head())
        Y_df_sel = Y_df[chosen_categories]



        #######



        # ---------------------------------------------
        # 1) scikit‑learn’s LabelEncoder  (simple, fit once)
        # ---------------------------------------------
        le = LabelEncoder()
        y_int = le.fit_transform(y_true)     # y_true is a 1‑D array‑like of strings
        # Inverse mapping:  le.inverse_transform(y_int)

        # You can inspect the mapping
        label_map = dict(zip(le.classes_, range(len(le.classes_))))
        print(label_map)
        # y_true is an array‑like of strings such as "cs.CL", "stats.ML", ...

        y_coarse = np.array([re.split(r'[.\-]', x)[0] for x in y_true])
        # ---------------------------------------------
        # 1) scikit‑learn’s LabelEncoder  (simple, fit once)
        # ---------------------------------------------
        le_coarse = LabelEncoder()
        y_int_coarse = le_coarse.fit_transform(y_coarse)     # y_true is a 1‑D array‑like of strings
        # Inverse mapping:  le.inverse_transform(y_int)

        # You can inspect the mapping
        label_coarse_map = dict(zip(le_coarse.classes_, range(len(le_coarse.classes_))))
        print(label_coarse_map)

        Y_df_sel_np = Y_df_sel.to_numpy()
        Y_df_sel_np = np.diag(1/Y_df_sel_np.sum(axis=1)) @ Y_df_sel_np

        def row_normalise(mat, eps=1e-12):
            """Ensure each row sums to 1 and has no zeros."""
            mat = np.asarray(mat, dtype=np.float64)
            mat = mat + eps                                # avoid exact zeros
            mat = mat / mat.sum(axis=1, keepdims=True)     # → probability simplex
            return mat





        categories = np.unique(y_true)
        categories_coarse = np.unique(y_int_coarse)
        results = []
        for k in range(NB_ALTERNATIVES_PER_SENTENCE +1):
            print("Begin k = ", k)
            X_count_enriched, vectorizer, freq_enriched, col_sums  = convert_to_dtm(by_k[k])

            model = pLSI(precondition=True, solver="projector")
            model.fit(freq_enriched.toarray(), K = len(categories))
            # Get the topic-word distribution

            topic_word_dist = model.A_hat
            # Get the document-topic distribution
            doc_topic_dist = model.W_hat
            clusters = np.argmax(doc_topic_dist, axis=1)
            m_spoc, acc_spoc = align_on_confusion(y_int, clusters)
            print("Accuracy pLSI: ", acc_spoc)
            entropy_spoc = entropy(row_normalise(Y_df_sel_np), row_normalise(doc_topic_dist), axis=1).mean()
            print("Accuracy multi pLSI: ", entropy_spoc)

            lda_model = LatentDirichletAllocation(n_components=len(categories), random_state=42)
            lda_model.fit(X_count_enriched.toarray())
            lda_topic_matrix = lda_model.transform(X_count_enriched.toarray())
            clusters_lda = np.argmax(lda_topic_matrix, axis=1)
            m_lda, acc_lda = align_on_confusion(y_int, clusters_lda)
            print("Accuracy LDA: ", acc_lda)
            entropy_lda = entropy(row_normalise(Y_df_sel_np), row_normalise(lda_topic_matrix), axis=1).mean()
            print("Accuracy multi LDA: ",  entropy_lda)

            model = pLSI(precondition=True, solver="projector")
            model.fit(freq_enriched.toarray(), K = len(categories_coarse))
            # Get the topic-word distribution

            topic_word_dist = model.A_hat
            # Get the document-topic distribution
            doc_topic_dist = model.W_hat
            clusters = np.argmax(doc_topic_dist, axis=1)
            m_spoc, acc_spoc_coarse = align_on_confusion(y_int_coarse, clusters)
            print("Accuracy Coarse pLSI: ", acc_spoc_coarse)

            lda_model = LatentDirichletAllocation(n_components=len(categories_coarse), random_state=42)
            lda_model.fit(X_count_enriched.toarray())
            lda_topic_matrix = lda_model.transform(X_count_enriched.toarray())
            clusters_lda = np.argmax(lda_topic_matrix, axis=1)
            m_lda, acc_lda_coarse = align_on_confusion(y_int_coarse, clusters_lda)
            print("Accuracy Coarse LDA: ", acc_lda_coarse)
            results += [[k, RANDOM_SEED, NB_EXAMPLES_PER_CAT, acc_spoc, acc_lda, acc_spoc_coarse, acc_lda_coarse, entropy_spoc, entropy_lda]]
            # Save results to CSV
            results_df = pd.DataFrame(results, columns=["k", "seed", "n_samples", "acc_pLSI", "acc_LDA", "acc_pLSI_coarse", "acc_LDA_coarse", "entropy_PLSI",
                                                        "entropy_LDA"])
            results_df.to_csv("results" + str(RANDOM_SEED)  + ".csv", index=False)



            print("End k = ", k)

        
if __name__ == "__main__":
    asyncio.run(main())
