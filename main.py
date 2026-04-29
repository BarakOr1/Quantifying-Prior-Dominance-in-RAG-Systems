import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import random
import re
from faker import Faker
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from tqdm.auto import tqdm
from IPython.display import display, clear_output
import warnings

# ==========================================

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

RUN_MODE = "FULL"
SAMPLES_PER_DATASET = 1000

MODELS_TO_TEST = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct-AWQ",
    "gpt-4o-mini"
]

MODEL_ORDER = ["Qwen2.5-1.5B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-72B-Instruct-AWQ", "gpt-4o-mini"]
PALETTE_MODELS = {
    "Qwen2.5-1.5B-Instruct": "#3498db", 
    "Qwen2.5-7B-Instruct": "#2ecc71", 
    "Qwen2.5-72B-Instruct-AWQ": "#f39c12",
    "gpt-4o-mini": "#e74c3c"
}
PALETTE_DATASETS = {"HotpotQA": "#9b59b6", "NQ": "#f1c40f", "TriviaQA": "#e67e22"}
PALETTE_CONDITIONS = {"acc_zero": "#bdc3c7", "acc_oracle": "#2980b9", "acc_noise": "#c0392b"}

from google.colab import drive
drive.mount('/content/drive')
BASE_DIR = f'/content/drive/MyDrive/RAG_Research_GoldenRun_{RUN_MODE}'
os.makedirs(BASE_DIR, exist_ok=True)

FORCE_FRESH_START = False
# ==========================================
class AcademicContextEngine:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.fake = Faker()
        Faker.seed(42)
        random.seed(42)

    def _generate_plausible_date(self, original_date_str):
        years = re.findall(r'\b(1[0-9]{3}|20[0-2][0-9])\b', original_date_str)
        if years:
            orig_year = int(years[0])
            fake_year = orig_year + random.choice([-1, 1]) * random.randint(1, 15)
            return original_date_str.replace(str(orig_year), str(fake_year))
        return self.fake.date_between(start_date='-50y', end_date='today').strftime("%B %d, %Y")

    def _generate_plausible_number(self, original_num_str):
        numbers = re.findall(r'\d+', original_num_str.replace(',', ''))
        if numbers:
            orig_num = int(numbers[0])
            if orig_num == 0: return "1"
            variation = orig_num * random.uniform(0.1, 0.5)
            fake_num = int(orig_num + random.choice([-1, 1]) * variation)
            return f"{max(1, fake_num):,}"
        return str(random.randint(2, 999))

    def create_conflict_context(self, context, answer):
        doc = self.nlp(answer)
        fake_val = "[UNVERIFIED]"
        if doc.ents:
            label = doc.ents[0].label_
            for _ in range(5):
                if label == "PERSON": candidate = self.fake.name()
                elif label in ["GPE", "LOC"]: candidate = self.fake.city() if random.random() > 0.5 else self.fake.country()
                elif label == "ORG": candidate = self.fake.company()
                elif label == "DATE": candidate = self._generate_plausible_date(answer)
                elif label in ["MONEY", "CARDINAL", "QUANTITY"]: candidate = str(random.randint(2, 999))
                else: candidate = self.fake.word().capitalize()

                if candidate.lower() not in answer.lower() and answer.lower() not in candidate.lower():
                    fake_val = candidate
                    break
        else:
            fake_val = self.fake.word().capitalize() if answer and answer[0].isupper() else self.fake.word()

        return context.replace(answer, fake_val)

    def create_noisy_context(self, oracle_context):
        noise_before = " ".join([self.fake.paragraph() for _ in range(2)])
        noise_after = " ".join([self.fake.paragraph() for _ in range(2)])
        return f"{noise_before} {oracle_context} {noise_after}"

context_engine = AcademicContextEngine()

# ==========================================
def load_all_data(limit):
    datasets_info = {
        "HotpotQA": ("hotpot_qa", "distractor"),
        "NQ": ("nq_open", None),
        "TriviaQA": ("trivia_qa", "rc.wikipedia")
    }

    final_df = []
    for name, config in datasets_info.items():
        print(f"Loading Dataset: {name}...")
        ds = load_dataset(config[0], config[1], split="validation") if config[1] else load_dataset(config[0], split="validation")

        if str(limit).lower() != "all":
            ds = ds.shuffle(seed=42).select(range(min(limit, len(ds))))

        for item in ds:
            if name == "HotpotQA":
                q, a = item['question'], item['answer']
                c = " ".join([" ".join(s) for t, s in zip(item['context']['title'], item['context']['sentences']) if t in item['supporting_facts']['title']])
            elif name == "NQ":
                q, a = item['question'], item['answer'][0]
                c = f"Fact: {q} The confirmed answer is {a}."
            else:
                q, a = item['question'], item['answer']['value']
                c = f"Data point: {q} Evidence suggests {a}."

            final_df.append({"q": q, "a": a, "c_star": c, "source": name})

    return pd.DataFrame(final_df)

# ==========================================
def get_robust_logprob(context, question, answer, model_id, model=None, tokenizer=None):
    messages = [{"role": "system", "content": "You are a highly precise QA system. Answer concisely with ONLY the requested entity, fact, or short phrase. No pleasantries, no full sentences."}]
    if context:
        messages.append({"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"})
    else:
        messages.append({"role": "user", "content": f"Question: {question}"})

    if "gpt" in model_id.lower():
        max_retries = 3 
        base_wait_time = 2

        for attempt in range(max_retries):
            try:
                res = client.chat.completions.create(
                    model=model_id, messages=messages, logprobs=True, top_logprobs=1, max_tokens=5, temperature=0.0
                )
                lps = [lp.logprob for lp in res.choices[0].logprobs.content if lp.token.strip() in answer]
                # תיקון קריטי: נרמול טוקנים (ממוצע)
                return sum(lps) / len(lps) if len(lps) > 0 else -15.0
            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "rate limit" in error_msg or "quota" in error_msg:
                    print(f"\n[API Quota Error] You have hit your OpenAI limit! Stopping GPT execution.")
                    return "QUOTA_ERROR"
                time.sleep(base_wait_time * (2 ** attempt))
        return -15.0

    else:
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to("cuda")
        full_text = prompt_text + answer
        full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"].to("cuda")
        prompt_len = prompt_ids.shape[1]
        target_ids = full_ids[0, prompt_len:]

        if len(target_ids) == 0: return -15.0

        with torch.no_grad():
            outputs = model(full_ids)
            logits = outputs.logits
            lps = torch.nn.functional.log_softmax(logits, dim=-1)

        ans_logprobs = []
        for i in range(len(target_ids)):
            ans_logprobs.append(lps[0, prompt_len + i - 1, target_ids[i]].item())
            
        return sum(ans_logprobs) / len(ans_logprobs) if len(ans_logprobs) > 0 else -15.0
# ==========================================
def update_master_dashboard(df_results, mode):
    clear_output(wait=True)
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    available_models = [m for m in MODEL_ORDER if m in df_results['model'].unique()]

    gap_df = df_results.melt(id_vars=['model'], value_vars=['acc_zero', 'acc_oracle', 'acc_noise'],
                             var_name='Condition', value_name='Accuracy')
    sns.barplot(data=gap_df, x='model', y='Accuracy', hue='Condition', ax=axes[0,0],
                palette=PALETTE_CONDITIONS, order=available_models)
    axes[0,0].set_title("The Utilization Gap & Noise Drop", fontweight='bold')
    axes[0,0].set_ylim(0, 1.1)

    sns.barplot(data=df_results, x='model', y='ncu', hue='source', ax=axes[0,1],
                palette=PALETTE_DATASETS, order=available_models)
    axes[0,1].set_title("Context Utilization Score (NCU) by Dataset", fontweight='bold')
    axes[0,1].set_ylim(0, 1.1)
    axes[0,1].legend(loc='upper right')

    conflict_survival = df_results.groupby('model')['acc_conflict'].mean().reindex(available_models) * 100
    sns.barplot(x=conflict_survival.index, y=conflict_survival.values, ax=axes[1,0],
                palette=PALETTE_MODELS, edgecolor='black')
    axes[1,0].set_title("Context Obedience (Ignored Fake Context %)", fontweight='bold')
    axes[1,0].set_ylabel("Survival Rate (%)")
    axes[1,0].set_ylim(0, 100)

    sns.violinplot(data=df_results, x='model', y='ncu', ax=axes[1,1],
                   palette=PALETTE_MODELS, order=available_models, inner='quartile')
    axes[1,1].set_title("NCU Distribution Profile (Violin Plot)", fontweight='bold')
    axes[1,1].set_ylabel("NCU Score")
    axes[1,1].set_ylim(-0.2, 1.2)

    avg_latency = df_results.groupby('model')['latency'].mean().round(2)
    kpi_text = " | ".join([f"{idx}: {val}s/q" for idx, val in avg_latency.items() if idx in avg_latency])

    plt.suptitle(f"[{mode} MODE] Resume Dashboard | Samples: {len(df_results)} | KPI: {kpi_text}", fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ==========================================
def run_resume_research():
    print(f"🚀 STARTING Research Run (Mode: {RUN_MODE})...")
    data = load_all_data(SAMPLES_PER_DATASET)
    checkpoint_file = f"{BASE_DIR}/golden_results_checkpoint.csv"

    if FORCE_FRESH_START and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("🗑️ FORCE_FRESH_START is True: Deleted old checkpoint. Starting entirely from scratch.")

    if os.path.exists(checkpoint_file):
        df_existing = pd.read_csv(checkpoint_file)
        master_results = df_existing.to_dict('records')
        print(f"✅ Loaded checkpoint with {len(master_results)} processed rows.")
    else:
        master_results = []
        df_existing = pd.DataFrame()
        print("⚠️ No checkpoint found. Starting fresh.")

    for mid in MODELS_TO_TEST:
        model, tokenizer = None, None
        display_name = mid.split("/")[-1]

        if not df_existing.empty and 'model' in df_existing.columns:
            completed_count = len(df_existing[df_existing['model'] == display_name])
        else:
            completed_count = 0

        print(f"\n{'='*50}\nEvaluating Model: {display_name} (Already done: {completed_count}/{len(data)})\n{'='*50}")

        if completed_count >= len(data):
            print(f"⏭️ Skipping {display_name} - already fully processed in previous run.")
            continue

        if "gpt" not in mid.lower():
            tokenizer = AutoTokenizer.from_pretrained(mid)
            # AWQ models and standard models will load automatically correctly with device_map
            model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float16, device_map={"": 0})

        quota_error = False

        for i, row in tqdm(data.iterrows(), total=len(data), desc=f"Processing {display_name}"):
            if i < completed_count:
                continue  

            q, a, c_star, source = row['q'], row['a'], row['c_star'], row['source']
            c_adv = context_engine.create_conflict_context(c_star, a)
            c_noise = context_engine.create_noisy_context(c_star)

            contexts = {"zero": None, "oracle": c_star, "conflict": c_adv, "noise": c_noise}
            lps = {}
            start_time = time.time()

            for key, ctx in contexts.items():
                res = get_robust_logprob(ctx, q, a, mid, model, tokenizer)
                if res == "QUOTA_ERROR":
                    quota_error = True
                    break
                lps[key] = res

            if quota_error:
                break 

            latency = time.time() - start_time
            cus = lps["oracle"] - lps["zero"]
            entropy = -lps["zero"] if lps["zero"] < 0 else 1e-5
            prob_z = np.exp(lps["zero"])
            
            raw_ncu = cus / entropy
            bounded_ncu = max(0, min(1, raw_ncu))

            master_results.append({
                'model': display_name, 'source': source, 'prob_zero': prob_z,
                'acc_zero': 1 if prob_z > 0.05 else 0, 'acc_oracle': 1 if np.exp(lps["oracle"]) > 0.05 else 0,
                'acc_conflict': 1 if np.exp(lps["conflict"]) > 0.05 else 0, 'acc_noise': 1 if np.exp(lps["noise"]) > 0.05 else 0,
                'ncu': bounded_ncu, 'raw_ncu': raw_ncu, 'latency': latency
            })

            if (i + 1) % 50 == 0:
                res_df = pd.DataFrame(master_results)
                res_df.to_csv(checkpoint_file, index=False)
                update_master_dashboard(res_df, RUN_MODE)

        if model:
            del model
            del tokenizer
            torch.cuda.empty_cache()

    final_df = pd.DataFrame(master_results)
    final_df.to_csv(f"{BASE_DIR}/final_golden_paper_results.csv", index=False)
    update_master_dashboard(final_df, RUN_MODE)

    print("\n" + "="*70)
    print(" CURRENT SUMMARY TABLE")
    available_models = [m for m in MODEL_ORDER if m in final_df['model'].unique()]
    summary_table = final_df.groupby('model')[['acc_zero', 'acc_oracle', 'acc_noise', 'acc_conflict', 'ncu', 'raw_ncu', 'latency']].mean().round(3).reindex(available_models)
    display(summary_table)

run_resume_research()
