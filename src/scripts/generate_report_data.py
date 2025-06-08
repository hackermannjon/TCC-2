# src/scripts/generate_report_data.py

import sys
import json
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- CONFIGURAÇÃO E PATHS ---
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.models.datasets import GraphRecDataset
from src.models.graphrec import GraphRec

# Paths dos artefatos
FEATS_PATH = ROOT / "data/processed/combined_features.npy"
MODEL_CKPT_PATH = ROOT / "data/models/graphrec.pth"
TEST_LOGS_PATH = ROOT / "data/logs/interactions_test.csv"
PROFILES_CSV_PATH = ROOT / "data/processed/selected_profiles.csv"
PATH_PERSONA_CONSISTENT = ROOT / "data/processed/persona_consistent_history.json"
PATH_PERSONA_INCONSISTENT = ROOT / "data/processed/persona_inconsistent_history.json"

# Diretório de saída
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --- Funções de Avaliação (Reutilizadas e adaptadas) ---
def prepare_persona_history(persona_path: Path, features: np.ndarray, device: torch.device):
    with open(persona_path, "r") as f:
        history_indices = json.load(f)
    history_features = torch.from_numpy(features[history_indices]).float().to(device)
    history_opinions = torch.ones(len(history_indices), 1, device=device)
    item_hist_tensor = history_features.unsqueeze(0)
    opinions_tensor = opinions_tensor = history_opinions.unsqueeze(0)
    return item_hist_tensor, opinions_tensor, history_indices

@torch.no_grad()
def calculate_predictions(model, test_dataset, features_tensor, device, persona_data=None):
    y_true, y_prob, item_indices = [], [], []
    desc = "Calculando Predições" + (f" ({persona_data['name']})" if persona_data else " (Baseline)")
    
    for sample in tqdm(test_dataset, desc=desc):
        idx_tensor = sample["item_idx"].unsqueeze(0).to(device)
        item_indices.append(sample["item_idx"].item())
        y_true.append(sample["rating"].item())
        
        target_feats = features_tensor[idx_tensor]
        neigh_feats = features_tensor[sample["neigh_idxs"].unsqueeze(0).to(device)]
        
        item_hist, opinions = (persona_data["item_hist_tensor"], persona_data["opinions_tensor"]) if persona_data else (target_feats.unsqueeze(1), torch.tensor([[[0.5]]], device=device, dtype=torch.float))

        logit = model(item_hist, opinions, target_feats, neigh_feats)
        y_prob.append(torch.sigmoid(logit).item())
        
    return pd.DataFrame({"profile_id": item_indices, "true_label": y_true, "predicted_prob": y_prob})

# --- Funções de Geração de Outputs ---
def save_metrics_tables(scenarios_data, output_dir):
    md_string = "### Tabela 1: Métricas de Desempenho Comparativas\n\n"
    md_string += f"| {'Cenário':<25} | {'AUC':<6} | {'Acurácia':<9} | {'Precisão':<9} | {'Recall':<7} | {'F1-Score':<9} |\n"
    md_string += f"|{'-'*27}|{'-'*8}|{'-'*11}|{'-'*11}|{'-'*9}|{'-'*11}|\n"
    
    for name, df in scenarios_data.items():
        y_true, y_prob = df["true_label"], df["predicted_prob"]
        y_pred = (y_prob >= 0.5).astype(int)
        auc, acc, prec, rec, f1 = roc_auc_score(y_true, y_prob), accuracy_score(y_true, y_pred), precision_score(y_true, y_pred, zero_division=0), recall_score(y_true, y_pred, zero_division=0), f1_score(y_true, y_pred, zero_division=0)
        md_string += f"| {name:<25} | {auc:^6.3f} | {acc:^9.3f} | {prec:^9.3f} | {rec:^7.3f} | {f1:^9.3f} |\n"

    md_string += "\n\n### Tabela 2: Análise das Probabilidades de Saída\n\n"
    md_string += f"| {'Cenário':<25} | {'Prob. Média (Likes)':<22} | {'Prob. Média (Dislikes)':<25} |\n"
    md_string += f"|{'-'*27}|{'-'*24}|{'-'*27}|\n"
    
    for name, df in scenarios_data.items():
        mean_like_prob = df.loc[df['true_label'] == 1, 'predicted_prob'].mean()
        mean_dislike_prob = df.loc[df['true_label'] == 0, 'predicted_prob'].mean()
        md_string += f"| {name:<25} | {mean_like_prob:^22.3f} | {mean_dislike_prob:^25.3f} |\n"

    with open(output_dir / "report_tables.md", "w", encoding="utf-8") as f:
        f.write(md_string)
    print(f"[OK] Tabelas de métricas salvas em: {output_dir / 'report_tables.md'}")

def save_probability_distribution_plot(scenarios_data, output_dir):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, len(scenarios_data), figsize=(18, 5), sharey=True)
    fig.suptitle('Distribuição das Probabilidades de Predição por Cenário', fontsize=16)

    for i, (name, df) in enumerate(scenarios_data.items()):
        ax = axes[i]
        sns.kdeplot(data=df, x='predicted_prob', hue='true_label', fill=True, ax=ax, palette={0: 'red', 1: 'green'})
        ax.set_title(name)
        ax.set_xlabel('Probabilidade de Like')
        ax.set_ylabel('Densidade')
        ax.legend(title='Rótulo Real', labels=['Dislike', 'Like'])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = output_dir / "probability_distributions.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"[OK] Gráfico de distribuição de probabilidade salvo em: {plot_path}")

def save_tsne_visualization(features, personas_indices, output_dir):
    print("[INFO] Calculando visualização t-SNE (pode demorar)...")
    n_samples = features.shape[0]
    # Usar um subconjunto para t-SNE se for muito lento, mas 2000 é factível
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    df_tsne = pd.DataFrame(features_2d, columns=['x', 'y'])
    df_tsne['persona'] = 'Nenhum'
    df_tsne.loc[personas_indices['Consistente'], 'persona'] = 'Consistente'
    df_tsne.loc[personas_indices['Inconsistente'], 'persona'] = 'Inconsistente'
    
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        data=df_tsne[df_tsne['persona'] == 'Nenhum'],
        x='x', y='y', color='lightgray', alpha=0.5, label='Outros Perfis'
    )
    sns.scatterplot(
        data=df_tsne[df_tsne['persona'] != 'Nenhum'],
        x='x', y='y', hue='persona', palette={'Consistente': 'blue', 'Inconsistente': 'red'}, s=50,
    )
    plt.title('Visualização t-SNE do Espaço de Features e Personas')
    plt.xlabel('Componente t-SNE 1')
    plt.ylabel('Componente t-SNE 2')
    plt.legend(title='Tipo de Persona')
    plt.tight_layout()
    plot_path = output_dir / "embedding_space_personas.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"[OK] Gráfico t-SNE salvo em: {plot_path}")

# --- FUNÇÃO PRINCIPAL ---
def main():
    print("Gerando todos os dados e visualizações para o relatório final...")
    device = torch.device("cpu")
    
    # Carregamentos
    features_np = np.load(FEATS_PATH)
    model = GraphRec(d_item=features_np.shape[1]).to(device)
    model.load_state_dict(torch.load(MODEL_CKPT_PATH, map_location=device))
    model.eval()
    features_tensor = torch.from_numpy(features_np).float().to(device)
    test_dataset = GraphRecDataset(FEATS_PATH, TEST_LOGS_PATH, k_social=10)

    # Prepara Personas
    personas_data = {}
    personas_indices = {}
    for name, path in [("Consistente", PATH_PERSONA_CONSISTENT), ("Inconsistente", PATH_PERSONA_INCONSISTENT)]:
        hist_tensor, opinions_tensor, indices = prepare_persona_history(path, features_np, device)
        personas_data[f"Persona: {name}"] = {"name": name, "item_hist_tensor": hist_tensor, "opinions_tensor": opinions_tensor}
        personas_indices[name] = indices

    # Calcula predições
    scenarios_results = {}
    scenarios_results["Baseline (Sem Histórico)"] = calculate_predictions(model, test_dataset, features_tensor, device)
    for name, p_data in personas_data.items():
        scenarios_results[name] = calculate_predictions(model, test_dataset, features_tensor, device, persona_data=p_data)

    # Salva todos os outputs
    for name, df in scenarios_results.items():
        clean_name = name.lower().replace(" ", "_").replace(":", "").replace("(", "").replace(")", "")
        path = OUTPUT_DIR / f"predictions_{clean_name}.csv"
        df.to_csv(path, index=False)
        print(f"[OK] Predições para '{name}' salvas em: {path}")

    save_metrics_tables(scenarios_results, OUTPUT_DIR)
    save_probability_distribution_plot(scenarios_results, OUTPUT_DIR)
    save_tsne_visualization(features_np, personas_indices, OUTPUT_DIR)
    
    print("\n[SUCESSO] Todos os outputs para o relatório foram gerados na pasta 'outputs/'.")

if __name__ == "__main__":
    main()