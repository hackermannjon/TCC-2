# src/scripts/run_all.py

import sys
import subprocess
import locale
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# Força UTF-8 para o stdout e o ambiente, conforme solicitado
try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8') # Usar 'en_US.UTF-8' ou 'C.UTF-8' para compatibilidade
except locale.Error:
    print("[AVISO] Nao foi possivel definir o locale para UTF-8. Continuando com a configuracao padrao.")


# ==============================================================================
# FUNÇÕES AUXILIARES DE EXECUÇÃO E RELATÓRIO
# ==============================================================================

def run_command(cmd_list: list[str], cwd: Path):
    """
    Executa um comando no shell, força o ambiente para UTF-8,
    e retorna True se sucesso, False caso contrário.
    """
    cmd_str = ' '.join(cmd_list)
    print(f"\n>>> Executando: {cmd_str}")
    start_time = time.time()
    
    try:
        # Configura o ambiente para usar UTF-8
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        # Executa o processo
        process = subprocess.run([sys.executable, "-m", *cmd_list],
                                 cwd=cwd, capture_output=True, text=True, 
                                 check=True, encoding='utf-8', env=env)
        return True, process.stdout, process.stderr
        
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr


def print_npy_info(path: Path, root_dir: Path):
    """
    Carrega um arquivo .npy e imprime seu nome relativo e shape.
    """
    try:
        arr = np.load(path)
        print(f"  - {path.relative_to(root_dir)} -> shape: {arr.shape}")
    except Exception as e:
        print(f"  - {path.relative_to(root_dir)} -> erro ao carregar: {e}")

def print_csv_info(path: Path, root_dir: Path, n_rows=3):
    """
    Lê um CSV e imprime número de linhas, colunas e as primeiras n_rows.
    """
    try:
        df = pd.read_csv(path)
        print(f"\n  --- CSV Info: {path.relative_to(root_dir)} ---")
        print(f"  Shape: {df.shape[0]} linhas, {df.shape[1]} colunas")
        print(f"  Colunas: {list(df.columns)}")
        print("  Primeiras linhas:")
        print(df.head(n_rows).to_string())
    except Exception as e:
        print(f"  - {path.relative_to(root_dir)} -> erro ao ler: {e}")


# ==============================================================================
# FUNÇÃO PRINCIPAL DO PIPELINE
# ==============================================================================

def main():
    ROOT = Path(__file__).resolve().parents[2]
    os.chdir(ROOT)  # Garante que caminhos relativos funcionem a partir da raiz

    print("=== INICIANDO PIPELINE COMPLETO: PROCESSAMENTO, TREINO E AVALIAÇÃO ===")

    # --- 0. Verificação de Pré-requisitos ---
    print("\n\n=== ETAPA 0: Verificando dados brutos e pré-requisitos ===")
    raw_okcupid_csv = ROOT / "data/raw/okcupid_profiles.csv"
    raw_images_dir = ROOT / "data/raw/images/ProfilesDataSet"
    interactions_train_csv = ROOT / "data/logs/interactions_train.csv"
    
    essential_files = [raw_okcupid_csv, raw_images_dir, interactions_train_csv]
    is_ready = True
    for path in essential_files:
        if not path.exists():
            print(f"[ERRO] ERRO CRÍTICO: Arquivo/diretório essencial não encontrado: {path}")
            is_ready = False
    
    if not is_ready:
        print("\n[ABORTADO] Pipeline abortado. Garanta que todos os dados brutos e o split de treino existam.")
        return
    print("[OK] Pré-requisitos verificados com sucesso.")

    # --- Definição das Etapas do Pipeline ---
    # MODIFICADO: Atualizado para incluir a criação de personas e a avaliação comparativa.
    steps = [
        # Etapa 1: Processamento de Dados
        {"title": "1/8: Gerando Features Multimodais", "module": "src.models.features"},
        {"title": "2/8: Gerando Embeddings Sociais", "module": "src.models.social_graph"},
        {"title": "3/8: Combinando Features Finais", "module": "src.models.combine_features"},
        # Etapa 2: Geração de Personas
        {"title": "4/8: Criando Personas de Usuario", "module": "src.scripts.create_personas"},
        # Etapa 3: Treinamento
        {"title": "5/8: Treinando Modelo GraphRec", "module": "src.models.train_graphrec"},
        # Etapa 4: Avaliação Comparativa
        {"title": "6/8: Avaliando (Baseline vs Personas)", "module": "src.scripts.evaluate_comparison"},
        # Etapa 5: Testes de Sanidade
        {"title": "7/8: Testando Preferencias", "module": "src.scripts.test_preference"},
        {"title": "8/8: Testando Ranking", "module": "src.scripts.test_ranking"},
    ]

    print("\n\n=== INICIANDO EXECUÇÃO DAS ETAPAS DO PIPELINE ===")
    
    with tqdm(total=len(steps), desc="Progresso Geral do Pipeline", unit="etapa") as pbar:
        for step in steps:
            pbar.set_description(f"Executando {step['title']}")
            
            start_time = time.time()
            success, stdout, stderr = run_command([step['module']], cwd=ROOT)
            duration = time.time() - start_time
            
            # Adiciona uma borda para separar visualmente a saída de cada etapa
            print("-" * 60)
            if stdout.strip():
                print(stdout.strip())
            if stderr.strip():
                print("\n[STDERR - Warnings ou Infos Adicionais]:")
                print(stderr.strip())
            print("-" * 60)
            
            if not success:
                print(f"\n[ERRO] Falha na etapa '{step['title']}' (duração: {duration:.2f}s)")
                print("[ABORTADO] Pipeline abortado.")
                return
            
            print(f"[OK] Etapa '{step['title']}' concluída com sucesso! (duração: {duration:.2f}s)\n")
            pbar.update(1)

    # --- ETAPA FINAL: Relatório de Artefatos Gerados ---
    print("\n\n=== ETAPA FINAL: Relatório de Artefatos Gerados ===")
    
    processed_dir = ROOT / "data/processed"
    models_dir = ROOT / "data/models"
    logs_dir = ROOT / "data/logs"

    print("\n--- Conteúdo dos diretórios de saída ---")
    for dir_path in [processed_dir, models_dir, logs_dir]:
        print(f"[DIR] {dir_path.relative_to(ROOT)}:")
        contents = sorted([p.name for p in dir_path.glob('*') if p.is_file()])
        if contents:
            for item in contents:
                print(f"  - {item}")
        else:
            print("  (vazio ou apenas subdiretórios)")

    print("\n\n--- Detalhes dos artefatos ---")
    
    print("\n[INFO] Arquivos .npy em data/processed:")
    for npy_file in processed_dir.glob("*.npy"):
        print_npy_info(npy_file, ROOT)

    print("\n[INFO] Arquivos .csv em data/processed e data/logs:")
    for csv_file in list(processed_dir.glob("*.csv")) + list(logs_dir.glob("*.csv")):
        print_csv_info(csv_file, ROOT)

    print("\n\n=== PIPELINE COMPLETO EXECUTADO COM SUCESSO! ===")


if __name__ == "__main__":
    main()