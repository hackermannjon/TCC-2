# src/scripts/split_interactions.py

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def main():
    ROOT = Path(__file__).resolve().parents[2]
    LOGS_ORIG = ROOT / "data" / "logs" / "interactions.csv"
    TRAIN_OUT = ROOT / "data" / "logs" / "interactions_train.csv"
    TEST_OUT  = ROOT / "data" / "logs" / "interactions_test.csv"

    # Carrega o CSV completo de interações
    df = pd.read_csv(LOGS_ORIG)

    # Shuffle e split 80% treino / 20% teste
    train, test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

    # Salva nos arquivos separados
    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_OUT, index=False)
    test.to_csv(TEST_OUT, index=False)

    print(f"[OK] Divisão concluída: {len(train)} linhas para TREINO, {len(test)} linhas para TESTE.")
    print(f"→ TREINO salvo em: {TRAIN_OUT}")
    print(f"→ TESTE  salvo em: {TEST_OUT}")

if __name__ == "__main__":
    main()
