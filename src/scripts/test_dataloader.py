# src/scripts/test_dataloader.py

import sys
import logging
from pathlib import Path
from tqdm import tqdm

# Entra na raiz do projeto para imports absolutos e paths corretos
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.models.datasets import GraphRecDataset

# Paths absolutos
FEATURES_PATH      = ROOT / "data" / "processed" / "combined_features.npy"
INTERACTIONS_PATH  = ROOT / "data" / "logs" / "interactions.csv"
LOG_FILE           = ROOT / "data" / "logs" / "test_dataloader.log"

# Configura logger para arquivo
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def main():
    logging.info("=== Iniciando teste do GraphRecDataset ===")

    # Carrega dataset
    ds = GraphRecDataset(
        features_path=FEATURES_PATH,
        interactions_path=INTERACTIONS_PATH,
        k_social=10
    )

    total = len(ds)
    logging.info(f"Total de interações no dataset: {total}")
    print(f"Total de interações: {total}")

    # Visualizar primeiras amostras
    n_preview = min(5, total)
    logging.info(f"Exibindo {n_preview} primeiras amostras...")
    print(f"\nPrimeiras {n_preview} amostras:")
    for i in range(n_preview):
        sample = ds[i]
        logging.info(f"Amostra {i}: {sample}")
        print(f"Amostra {i}: {sample}")

    # Iterar sobre todo o dataset com barra de progresso
    logging.info("Iterando sobre todas as amostras com tqdm...")
    for _ in tqdm(range(total), desc="Carregando amostras", unit="amostra"):
        _ = ds[_]  # acessa cada amostra

    logging.info("Teste do dataset concluído com sucesso.")
    print("\nTeste concluído. Verifique o log em:", LOG_FILE)

if __name__ == '__main__':
    main()
