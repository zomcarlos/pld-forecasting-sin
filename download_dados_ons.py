import urllib.request
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DADOS_DIR = BASE_DIR / "dados"

def download_files():
    # Estruturas de pastas
    ear_dir = DADOS_DIR / "EAR"
    carga_dir = DADOS_DIR / "CARGA"
    ena_dir = DADOS_DIR / "ENA"
    
    ear_dir.mkdir(parents=True, exist_ok=True)
    carga_dir.mkdir(parents=True, exist_ok=True)
    ena_dir.mkdir(parents=True, exist_ok=True)

    # Considerando histórico a partir de 2018
    anos = range(2018, 2027)
    
    # Dicionário de datasets para simplificar o loop
    datasets = {
        "EAR": {
            "dir": ear_dir,
            "url_base": "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/ear_subsistema_di/EAR_DIARIO_SUBSISTEMA_{ano}.csv",
            "file_prefix": "EAR_DIARIO_SUBSISTEMA"
        },
        "CARGA": {
            "dir": carga_dir,
            "url_base": "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/carga_energia_di/CARGA_ENERGIA_{ano}.csv",
            "file_prefix": "CARGA_ENERGIA"
        },
        "ENA": {
            "dir": ena_dir,
            "url_base": "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/ena_subsistema_di/ENA_DIARIO_SUBSISTEMA_{ano}.csv",
            "file_prefix": "ENA_DIARIO_SUBSISTEMA"
        }
    }

    for name, config in datasets.items():
        print(f"\nIniciando download dos dados de {name}...")
        for ano in anos:
            url = config["url_base"].format(ano=ano)
            dest = config["dir"] / f"{config['file_prefix']}_{ano}.csv"
            if not dest.exists():
                print(f"  Baixando {name} {ano}...")
                try:
                    urllib.request.urlretrieve(url, dest)
                except Exception as e:
                    print(f"    Erro ao baixar {name} {ano}: {e}")
            else:
                print(f"  {name} {ano} já existe.")

    print("\nDownload concluído! Lembre-se de adicionar os arquivos do PLD (CCEE) na pasta dados/PLD/ caso deseje alinhar as datas.")

if __name__ == "__main__":
    download_files()
