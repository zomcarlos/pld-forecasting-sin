import json
import ssl
import urllib.request
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

    # Download do PLD_HORARIO (CCEE)
    pld_dir = DADOS_DIR / "PLD"
    pld_dir.mkdir(parents=True, exist_ok=True)
    print("\nIniciando download dos dados do PLD (CCEE)...")

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'pt-BR,pt;q=0.8,en-US;q=0.5,en;q=0.3',
    }

    try:
        req = urllib.request.Request(
            'https://dadosabertos.ccee.org.br/api/3/action/package_show?id=pld_horario',
            headers=headers
        )
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            resources = data.get('result', {}).get('resources', [])

        for res in resources:
            name = res.get('name', '')
            url = res.get('url', '')
            if name.startswith('pld_horario_'):
                dest = pld_dir / f"{name}.csv"
                if not dest.exists():
                    print(f"  Baixando PLD {name}...")
                    try:
                        r_file = urllib.request.Request(url, headers=headers)
                        with urllib.request.urlopen(r_file, context=ctx) as r_resp, open(dest, 'wb') as out:
                            out.write(r_resp.read())
                    except Exception as e:
                        print(f"    Erro ao baixar {name}: {e}")
                else:
                    print(f"  PLD {name} já existe.")
    except Exception as e:
        print(f"  Erro ao consultar API da CCEE: {e}")

    print("\nDownload de todos os dados concluído com sucesso!")

if __name__ == "__main__":
    download_files()

