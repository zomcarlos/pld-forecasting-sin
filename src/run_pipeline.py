"""
Pipeline Central de Execução v4.

Executa todos os modelos e diagnósticos estatísticos de forma sequencial
para garantir a reprodutibilidade integral do experimento.
"""
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

# Ordem de execução: Diagnósticos primeiro, baselines, ML, e Deep Learning.
SCRIPTS = [
    "diagnostico_estatistico.py",
    "sarimax_diario.py",
    "tar_diario.py",
    "ridge_diario.py",
    "rf_diario.py",
    "ml_diario.py",  # LightGBM
    "lstm_diario.py",
    "gru_diario.py",
    # Análises comparativas
    "metricas_por_fold.py",
    "diebold_mariano.py",
    "diagnostico_avancado.py",
    "intervalos_previsao.py",
]

def main():
    print("="*60)
    print(" INICIANDO PIPELINE CENTRAL DE PREVISÃO DO PLD")
    print("="*60)
    
    start_time_total = time.time()

    for script in SCRIPTS:
        script_path = SRC_DIR / script
        if not script_path.exists():
            print(f"[AVISO] Script {script} não encontrado em src/. Pulando...")
            continue
        
        print(f"\n>>> Executando {script} ...")
        start_time = time.time()
        
        # O uso de subprocess garante que cada modelo rode em um processo isolado.
        # Isso previne vazamentos de memória e conflitos globais (ex: sessões de 
        # frameworks de Deep Learning ou instâncias do Matplotlib).
        result = subprocess.run(
            [sys.executable, script_path.name],
            cwd=SRC_DIR,
            capture_output=False
        )
        
        elapsed = time.time() - start_time
        if result.returncode == 0:
            print(f">>> {script} finalizado com SUCESSO em {elapsed:.1f} segundos.")
        else:
            print(f">>> [ERRO CRÍTICO] Falha ao executar {script}. Código: {result.returncode}")
            sys.exit(result.returncode)

    elapsed_total = time.time() - start_time_total
    print("\n" + "="*60)
    print(f" PIPELINE CONCLUÍDO COM SUCESSO! Tempo total: {elapsed_total:.1f} segundos.")
    print(f" Todos os artefatos visuais e métricas estão salvos em: {BASE_DIR / 'output'}")
    print("="*60)

if __name__ == "__main__":
    main()
