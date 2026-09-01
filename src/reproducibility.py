import json
import sys
import platform
import datetime
from pathlib import Path

def save_reproducibility_report(model_name, global_params, model_params, metrics, output_dir=None):
    """
    Gera e salva relatórios de reprodutibilidade (JSON e Markdown) contendo informações do sistema,
    versões de pacotes, parâmetros globais do script, parâmetros específicos do modelo e métricas resultantes.
    """
    timestamp = datetime.datetime.now().isoformat()
    
    # 1. Informações de ambiente
    env_info = {
        "timestamp": timestamp,
        "python_version": sys.version.split('\n')[0],
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
    }
    
    # 2. Versões de pacotes python relevantes
    packages = ["numpy", "pandas", "sklearn", "statsmodels", "pmdarima", "lightgbm", "torch"]
    package_versions = {}
    for pkg in packages:
        try:
            import importlib
            mod = importlib.import_module(pkg)
            package_versions[pkg] = getattr(mod, "__version__", "desconhecido")
        except ImportError:
            package_versions[pkg] = "não instalado"
            
    # 3. Construção do relatório estruturado
    report = {
        "model_name": model_name,
        "environment": env_info,
        "packages": package_versions,
        "global_parameters": global_params,
        "model_parameters": model_params,
        "metrics": metrics
    }
    
    # 4. Caminho de saída
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "output" / "reprodutibilidade"
    else:
        output_dir = Path(output_dir)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp_str = datetime.datetime.fromisoformat(timestamp).strftime("%Y%m%d_%H%M%S")
    
    # Salva arquivos JSON (reprodutibilidade de máquina)
    json_latest = output_dir / f"{model_name}_params_latest.json"
    json_ts = output_dir / f"{model_name}_params_{timestamp_str}.json"
    
    with open(json_latest, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    with open(json_ts, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
        
    # Gera conteúdo Markdown (relatório legível academicamente)
    md_content = []
    md_content.append(f"# Relatório de Reprodutibilidade e Parâmetros: {model_name}")
    md_content.append(f"Gerado automaticamente em: `{timestamp}`\n")
    
    md_content.append("## 1. Ambiente do Sistema")
    md_content.append(f"- **Sistema Operacional:** {env_info['os']} (Release: {env_info['os_release']})")
    md_content.append(f"- **Python:** `{env_info['python_version']}`\n")
    
    md_content.append("## 2. Dependências e Versões das Bibliotecas")
    for pkg, version in package_versions.items():
        md_content.append(f"- **{pkg}:** `{version}`")
    md_content.append("")
    
    md_content.append("## 3. Parâmetros Globais de Execução")
    for k, v in global_params.items():
        md_content.append(f"- **{k}:** `{v}`")
    md_content.append("")
    
    md_content.append("## 4. Parâmetros Específicos do Modelo / Testes")
    for k, v in model_params.items():
        if isinstance(v, dict):
            md_content.append(f"- **{k}:**")
            for sub_k, sub_v in v.items():
                md_content.append(f"  - *{sub_k}:* `{sub_v}`")
        else:
            md_content.append(f"- **{k}:** `{v}`")
    md_content.append("")
    
    md_content.append("## 5. Resultados e Métricas Obtidas")
    if isinstance(metrics, list):
        md_content.append("| Submercado | MAE | RMSE | MAPE |")
        md_content.append("| --- | --- | --- | --- |")
        for m in metrics:
            # Trata métricas que podem variar de formato
            sub = m.get("submercado", "N/A")
            mae = m.get("mae", "N/A")
            rmse = m.get("rmse", "N/A")
            mape_val = m.get("mape", "N/A")
            
            mae_str = f"{mae:.4f}" if isinstance(mae, (int, float)) else str(mae)
            rmse_str = f"{rmse:.4f}" if isinstance(rmse, (int, float)) else str(rmse)
            mape_str = f"{mape_val:.4f}%" if isinstance(mape_val, (int, float)) else str(mape_val)
            
            md_content.append(f"| {sub} | {mae_str} | {rmse_str} | {mape_str} |")
    elif isinstance(metrics, dict):
        for k, v in metrics.items():
            md_content.append(f"- **{k}:** `{v}`")
    else:
        md_content.append(f"`{metrics}`")
        
    md_text = "\n".join(md_content) + "\n"
    
    md_latest = output_dir / f"{model_name}_relatorio_latest.md"
    md_ts = output_dir / f"{model_name}_relatorio_{timestamp_str}.md"
    
    with open(md_latest, "w", encoding="utf-8") as f:
        f.write(md_text)
    with open(md_ts, "w", encoding="utf-8") as f:
        f.write(md_text)
            
    print(f"[Reprodutibilidade] Relatório JSON salvo em: {json_latest} e {json_ts}")
    print(f"[Reprodutibilidade] Relatório Markdown salvo em: {md_latest} e {md_ts}")
    return json_latest, md_latest
