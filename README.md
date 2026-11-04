# TFGINFO

Simulación del modelo neuronal Hindmarsh–Rose y representación de sus resultados.

## Requisitos

- Python 3.10 o posterior
- `pip`

## Instalación

Desde la carpeta del proyecto, crea y activa un entorno virtual:

```bash
python -m venv .venv
```

En Linux o macOS:

```bash
source .venv/bin/activate
```

En Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

Instala todas las dependencias del proyecto:

```bash
python -m pip install -r requirements.txt
```

## Ejecución

Con el entorno virtual activado:

```bash
python src/main.py
```

El programa integra el sistema con SciPy y muestra las gráficas mediante Matplotlib.
