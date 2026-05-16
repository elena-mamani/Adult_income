# Deploy: XGBoost Adult Income Explorer — Streamlit Cloud

Esta carpeta contiene todo lo necesario para desplegar la app en **Streamlit Community Cloud**.

## Archivos

| Archivo | Propósito |
|---|---|
| `streamlit_app.py` | App principal (renombrada para detección automática) |
| `xgb_adult_income_model.joblib` | Modelo XGBoost serializado con pipeline completo |
| `requirements.txt` | Dependencias Python |

## Pasos para el deploy

1. Sube esta carpeta a un **repositorio en GitHub**:
   ```bash
   git init
   git add .
   git commit -m "deploy: streamlit adult income explorer"
   git remote add origin https://github.com/TU_USUARIO/TU_REPO.git
   git push -u origin main
   ```

2. Ve a **https://streamlit.io/community-cloud** e inicia sesión con tu cuenta de GitHub.

3. Presiona **"New app"**:
   - Repositorio: `TU_USUARIO/TU_REPO`
   - Branch: `main`
   - Main file: `streamlit_app.py`
   - Presiona **"Deploy"**

4. En minutos la app estará online en `https://TU_USUARIO-TU_REPO.streamlit.app/`

### Notas

- El modelo (~345 KB) se incluye directamente en el repo. No necesita Git LFS.
- La app usa `altair` para gráficos (incluido en `requirements.txt`).
- No requiere secretos ni keys de API.
- No necesita `packages.txt` — todas las dependencias son Python puras.
