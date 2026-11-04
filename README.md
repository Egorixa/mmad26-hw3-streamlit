# ДЗ-3. Логистическая регрессия и Streamlit

- `Курюкин246_HW3_MMDA.ipynb` – ноутбук с решением
- `wine_app/` – приложение на Streamlit с итоговой моделью, инструкция по запуску в `wine_app/README.md`
- `Data/wine_quality.csv` – датасет [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) (P. Cortez et al., 2009, CC BY 4.0)

Приложение: https://6pz5nqe8ny3nh8ioacbrb2.streamlit.app/

Локальный запуск приложения (нужен Python 3.12):

```bash
cd wine_app
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
