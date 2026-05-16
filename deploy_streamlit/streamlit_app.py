from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import altair as alt
import numpy as np

st.set_page_config(page_title='XGBoost Adult Income Explorer', layout='wide')

HERE = Path(__file__).parent
artifact = joblib.load(HERE / 'xgb_adult_income_model.joblib')
model = artifact['model']
feature_names = artifact['feature_names']
numeric_features = artifact['numeric_features']
categorical_features = artifact['categorical_features']
defaults = artifact['feature_defaults']
category_options = artifact['category_options']
label_map = artifact['label_map']
best_params = artifact.get('best_params', {})
test_metrics = artifact.get('test_metrics', {})

PERSONAS = {
    'Joven Tech (25)': {
        'age': 25, 'fnlwgt': 150000, 'education-num': 14, 'capital-gain': 0, 'capital-loss': 0, 'hours-per-week': 35,
        'workclass': 'Private', 'education': 'Bachelors', 'marital-status': 'Never-married', 'occupation': 'Tech-support',
        'relationship': 'Not-in-family', 'race': 'White', 'sex': 'Male', 'native-country': 'United-States',
    },
    'Ejecutivo Senior (55)': {
        'age': 55, 'fnlwgt': 250000, 'education-num': 16, 'capital-gain': 20000, 'capital-loss': 0, 'hours-per-week': 50,
        'workclass': 'Self-emp-inc', 'education': 'Doctorate', 'marital-status': 'Married-civ-spouse', 'occupation': 'Exec-managerial',
        'relationship': 'Husband', 'race': 'White', 'sex': 'Male', 'native-country': 'United-States',
    },
    'Emprendedora (38)': {
        'age': 38, 'fnlwgt': 180000, 'education-num': 13, 'capital-gain': 5000, 'capital-loss': 0, 'hours-per-week': 45,
        'workclass': 'Self-emp-inc', 'education': 'Some-college', 'marital-status': 'Divorced', 'occupation': 'Sales',
        'relationship': 'Not-in-family', 'race': 'White', 'sex': 'Female', 'native-country': 'United-States',
    },
    'Medico (48)': {
        'age': 48, 'fnlwgt': 220000, 'education-num': 18, 'capital-gain': 3000, 'capital-loss': 0, 'hours-per-week': 55,
        'workclass': 'Private', 'education': 'Doctorate', 'marital-status': 'Married-civ-spouse', 'occupation': 'Prof-specialty',
        'relationship': 'Husband', 'race': 'White', 'sex': 'Male', 'native-country': 'United-States',
    },
    'Estudiante (20)': {
        'age': 20, 'fnlwgt': 80000, 'education-num': 10, 'capital-gain': 0, 'capital-loss': 0, 'hours-per-week': 20,
        'workclass': 'Private', 'education': 'HS-grad', 'marital-status': 'Never-married', 'occupation': 'Other-service',
        'relationship': 'Own-child', 'race': 'White', 'sex': 'Female', 'native-country': 'United-States',
    },
    'Obrero (35)': {
        'age': 35, 'fnlwgt': 170000, 'education-num': 9, 'capital-gain': 0, 'capital-loss': 0, 'hours-per-week': 42,
        'workclass': 'Private', 'education': 'HS-grad', 'marital-status': 'Married-civ-spouse', 'occupation': 'Craft-repair',
        'relationship': 'Husband', 'race': 'White', 'sex': 'Male', 'native-country': 'United-States',
    },
    'Profesor (30)': {
        'age': 30, 'fnlwgt': 140000, 'education-num': 16, 'capital-gain': 0, 'capital-loss': 0, 'hours-per-week': 38,
        'workclass': 'State-gov', 'education': 'Masters', 'marital-status': 'Married-civ-spouse', 'occupation': 'Prof-specialty',
        'relationship': 'Wife', 'race': 'White', 'sex': 'Female', 'native-country': 'United-States',
    },
    'Jubilado (65)': {
        'age': 65, 'fnlwgt': 120000, 'education-num': 8, 'capital-gain': 1000, 'capital-loss': 0, 'hours-per-week': 10,
        'workclass': 'Private', 'education': 'HS-grad', 'marital-status': 'Widowed', 'occupation': 'Other-service',
        'relationship': 'Not-in-family', 'race': 'White', 'sex': 'Male', 'native-country': 'United-States',
    },
    'Inmigrante (28)': {
        'age': 28, 'fnlwgt': 90000, 'education-num': 10, 'capital-gain': 0, 'capital-loss': 0, 'hours-per-week': 40,
        'workclass': 'Private', 'education': 'Some-college', 'marital-status': 'Never-married', 'occupation': 'Handlers-cleaners',
        'relationship': 'Not-in-family', 'race': 'Other', 'sex': 'Male', 'native-country': 'Mexico',
    },
}


def predict_proba(row_dict):
    df = pd.DataFrame([row_dict], columns=feature_names)
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in feature_names:
        df[col] = df[col].fillna(defaults[col])
    proba_pos = float(model.predict_proba(df)[:, 1][0])
    pred = int(proba_pos >= 0.5)
    return proba_pos, pred


def build_persona_row(persona_key):
    row = {}
    for f in numeric_features:
        row[f] = float(PERSONAS[persona_key].get(f, defaults[f]))
    for f in categorical_features:
        row[f] = PERSONAS[persona_key].get(f, str(defaults[f]))
    return row


def proba_bar(proba, height=100):
    pred_class = '>50K' if proba >= 0.5 else '<=50K'
    color = '#00cc66' if proba >= 0.5 else '#ff6b6b'
    df = pd.DataFrame({'probabilidad': [proba], 'clase': [pred_class]})
    chart = alt.Chart(df).mark_bar(color=color, cornerRadius=8).encode(
        x=alt.X('probabilidad:Q', scale=alt.Scale(domain=[0, 1]), title=None),
        tooltip=[alt.Tooltip('probabilidad:Q', format='.1%')],
    ).properties(height=height)
    rule = alt.Chart(pd.DataFrame({'threshold': [0.5]})).mark_rule(color='red', strokeDash=[5, 5]).encode(x='threshold:Q')
    text = alt.Chart(df).mark_text(
        align='left', dx=10, dy=-10, size=14, fontWeight='bold', color=color,
    ).encode(x='probabilidad:Q', text=alt.Text('probabilidad:Q', format='.1%'))
    return (chart + rule + text).properties(height=height)


def probability_gauge(proba):
    pct = proba * 100
    if pct >= 50:
        color = '#00cc66'
        label = f'>50K ({pct:.1f}%)'
    else:
        color = '#ff6b6b'
        label = f'<=50K ({100-pct:.1f}%)'
    return st.markdown(f"""
    <div style="background:#f0f2f6; border-radius:12px; padding:20px; text-align:center;">
        <div style="font-size:48px; font-weight:800; color:{color};">{label}</div>
        <div style="height:20px; background:#ddd; border-radius:10px; margin-top:10px; overflow:hidden;">
            <div style="height:100%; width:{pct}%; background:{color}; border-radius:10px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def comparison_chart(results):
    df = pd.DataFrame([
        {'Persona': k, 'Probabilidad >50K': v['proba'], 'Prediccion': v['label']}
        for k, v in results.items()
    ])
    chart = alt.Chart(df).mark_bar(cornerRadius=4).encode(
        x=alt.X('Persona:N', sort=None, title=None),
        y=alt.Y('Probabilidad >50K:Q', scale=alt.Scale(domain=[0, 1]), title='Probabilidad'),
        color=alt.Color('Prediccion:N', scale=alt.Scale(domain=['>50K', '<=50K'], range=['#00cc66', '#ff6b6b'])),
        tooltip=['Persona', alt.Tooltip('Probabilidad >50K:Q', format='.1%')],
    ).properties(height=350, title='Comparación entre personas')
    rule = alt.Chart(pd.DataFrame({'y': [0.5]})).mark_rule(color='red', strokeDash=[5, 5], strokeWidth=1.5).encode(y='y:Q')
    return chart + rule


def feature_impact_local(row_dict, proba_base, n_top=8):
    impacts = {}
    for col in numeric_features:
        orig = row_dict[col]
        delta = max(abs(orig) * 0.2, 1) if orig != 0 else 1
        modified = row_dict.copy()
        modified[col] = orig + delta
        p_up, _ = predict_proba(modified)
        modified[col] = max(orig - delta, 0)
        p_down, _ = predict_proba(modified)
        impacts[col] = max(abs(p_up - proba_base), abs(p_down - proba_base))

    for col in categorical_features:
        modified = row_dict.copy()
        for alt_val in category_options.get(col, []):
            if alt_val != row_dict[col]:
                modified[col] = alt_val
                p_alt, _ = predict_proba(modified)
                impacts[f'{col}={row_dict[col]}'] = max(impacts.get(f'{col}={row_dict[col]}', 0), abs(p_alt - proba_base))

    sorted_imp = sorted(impacts.items(), key=lambda x: x[1], reverse=True)[:n_top]
    df = pd.DataFrame(sorted_imp, columns=['feature', 'impact'])
    chart = alt.Chart(df).mark_bar(cornerRadius=4).encode(
        y=alt.Y('feature:N', sort='-x', title=None),
        x=alt.X('impact:Q', title='Impacto en probabilidad', scale=alt.Scale(domain=[0, df['impact'].max() * 1.1])),
        color=alt.Color('impact:Q', scale=alt.Scale(scheme='blues'), legend=None),
        tooltip=[alt.Tooltip('impact:Q', format='.3f')],
    ).properties(height=300, title='Variables con mayor impacto en esta predicción')
    return chart


st.title('Explorador de Ingreso >50K con XGBoost')
st.markdown('Selecciona una **persona predefinida** o crea tu propio perfil desde cero. El modelo predice si su ingreso supera los 50K USD/año.')

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader('Persona')
    persona_names = ['--- Personalizado ---'] + list(PERSONAS.keys())
    selected = st.selectbox('Selecciona un perfil', persona_names)

    st.subheader('Variables numéricas')
    if selected != '--- Personalizado ---' and selected in PERSONAS:
        row = build_persona_row(selected)
    else:
        row = {f: float(defaults[f]) for f in numeric_features}
        row.update({f: str(defaults[f]) for f in categorical_features})

    numeric_inputs = {}
    for f in numeric_features:
        val = row[f]
        if val == 0:
            lo, hi = 0.0, 100.0
        else:
            lo = float(max(0, val * 0.2))
            hi = float(val * 2.5 + 1)
        numeric_inputs[f] = st.number_input(f, value=float(val), min_value=lo, max_value=hi, key=f'num_{f}')
        row[f] = numeric_inputs[f]

    with st.expander('Variables categóricas'):
        for f in categorical_features:
            options = list(category_options.get(f, []))
            val = str(row.get(f, defaults[f]))
            if val not in options:
                options = [val] + options
            row[f] = st.selectbox(f, options=options, index=options.index(val), key=f'cat_{f}')

with col_right:
    proba, pred = predict_proba(row)
    pred_label = label_map[pred]

    g1, g2, g3 = st.columns(3)
    with g1:
        st.metric('Predicción', pred_label, delta=f'{"Alto" if pred==1 else "Bajo"} ingreso')
    with g2:
        st.metric('Probabilidad >50K', f'{proba:.1%}')
    with g3:
        st.metric('Probabilidad <=50K', f'{(1-proba):.1%}')

    probability_gauge(proba)

    chart_col, impact_col = st.columns(2)
    with chart_col:
        st.altair_chart(proba_bar(proba, height=80), use_container_width=True)
    with impact_col:
        st.altair_chart(feature_impact_local(row, proba), use_container_width=True)

st.divider()
st.subheader('Comparar múltiples personas')

selected_multi = st.multiselect(
    'Selecciona personas para comparar',
    list(PERSONAS.keys()),
    default=list(PERSONAS.keys())[:4],
)

if selected_multi:
    results = {}
    data_rows = []
    for k in selected_multi:
        r = build_persona_row(k)
        p, pr = predict_proba(r)
        lbl = label_map[pr]
        results[k] = {'proba': p, 'pred': pr, 'label': lbl}
        data_rows.append({'Persona': k, 'Edad': r['age'], 'Horas/sem': r['hours-per-week'], 'Educación': r['education-num'], 'Ganancia capital': r['capital-gain'], 'Probabilidad >50K': f'{p:.1%}', 'Predicción': lbl})

    comp_col, table_col = st.columns([3, 2])
    with comp_col:
        st.altair_chart(comparison_chart(results), use_container_width=True)
    with table_col:
        st.dataframe(pd.DataFrame(data_rows), use_container_width=True, hide_index=True)

    st.markdown('**Interpretación:** La línea roja marca el umbral del 50%. Personas a la derecha son predichas como >50K.')

st.divider()
col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
    st.subheader('Métricas del modelo (test)')
    if test_metrics:
        for k, v in test_metrics.items():
            if isinstance(v, float) and k != 'modelo':
                st.metric(k.replace('_', ' ').title(), f'{v:.4f}')

with col_m2:
    st.subheader('Hiperparámetros')
    for k, v in best_params.items():
        clean_k = k.replace('model__', '').replace('_', ' ').title()
        st.text(f'{clean_k}: {v}')

with col_m3:
    st.subheader('Distribución de referencia')
    st.markdown(f"""
    - Clase mayoritaria: **<=50K** (~72%)
    - Clase minoritaria: **>50K** (~28%)
    - `scale_pos_weight` calculado automáticamente
    - Validación cruzada: **5-folds StratifiedKFold**
    """)
