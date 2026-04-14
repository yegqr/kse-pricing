import streamlit as st
from pathlib import Path
HERE = Path(__file__).parent
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="KSE · Оптимальна ціна", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

BLUE   = "#1B2B6B"
LIGHT  = "#4A6CF7"
GREEN  = "#27AE60"
RED    = "#E74C3C"
ORANGE = "#F39C12"
GRAY   = "#F4F6FB"
PALETTE = [BLUE, LIGHT, GREEN, ORANGE, RED, "#8E44AD", "#1ABC9C", "#E67E22"]
SURVEY_MIN = 510000.0   # T3 price_total min (4-річний тотал)
SURVEY_MAX = 610000.0   # T3 price_total max (4-річний тотал)
SURVEY_MIN_YR = SURVEY_MIN / 4   # 127.5k грн/рік
SURVEY_MAX_YR = SURVEY_MAX / 4   # 152.5k грн/рік

st.markdown(f"""<style>
[data-testid="stSidebar"]{{background:{BLUE};}}
[data-testid="stSidebar"] *{{color:white!important;}}
.kpi{{background:{GRAY};border-left:5px solid {BLUE};border-radius:8px;padding:13px 16px;margin:5px 0;}}
.kpi-label{{color:#777;font-size:12px;margin:0 0 2px 0;}}
.kpi-value{{color:{BLUE};font-size:22px;font-weight:700;margin:0;line-height:1.2;}}
.kpi-sub{{color:#999;font-size:11px;margin:2px 0 0 0;}}
.kpi-green{{border-left-color:{GREEN};}}
.kpi-red{{border-left-color:{RED};}}
.kpi-orange{{border-left-color:{ORANGE};}}
.kpi-gray{{border-left-color:#bbb;}}
.formula{{background:white;border:2px solid {BLUE};border-radius:10px;padding:12px 18px;
  font-size:18px;font-weight:700;color:{BLUE};text-align:center;margin:10px 0;
  font-family:'Courier New',monospace;}}
.note{{background:#FFF8E1;border-left:4px solid {ORANGE};border-radius:6px;padding:10px 14px;margin:8px 0;font-size:14px;}}
.warn{{background:#FDEDEC;border-left:4px solid {RED};border-radius:6px;padding:10px 14px;margin:8px 0;font-size:14px;}}
.good{{background:#E8F8F0;border-left:4px solid {GREEN};border-radius:6px;padding:10px 14px;margin:8px 0;font-size:14px;}}
.expl{{background:{GRAY};border-radius:10px;padding:14px 18px;margin:8px 0;font-size:14px;line-height:1.7;}}
.src-badge{{display:inline-block;background:{LIGHT};color:white;border-radius:4px;
  padding:2px 8px;font-size:11px;font-weight:600;margin-right:6px;}}
.src-badge-g{{display:inline-block;background:{GREEN};color:white;border-radius:4px;
  padding:2px 8px;font-size:11px;font-weight:600;margin-right:6px;}}
</style>""", unsafe_allow_html=True)

# ─── DATA ────────────────────────────────────────────────────────────────────
@st.cache_data
def load():
    base = str(HERE) + "/"
    p1  = pd.read_csv(base + "pivot1_enrolled_by_spec_year.csv")
    t1  = pd.read_csv(base + "T1_programs_prices.csv")
    t3  = pd.read_csv(base + "T3_abit_wtp.csv")
    t2a = pd.read_csv(base + "T2_survey_agg.csv")
    t2i = pd.read_csv(base + "T2_survey_individual.csv")
    p4  = pd.read_csv(base + "pivot4_median_score_enrolled.csv")
    p2  = pd.read_csv(base + "pivot2_budget_contract.csv")
    p3  = pd.read_csv(base + "pivot3_priority_enrolled.csv")
    t1b = t1[t1['degree'] == 'bachelor'].copy()

    MAPPING = {
        'Економіка':                          'Бізнес-економіка',
        'Інженерія програмного забезпечення': 'Програмна інженерія',
        "Комп'ютерні науки":                  'Штучний інтелект',
        'Кібербезпека та захист інформації':  'Кібербезпека',
        'Математика':                          'Прикладна математика',
        'Право':                               'Право',
        'Прикладна математика':                'Прикладна математика',
        'Психологія':                          'Психологія',
    }
    rows = []
    for i in range(len(p1)):
        r    = p1.iloc[i]
        sname = r['spec_name']
        kse   = MAPPING.get(sname, sname)
        tr    = t1b[t1b['program'] == kse]
        pr2   = p2[p2['spec_name'] == sname]
        contract_2025 = float(pr2['contract_2025'].values[0]) if not pr2.empty and not pd.isna(pr2['contract_2025'].values[0]) else None
        budget_2025   = float(pr2['budget_2025'].values[0])   if not pr2.empty and not pd.isna(pr2['budget_2025'].values[0])   else 0.0
        enrollment_plan = float(tr['enrollment_plan'].values[0]) if not tr.empty and not pd.isna(tr['enrollment_plan'].values[0]) else None
        rows.append({
            'edebo':           sname,
            'kse':             kse,
            'price_cur':       float(tr['price_2025_year'].values[0])  if not tr.empty and not pd.isna(tr['price_2025_year'].values[0])  else None,
            'mc_default':      float(tr['true_price_year'].values[0])  if not tr.empty and not pd.isna(tr['true_price_year'].values[0])  else None,
            'prop1':           float(tr['price_prop1_year'].values[0]) if not tr.empty and not pd.isna(tr['price_prop1_year'].values[0]) else None,
            'prop2':           float(tr['price_prop2_year'].values[0]) if not tr.empty and not pd.isna(tr['price_prop2_year'].values[0]) else None,
            'enrollment_plan': enrollment_plan,
            'contract_2025':   contract_2025,
            'budget_2025':     budget_2025,
            'q2025':           float(r['2025']) if not pd.isna(r['2025']) else None,
            'q2024':           float(r['2024']) if not pd.isna(r['2024']) else None,
            'q2023':           float(r['2023']) if not pd.isna(r['2023']) else None,
        })
    master = pd.DataFrame(rows)
    return master, t3, t2a, t2i, p4, p2, p3, p1, t1

master, t3, t2a, t2i, p4, p2, p3, p1, t1 = load()

@st.cache_data
def fit_elas():
    pts = t3[t3['wtp_weighted'] > 0].copy()
    lp  = np.log(pts['price_year'].values.astype(float))
    lq  = np.log(pts['wtp_weighted'].values.astype(float))
    sl, ic, r, _, se = stats.linregress(lp, lq)
    return float(sl), float(ic), float(r**2), float(se)

EPS_BASE, EPS_INT, EPS_R2, EPS_SE = fit_elas()
EPS_LOW  = max(1.5, abs(EPS_BASE) - 2.0)   # нижня межа чутливості
EPS_HIGH = abs(EPS_BASE) + 2.0              # верхня межа чутливості

def q_of_p(P, P_anch, Q_anch, eps):
    if Q_anch is None or P_anch is None or P_anch == 0 or P <= 0:
        return 0.0
    return max(0.0, Q_anch * (P / P_anch) ** eps)

def p_star_fn(mc, eps_abs):
    if eps_abs <= 1.0:
        return None
    return mc * eps_abs / (eps_abs - 1.0)

def kpi(label, value, sub="", cls=""):
    sub_html = f'<p class="kpi-sub">{sub}</p>' if sub else ""
    return f'<div class="kpi {cls}"><p class="kpi-label">{label}</p><p class="kpi-value">{value}</p>{sub_html}</div>'

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 KSE Pricing")
    st.markdown("---")
    page = st.radio("", ["💰 Калькулятор", "📚 Методологія", "🔧 Розширені дані"])

# ═══════════════════════════════════════════════════════════════════════════════
# КАЛЬКУЛЯТОР
# ═══════════════════════════════════════════════════════════════════════════════
if page == "💰 Калькулятор":
    st.title("💰 Яка ціна максимізує прибуток KSE?")
    st.caption("Бакалавраt · Реальний набір 2025 + опитування абітурієнтів (n=108) · Всі ціни в грн 2025 року")

    # ── Spec selection
    st.markdown("### ① Оберіть спеціальність")
    specs_ok = master[master['q2025'].notna() & master['price_cur'].notna()].copy()
    col_sp, _ = st.columns([2, 1])
    with col_sp:
        sel = st.selectbox("", specs_ok['edebo'].tolist(), label_visibility="collapsed")
    row = specs_ok[specs_ok['edebo'] == sel].iloc[0]

    st.markdown("---")
    st.markdown("### ② Параметри моделі")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        mc_def = row['mc_default'] if row['mc_default'] else 130000.0
        mc_val = st.number_input(
            "Граничні витрати MC, грн/рік",
            value=float(mc_def), step=1000.0, format="%.0f",
            help="Оцінка додаткових витрат KSE на одного студента за рік. "
                 "За замовчуванням — proxy з T1 (true_price_year). "
                 "Це не обов'язково бухгалтерська собівартість — змінюй вручну.")

    with c2:
        eps_val = st.number_input(
            "Еластичність попиту |ε|",
            value=round(abs(EPS_BASE), 2), step=0.01, format="%.2f",
            help=f"На скільки % зміниться попит, якщо ціна зміниться на 1%. "
                 f"Базове значення {abs(EPS_BASE):.2f} отримано з опитування абітурієнтів "
                 f"через log-log регресію на WTP-точках (R²={EPS_R2:.3f}). "
                 f"Не є program-specific оцінкою.")

    with c3:
        st.markdown("**Інфляційна поправка**")
        st.markdown('<div style="background:#E8F8F0;border-left:3px solid #27AE60;border-radius:6px;padding:8px 12px;font-size:13px;">✅ Усі ціни вже в грн 2025 року. Інфляція = 0 за замовчуванням.</div>', unsafe_allow_html=True)
        infl_val = 0.0

    with c4:
        share_fp = st.number_input(
            "Частка full-pay студентів, %",
            value=40.0, step=1.0, format="%.0f",
            help="Частка контрактників що платять повну вартість без знижок/грантів. "
                 "Використовується для коректного розрахунку revenue і profit. "
                 "Дефолт 40% — приблизна частка контрактників від загального набору 2025.")

    # ── Core calculations
    P_cur        = row['price_cur'] * (1 + infl_val / 100)
    Q_total      = row['q2025']
    # contract_2025 від P2 = кількість у рейтингу, НЕ зараховані.
    # Оцінюємо частку зарахованих контрактників пропорційно до структури рейтингу.
    _contract_rating = row['contract_2025'] if row['contract_2025'] else Q_total * 0.7
    _budget_rating   = row['budget_2025']   if row['budget_2025']   else 0.0
    _total_rating    = _contract_rating + _budget_rating
    if _total_rating > 0:
        contract_share_in_enrolled = _contract_rating / _total_rating
        contract_25 = min(round(Q_total * contract_share_in_enrolled), int(Q_total))
    else:
        contract_25 = round(Q_total * 0.7)
    share        = share_fp / 100.0
    Q_pay_anch   = contract_25 * share           # full-pay anchor
    eps_neg      = -abs(eps_val)
    enroll_plan  = row['enrollment_plan']

    P_opt        = p_star_fn(mc_val, eps_val)
    Q_total_opt  = q_of_p(P_opt, P_cur, Q_total,   eps_neg) if P_opt else None
    Q_pay_opt    = q_of_p(P_opt, P_cur, Q_pay_anch, eps_neg) if P_opt else None
    # Uncertainty range: P* for low/base/high ε
    P_opt_low    = p_star_fn(mc_val, EPS_LOW)   # менш еластичний → вища P*
    P_opt_high   = p_star_fn(mc_val, EPS_HIGH)  # більш еластичний → нижча P*
    P_opt_lo_val = min(x for x in [P_opt_low, P_opt, P_opt_high] if x) if P_opt else None
    P_opt_hi_val = max(x for x in [P_opt_low, P_opt, P_opt_high] if x) if P_opt else None

    Rev_cur      = P_cur * Q_pay_anch
    Rev_opt      = P_opt * Q_pay_opt if P_opt else None
    Pr_cur       = (P_cur - mc_val) * Q_pay_anch
    Pr_opt       = (P_opt - mc_val) * Q_pay_opt if P_opt else None

    dp           = (P_opt - P_cur) / P_cur * 100 if P_opt else None
    dpr          = Pr_opt - Pr_cur if Pr_opt else None

    # Extrapolation warning
    out_of_range = P_opt is not None and (P_opt < SURVEY_MIN or P_opt > SURVEY_MAX)

    st.markdown("---")
    st.markdown("### ③ Результат")

    # Row 1 KPIs: price
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1:
        st.markdown(kpi("Поточна ціна (4 роки)", f"{P_cur/1000:.0f}k грн",
                        f"= {P_cur/4/1000:.0f}k грн/рік"), unsafe_allow_html=True)
    with r1c2:
        if P_opt:
            cls = "kpi-green" if dp > 0 else "kpi-red"
            arr = "▲" if dp > 0 else "▼"
            range_str = f"діапазон {P_opt_lo_val/1000:.0f}k–{P_opt_hi_val/1000:.0f}k (ε sensitivity)" if P_opt_lo_val else ""
            st.markdown(kpi("Оптимальна ціна P*",
                            f"{P_opt/1000:.0f}k грн (4 роки) {arr}",
                            f"= {P_opt/4/1000:.0f}k/рік | {dp:+.1f}% | {range_str}", cls),
                        unsafe_allow_html=True)
    with r1c3:
        if dpr is not None:
            cls = "kpi-green" if dpr > 0 else "kpi-red"
            st.markdown(kpi("Δ Прибуток / рік", f"{dpr/1e6:+.2f}M грн",
                            "на full-pay базі", cls), unsafe_allow_html=True)
    with r1c4:
        plan_str = f"{enroll_plan:.0f}" if enroll_plan else "—"
        st.markdown(kpi("План набору (T1)", plan_str, "enrollment_plan", "kpi-gray"),
                    unsafe_allow_html=True)

    # Row 2 KPIs: students
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    with r2c1:
        st.markdown(kpi("Студентів зараз (всього)", f"{Q_total:.0f}",
                        f"контракт (оцінка): {contract_25:.0f}"), unsafe_allow_html=True)
    with r2c2:
        st.markdown(kpi("Full-pay еквівалент зараз", f"{Q_pay_anch:.0f}",
                        f"= {contract_25:.0f} контракт × {share_fp:.0f}% full-pay"), unsafe_allow_html=True)
    with r2c3:
        if Q_total_opt is not None:
            dq = (Q_total_opt - Q_total) / Q_total * 100
            cls = "kpi-green" if Q_total_opt > Q_total else "kpi-red"
            st.markdown(kpi("Студентів при P* (всього)", f"{Q_total_opt:.0f}",
                            f"{dq:+.0f}% від поточних", cls), unsafe_allow_html=True)
    with r2c4:
        st.markdown(kpi("Виручка зараз / рік", f"{Rev_cur/1e6:.2f}M грн",
                        "на full-pay базі"), unsafe_allow_html=True)

    # Insight box
    if P_opt and dp is not None:
        if out_of_range:
            st.markdown(f'<div class="warn">⚠️ <strong>Екстраполяція:</strong> P* = {P_opt/1000:.0f}k грн за 4 роки ({P_opt/4/1000:.0f}k/рік) виходить за межі survey-точок ({SURVEY_MIN/1000:.0f}k–{SURVEY_MAX/1000:.0f}k за 4 роки / {SURVEY_MIN_YR/1000:.0f}k–{SURVEY_MAX_YR/1000:.0f}k/рік). Крива попиту в цій зоні не підкріплена реальними даними опитування — оцінка менш надійна.</div>', unsafe_allow_html=True)
        if abs(dp) < 5:
            st.markdown(f'<div class="good">✅ <strong>Поточна ціна близька до оптимальної</strong> (відхилення {abs(dp):.1f}%). Суттєвих змін не потрібно.</div>', unsafe_allow_html=True)
        elif dp > 0:
            st.markdown(f'<div class="good">✅ <strong>Можна підняти ціну</strong> з {P_cur/1000:.0f}k до {P_opt/1000:.0f}k грн (4 роки, {P_opt/4/1000:.0f}k/рік) (+{dp:.1f}%). Full-pay студентів зміниться з {Q_pay_anch:.0f} до {Q_pay_opt:.0f}. Δ прибуток: <strong>{dpr/1e6:+.2f}M грн/рік</strong>.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="note">⚠️ <strong>Поточна ціна вища за оптимальну</strong> на {abs(dp):.1f}%. Зниження до {P_opt/1000:.0f}k/рік підвищить прибуток на <strong>{dpr/1e6:+.2f}M грн/рік</strong>.</div>', unsafe_allow_html=True)

    # ── Assumption disclaimer
    st.markdown('<div class="note">📌 <strong>Припущення моделі:</strong> (1) частка full-pay студентів не залежить від ціни — в реальності при вищих цінах частка грантів зростає; (2) еластичність постійна по всьому діапазону цін.</div>', unsafe_allow_html=True)

    # ── Charts
    st.markdown("---")
    st.markdown("### ④ Графіки")

    p_lo  = max(5000.0, P_cur * 0.3)
    p_hi  = P_cur * 3.0
    p_rng = np.linspace(p_lo, p_hi, 600)

    Q_total_rng  = np.array([q_of_p(p, P_cur, Q_total,    eps_neg) for p in p_rng])
    Q_pay_rng    = np.array([q_of_p(p, P_cur, Q_pay_anch, eps_neg) for p in p_rng])
    Rev_rng      = p_rng * Q_pay_rng
    Pr_rng       = (p_rng - mc_val) * Q_pay_rng

    def base_layout(ytitle, height=420):
        return dict(height=height, plot_bgcolor='white',
                    xaxis=dict(title="Ціна, тис. грн (4 роки)", showgrid=True, gridcolor='#eee'),
                    yaxis=dict(title=ytitle, showgrid=True, gridcolor='#eee'),
                    legend=dict(bgcolor='rgba(255,255,255,0.92)', bordercolor='#ddd',
                                borderwidth=1, x=0.01, y=0.99),
                    margin=dict(t=30, b=55, l=72, r=25))

    tab_d, tab_r, tab_p, tab_s = st.tabs([
        "📉 Попит Q(P)", "💵 Виручка TR(P)", "📈 Прибуток π(P)", "🎚 Чутливість MC"
    ])

    with tab_d:
        fig = go.Figure()
        # Survey support shading
        fig.add_vrect(x0=SURVEY_MIN/1000, x1=SURVEY_MAX/1000,
                      fillcolor="rgba(39,174,96,0.07)", layer="below",
                      line_width=0, annotation_text="Survey опитування (4 роки)",
                      annotation_position="top left",
                      annotation_font=dict(size=10, color=GREEN))
        fig.add_trace(go.Scatter(x=p_rng/1000, y=Q_total_rng, mode='lines',
            name='Всього студентів Q(P)',
            line=dict(color=BLUE, width=2.5)))
        fig.add_trace(go.Scatter(x=p_rng/1000, y=Q_pay_rng, mode='lines',
            name='Full-pay еквівалент',
            line=dict(color=LIGHT, width=2, dash='dash')))
        # enrollment plan
        if enroll_plan:
            fig.add_hline(y=enroll_plan, line_dash='dot', line_color=ORANGE, line_width=2,
                          annotation_text=f"План набору: {enroll_plan:.0f}",
                          annotation_position="bottom right",
                          annotation_font=dict(color=ORANGE, size=11))
        # Current point
        fig.add_trace(go.Scatter(x=[P_cur/1000], y=[Q_total],
            mode='markers', name=f'Зараз ({Q_total:.0f} студ.)',
            marker=dict(color=GREEN, size=14, symbol='star',
                        line=dict(color='white', width=2))))
        if P_opt and Q_total_opt:
            fig.add_trace(go.Scatter(x=[P_opt/1000], y=[Q_total_opt],
                mode='markers', name=f'P* ({Q_total_opt:.0f} студ.)',
                marker=dict(color=RED, size=14, symbol='diamond',
                            line=dict(color='white', width=2))))
            fig.add_vline(x=P_opt/1000, line_dash='dash', line_color=RED, line_width=1.5)
        fig.add_vline(x=P_cur/1000, line_dash='dot', line_color=GREEN, line_width=1.5)
        fig.update_layout(**base_layout("Кількість студентів"))
        st.plotly_chart(fig, use_container_width=True)
        # Plan vs model note
        if P_opt and Q_total_opt and enroll_plan:
            if Q_total_opt > enroll_plan * 1.05:
                st.markdown(f'<div class="note">📋 При P* модельний попит ({Q_total_opt:.0f}) вищий за план набору ({enroll_plan:.0f}). KSE може або обмежити набір, або підняти ціну ще вище.</div>', unsafe_allow_html=True)
            elif Q_total_opt < enroll_plan * 0.95:
                st.markdown(f'<div class="note">📋 При P* модельний попит ({Q_total_opt:.0f}) нижчий за план набору ({enroll_plan:.0f}). Для виконання плану потрібна ціна нижче P*.</div>', unsafe_allow_html=True)

    with tab_r:
        fig = go.Figure()
        fig.add_vrect(x0=SURVEY_MIN/1000, x1=SURVEY_MAX/1000,
                      fillcolor="rgba(39,174,96,0.07)", layer="below", line_width=0)
        fig.add_trace(go.Scatter(x=p_rng/1000, y=Rev_rng/1e6, mode='lines',
            name='Виручка TR (full-pay)',
            line=dict(color=LIGHT, width=3),
            fill='tozeroy', fillcolor='rgba(74,108,247,0.07)'))
        fig.add_trace(go.Scatter(x=[P_cur/1000], y=[Rev_cur/1e6], mode='markers',
            name=f'Зараз ({Rev_cur/1e6:.2f}M)',
            marker=dict(color=GREEN, size=14, symbol='star', line=dict(color='white', width=2))))
        if P_opt and Rev_opt:
            fig.add_trace(go.Scatter(x=[P_opt/1000], y=[Rev_opt/1e6], mode='markers',
                name=f'При P* ({Rev_opt/1e6:.2f}M)',
                marker=dict(color=RED, size=14, symbol='diamond', line=dict(color='white', width=2))))
            fig.add_vline(x=P_opt/1000, line_dash='dash', line_color=RED, line_width=1.5)
        fig.add_vline(x=P_cur/1000, line_dash='dot', line_color=GREEN, line_width=1.5)
        fig.update_layout(**base_layout("Виручка, млн грн/рік"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Виручка рахується на full-pay базі: contract_2025 × share_fullpay × ціна")

    with tab_p:
        fig = go.Figure()
        fig.add_vrect(x0=SURVEY_MIN/1000, x1=SURVEY_MAX/1000,
                      fillcolor="rgba(39,174,96,0.07)", layer="below", line_width=0)
        fig.add_trace(go.Scatter(x=p_rng/1000, y=Pr_rng/1e6, mode='lines',
            name='Прибуток π (full-pay)',
            line=dict(color=GREEN, width=3),
            fill='tozeroy', fillcolor='rgba(39,174,96,0.07)'))
        fig.add_hline(y=0, line_color='#bbb', line_width=1)
        fig.add_trace(go.Scatter(x=[P_cur/1000], y=[Pr_cur/1e6], mode='markers',
            name=f'Зараз ({Pr_cur/1e6:.2f}M)',
            marker=dict(color=BLUE, size=14, symbol='star', line=dict(color='white', width=2))))
        if P_opt and Pr_opt:
            fig.add_trace(go.Scatter(x=[P_opt/1000], y=[Pr_opt/1e6], mode='markers',
                name=f'При P* ({Pr_opt/1e6:.2f}M, макс.)',
                marker=dict(color=RED, size=14, symbol='diamond', line=dict(color='white', width=2))))
            fig.add_vline(x=P_opt/1000, line_dash='dash', line_color=RED, line_width=1.5)
        fig.add_vline(x=P_cur/1000, line_dash='dot', line_color=BLUE, line_width=1.5)
        # Sensitivity bands
        Pr_rng_low  = np.array([(p - mc_val) * q_of_p(p, P_cur, Q_pay_anch, -EPS_LOW)  for p in p_rng])
        Pr_rng_high = np.array([(p - mc_val) * q_of_p(p, P_cur, Q_pay_anch, -EPS_HIGH) for p in p_rng])
        fig.add_trace(go.Scatter(
            x=np.concatenate([p_rng, p_rng[::-1]])/1000,
            y=np.concatenate([Pr_rng_high, Pr_rng_low[::-1]])/1e6,
            fill='toself', fillcolor='rgba(39,174,96,0.12)',
            line=dict(color='rgba(0,0,0,0)'),
            name=f'Діапазон ε={EPS_LOW:.1f}–{EPS_HIGH:.1f}',
            showlegend=True
        ))
        fig.update_layout(**base_layout("Прибуток, млн грн/рік"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Зелена крива = базовий сценарій (ε={eps_val:.2f}). Смуга = діапазон при ε від {EPS_LOW:.1f} до {EPS_HIGH:.1f}. Прибуток = (P − MC) × Q_full_pay.")

    with tab_s:
        mc_g  = np.linspace(30000, 500000, 400)
        ps_g  = mc_g * eps_val / (eps_val - 1)
        pr_g  = np.array([(ps-mc)*q_of_p(ps, P_cur, Q_pay_anch, eps_neg)
                           for ps, mc in zip(ps_g, mc_g)])
        fig = make_subplots(rows=1, cols=2,
                             subplot_titles=("P* при різних MC", "Прибуток при P* — різні MC"),
                             horizontal_spacing=0.13)
        fig.add_trace(go.Scatter(x=mc_g/1000, y=ps_g/1000, mode='lines',
            name='P*(MC)', line=dict(color=BLUE, width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=[mc_val/1000], y=[P_opt/1000 if P_opt else 0],
            mode='markers', name='Ваш MC',
            marker=dict(color=RED, size=12, symbol='circle', line=dict(color='white', width=2))),
            row=1, col=1)
        # Survey range shading on sensitivity
        fig.add_hrect(y0=SURVEY_MIN/1000, y1=SURVEY_MAX/1000,
                      fillcolor="rgba(39,174,96,0.1)", layer="below", line_width=0,
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=mc_g/1000, y=pr_g/1e6, mode='lines',
            name='π при P*', line=dict(color=GREEN, width=2.5),
            showlegend=False), row=1, col=2)
        if P_opt and Pr_opt:
            fig.add_trace(go.Scatter(x=[mc_val/1000], y=[Pr_opt/1e6],
                mode='markers', showlegend=False,
                marker=dict(color=RED, size=12, symbol='circle', line=dict(color='white', width=2))),
                row=1, col=2)
        fig.update_xaxes(title_text="MC, тис. грн/рік", showgrid=True, gridcolor='#eee')
        fig.update_yaxes(title_text="P*, тис. грн/рік", row=1, col=1, showgrid=True, gridcolor='#eee')
        fig.update_yaxes(title_text="Прибуток, млн грн/рік", row=1, col=2, showgrid=True, gridcolor='#eee')
        fig.update_layout(height=420, plot_bgcolor='white',
                           margin=dict(t=55, b=55, l=72, r=25),
                           legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.9)',
                                       bordercolor='#ddd', borderwidth=1))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Зелена смуга = survey-діапазон. Розраховано за фіксованої форми попиту та еластичності — якщо вони зміняться з ціною, реальний прибуток відрізнятиметься.")

    # All-specs table
    with st.expander("📋 Всі спеціальності одним поглядом", expanded=False):
        rows_all = []
        for _, r in specs_ok.iterrows():
            if r['q2025'] is None or r['price_cur'] is None: continue
            Pc    = r['price_cur'] * (1 + infl_val / 100)
            mc_r  = r['mc_default'] if r['mc_default'] else mc_val
            c25   = r['contract_2025'] if r['contract_2025'] else r['q2025'] * 0.7
            Qpa   = c25 * share
            Ps    = p_star_fn(mc_r, eps_val)
            if not Ps: continue
            Qs_t  = q_of_p(Ps, Pc, r['q2025'], eps_neg)
            Qs_p  = q_of_p(Ps, Pc, Qpa, eps_neg)
            dp2   = (Ps - Pc) / Pc * 100
            dpr2  = ((Ps-mc_r)*Qs_p) - ((Pc-mc_r)*Qpa)
            oob   = "⚠️" if (Ps < SURVEY_MIN or Ps > SURVEY_MAX) else "✅"
            rows_all.append({
                'Спец':        r['edebo'],
                'P_cur k/рік': f"{Pc/1000:.0f}",
                'MC k/рік':    f"{mc_r/1000:.0f}",
                'P* k/рік':    f"{Ps/1000:.0f}",
                'Δ ціна %':    f"{dp2:+.1f}%",
                'Q_total':     int(r['q2025']),
                'Q при P*':    f"{Qs_t:.0f}",
                'План':        f"{r['enrollment_plan']:.0f}" if r['enrollment_plan'] else "—",
                'Δ Прибуток M':f"{dpr2/1e6:+.2f}",
                'Survey':      oob,
            })
        if rows_all:
            st.dataframe(pd.DataFrame(rows_all), hide_index=True, use_container_width=True)
            st.caption("MC — true_price_year з T1. |ε| з поля вище. ⚠️ = P* поза survey-діапазоном.")

    with st.expander("📊 P* sensitivity: низька / базова / висока еластичність", expanded=False):
        st.markdown(f"При MC = {mc_val/1000:.0f}k грн і трьох сценаріях |ε|:")
        sens_rows = []
        for eps_s, label in [(EPS_LOW, f"Низька ε={EPS_LOW:.1f}"), (eps_val, f"Базова ε={eps_val:.2f}"), (EPS_HIGH, f"Висока ε={EPS_HIGH:.1f}")]:
            ps = p_star_fn(mc_val, eps_s)
            if ps:
                qt = q_of_p(ps, P_cur, Q_total, -eps_s)
                qp = q_of_p(ps, P_cur, Q_pay_anch, -eps_s)
                pr = (ps - mc_val) * qp
                sens_rows.append({
                    'Сценарій': label,
                    'P* (4 роки)': f"{ps/1000:.0f}k грн",
                    'P* (рік)': f"{ps/4/1000:.0f}k грн",
                    'Q всього': f"{qt:.0f}",
                    'Q full-pay': f"{qp:.0f}",
                    'Прибуток M грн/рік': f"{pr/1e6:.2f}",
                })
        if sens_rows:
            st.dataframe(pd.DataFrame(sens_rows), hide_index=True, use_container_width=True)
            if P_opt_lo_val and P_opt_hi_val:
                st.markdown(f'<div class="note">📌 При поточних параметрах: P* ≈ <strong>{P_opt/1000:.0f}k грн</strong> (діапазон <strong>{P_opt_lo_val/1000:.0f}k–{P_opt_hi_val/1000:.0f}k</strong> за 4 роки залежно від ε). Невизначеність ≈ ±{(P_opt_hi_val-P_opt_lo_val)/2/1000:.0f}k грн.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# МЕТОДОЛОГІЯ
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📚 Методологія":
    st.title("📚 Як це працює")

    tab_ov, tab_eps, tab_mc, tab_infl, tab_ex = st.tabs([
        "🧠 Загальна логіка", "📐 Звідки ε", "💸 Звідки MC", "📅 Інфляція", "🔢 Числовий приклад"
    ])

    with tab_ov:
        st.markdown("""<div class="expl">
        <strong>Мета:</strong> знайти ціну яка максимізує прибуток KSE.<br>
        Прибуток = (Ціна − MC) × Кількість full-pay студентів.<br><br>
        Проблема: ціна і кількість <strong>пов'язані зворотно</strong>.
        Підняв ціну — менше студентів. Знизив — більше, але з меншою маржею.<br>
        Є точка де добуток (P−MC)×Q максимальний — це <strong>P*</strong>.
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**① Дані про реальний набір (P1)**")
            st.markdown('<span class="src-badge">P1</span> pivot1_enrolled_by_spec_year.csv', unsafe_allow_html=True)
            st.markdown("""<div class="expl">
            Скільки студентів реально прийшло в 2021–2025. Дає <strong>Q_anchor</strong> — реальну кількість при поточній ціні.
            </div>""", unsafe_allow_html=True)
            p1_2025 = master[master['q2025'].notna()][['edebo','q2025']].sort_values('q2025')
            fig = go.Figure(go.Bar(x=p1_2025['q2025'], y=p1_2025['edebo'], orientation='h',
                marker_color=BLUE, text=p1_2025['q2025'].apply(lambda x: f"{x:.0f}"),
                textposition='outside'))
            fig.update_layout(height=280, margin=dict(t=10, b=20, l=10, r=45),
                              xaxis=dict(title="Зараховано 2025", showgrid=True, gridcolor='#eee'),
                              yaxis=dict(showgrid=False), plot_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**② Опитування WTP (T3)**")
            st.markdown('<span class="src-badge-g">T3</span> T3_abit_wtp.csv · n=108 абітурієнтів', unsafe_allow_html=True)
            st.markdown("""<div class="expl">
            Питали: «Чи вступиш якщо ціна за 4 роки = X?» при 4 рівнях X. Дає <strong>форму кривої попиту</strong>.
            </div>""", unsafe_allow_html=True)
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=t3['price_year']/1000, y=t3['pct_yes']*100, name='«Так»',
                marker_color=BLUE, width=3.5))
            fig2.add_trace(go.Bar(x=t3['price_year']/1000, y=t3['pct_maybe']*100, name='«Можливо»',
                marker_color=LIGHT, width=3.5))
            fig2.add_trace(go.Bar(x=t3['price_year']/1000, y=t3['pct_no']*100, name='«Ні»',
                marker_color='#ddd', width=3.5))
            fig2.update_layout(barmode='stack', height=280,
                xaxis=dict(title="Ціна, тис. грн/рік", showgrid=True, gridcolor='#eee', dtick=5),
                yaxis=dict(title="%", showgrid=True, gridcolor='#eee'),
                plot_bgcolor='white', legend=dict(x=0.60, y=0.98, bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#ddd', borderwidth=1),
                margin=dict(t=10, b=35, l=50, r=10))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.markdown("**③ Формула оптимальної ціни (правило Лернера)**")
        st.markdown('<div class="formula">P* = MC × |ε| / (|ε| − 1)</div>', unsafe_allow_html=True)
        st.markdown("""<div class="expl">
        <strong>Логіка:</strong> підняли ціну на 1 грн.<br>
        &nbsp;&nbsp;➕ Заробили 1 грн з кожного з Q_pay студентів<br>
        &nbsp;&nbsp;➖ Втратили |ε|/P × Q студентів, кожен приносив (P−MC)<br><br>
        Оптимум — де ці ефекти рівні. Прирівнюємо і виводимо формулу.
        </div>""", unsafe_allow_html=True)

    with tab_eps:
        st.subheader("Звідки взялося базове значення |ε|?")
        st.markdown(f'<span class="src-badge-g">T3</span> Джерело: T3_abit_wtp.csv · n=108', unsafe_allow_html=True)
        st.markdown("""<div class="expl">
        <strong>Кроки:</strong><br>
        1. З T3 беремо 4 точки (ціна/рік, wtp_weighted) де wtp_weighted = pct_yes + 0.5 × pct_maybe<br>
        2. Логарифмуємо обидві змінні: ln(Q) і ln(P)<br>
        3. Будуємо лінійну регресію на лог-лог просторі<br>
        4. Нахил прямої = ε<br>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="formula">ln(Q) = α + ε × ln(P)</div>', unsafe_allow_html=True)

        col_l, col_r = st.columns([1, 1])
        with col_l:
            st.markdown("**Вхідні точки:**")
            t3_show = t3[['price_year','pct_yes','pct_maybe','wtp_weighted','n']].copy()
            t3_show.columns = ['Ціна/рік', '% Так', '% Можливо', 'WTP weighted', 'n']
            st.dataframe(t3_show, hide_index=True, use_container_width=True)
            st.markdown(f"""<div class="expl">
            <strong>Результат регресії:</strong><br>
            ε = <strong>{EPS_BASE:.4f}</strong><br>
            R² = <strong>{EPS_R2:.4f}</strong> (майже ідеальна лінія)<br>
            SE(ε) = {EPS_SE:.4f}<br><br>
            Інтерпретація: ціна +1% → попит −{abs(EPS_BASE):.1f}%.
            </div>""", unsafe_allow_html=True)
            st.markdown("""<div class="note">
            ⚠️ <strong>Обмеження:</strong><br>
            • Не є program-specific — одна ε для всіх програм<br>
            • Не є causal estimate — кореляція з survey WTP<br>
            • Тільки 4 точки в діапазоні 127–152k/рік<br>
            • Змінюй в калькуляторі якщо маєш кращу оцінку
            </div>""", unsafe_allow_html=True)

        with col_r:
            lp_v = np.log(t3['price_year'].values.astype(float))
            lq_v = np.log(t3['wtp_weighted'].values.astype(float))
            pf   = np.linspace(lp_v.min()-0.08, lp_v.max()+0.08, 100)
            qf   = EPS_INT + EPS_BASE * pf
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=lp_v, y=lq_v, mode='markers',
                name='Точки T3',
                marker=dict(color=BLUE, size=14, line=dict(color='white', width=2))))
            fig3.add_trace(go.Scatter(x=pf, y=qf, mode='lines',
                name=f'Регресія (ε={EPS_BASE:.2f}, R²={EPS_R2:.3f})',
                line=dict(color=RED, width=2.5, dash='dash')))
            fig3.update_layout(height=380,
                xaxis=dict(title="ln(ціна/рік)", showgrid=True, gridcolor='#eee'),
                yaxis=dict(title="ln(WTP weighted)", showgrid=True, gridcolor='#eee'),
                plot_bgcolor='white', legend=dict(x=0.25, y=0.98,
                    bgcolor='rgba(255,255,255,0.9)', bordercolor='#ddd', borderwidth=1),
                margin=dict(t=15, b=50, l=65, r=15))
            st.plotly_chart(fig3, use_container_width=True)

    with tab_mc:
        st.subheader("Звідки беруться граничні витрати MC?")
        st.markdown('<span class="src-badge">T1</span> Джерело: T1_programs_prices.csv · колонка true_price_year', unsafe_allow_html=True)
        st.markdown("""<div class="expl">
        За замовчуванням MC = <code>true_price_year</code> з таблиці T1.<br><br>
        <strong>Що це таке:</strong> «справжня ціна» — оцінка повних витрат KSE на одного студента за рік,
        розрахована внутрішньо (включаючи викладачів, адміністрацію, інфраструктуру).<br><br>
        <strong>Важлива відмінність:</strong> це <em>average cost proxy</em>, а не строгий <em>marginal cost</em>.
        Різниця: marginal cost = витрати на одного ДОДАТКОВОГО студента (без fixed costs).
        True_price_year включає і фіксовані витрати → може завищувати MC.<br><br>
        <strong>Висновок:</strong> для стратегічного аналізу підходить як перший наближення.
        Для точнішого розрахунку потрібен cost breakdown по програмах.
        </div>""", unsafe_allow_html=True)
        t1b_show = t1[t1['degree']=='bachelor'][['program','price_2025_year','true_price_year','enrollment_plan']].copy()
        t1b_show.columns = ['Програма','Ціна 2025 грн/рік','MC (true_price) грн/рік','План набору']
        st.dataframe(t1b_show, hide_index=True, use_container_width=True)
        st.markdown("""<div class="good">
        ✅ <strong>MC можна змінити вручну в калькуляторі</strong> для будь-якої програми.
        Після отримання реального cost breakdown — підставляй туди.
        </div>""", unsafe_allow_html=True)

    with tab_infl:
        st.subheader("Інфляційна поправка: навіщо і коли")
        st.markdown("""<div class="expl">
        <strong>Default = 0%</strong>, бо основні ціни в T1 вже задані в цінах 2025 року.<br><br>
        <strong>Навіщо взагалі є цей параметр:</strong><br>
        Якщо порівнюєш поточні ціни з <em>історичними витратами</em> (наприклад, MC оцінювався у 2022 р.),
        треба привести їх до спільної бази року.<br><br>
        <strong>Формула приведення:</strong>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="formula">X₂₀₂₅ = X_t × (1 + π)^(2025 − t)</div>', unsafe_allow_html=True)
        st.markdown("""<div class="expl">
        де π — річна інфляція, t — рік вихідного значення.<br><br>
        <strong>Приклад:</strong> MC оцінений у 2022 = 100k, π = 10%:<br>
        MC₂₀₂₅ = 100k × (1.10)³ = 133.1k<br><br>
        <strong>Що НЕ робить ця кнопка:</strong> не «автоматично підвищує ціну на інфляцію».
        Це лише інструмент нормалізації дат.
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="note">⚠️ Якщо не впевнений — лишай 0%. Краще явно задати MC в цінах 2025, ніж автоматично застосовувати поправку.</div>', unsafe_allow_html=True)

    with tab_ex:
        st.subheader("Числовий приклад: Право")
        st.markdown('<span class="src-badge">T1</span><span class="src-badge-g">P1</span><span class="src-badge">P2</span>', unsafe_allow_html=True)
        st.markdown("""<div class="expl">
        Покроковий розрахунок на реальних даних для спеціальності <strong>«Право»</strong>.
        </div>""", unsafe_allow_html=True)

        # Real values from data
        ex_price    = 440000.0   # price_2025_year
        ex_mc       = 301000.0   # true_price_year
        ex_q_total  = 38.0       # p1 q2025
        ex_contract = 50.0       # p2 contract_2025
        ex_share    = 0.40       # 40%
        ex_q_pay    = ex_contract * ex_share  # = 20
        ex_eps      = abs(EPS_BASE)
        ex_pstar    = ex_mc * ex_eps / (ex_eps - 1)
        ex_q_opt_t  = ex_q_total * (ex_pstar / ex_price) ** (-ex_eps)
        ex_q_opt_p  = ex_q_pay   * (ex_pstar / ex_price) ** (-ex_eps)
        ex_rev_cur  = ex_price  * ex_q_pay
        ex_rev_opt  = ex_pstar  * ex_q_opt_p
        ex_pr_cur   = (ex_price - ex_mc) * ex_q_pay
        ex_pr_opt   = (ex_pstar - ex_mc) * ex_q_opt_p

        col_ex1, col_ex2 = st.columns([1, 1])
        with col_ex1:
            st.markdown("**Вхідні дані:**")
            ex_data = {
                'Параметр': ['Поточна ціна P_cur', 'Граничні витрати MC', 'Студентів всього Q_total',
                             'Контрактників contract_2025', 'Частка full-pay',
                             'Full-pay еквівалент Q_pay', 'Еластичність |ε|'],
                'Значення': [f'{ex_price/1000:.0f}k грн/рік', f'{ex_mc/1000:.0f}k грн/рік',
                             f'{ex_q_total:.0f}', f'{ex_contract:.0f}', f'{ex_share*100:.0f}%',
                             f'{ex_q_pay:.0f} (= {ex_contract:.0f} × {ex_share*100:.0f}%)',
                             f'{ex_eps:.2f}'],
                'Джерело': ['T1', 'T1 true_price_year', 'P1', 'P2', 'Ввід користувача',
                            'Розраховано', 'T3 регресія'],
            }
            st.dataframe(pd.DataFrame(ex_data), hide_index=True, use_container_width=True)

        with col_ex2:
            st.markdown("**Покрокові розрахунки:**")
            steps = [
                ("Крок 1", "P* = MC × |ε| / (|ε| − 1)",
                 f"P* = {ex_mc/1000:.0f}k × {ex_eps:.2f} / {ex_eps-1:.2f} = <strong>{ex_pstar/1000:.0f}k грн/рік</strong>"),
                ("Крок 2", "Q_pay(P*) = Q_pay × (P*/P_cur)^ε",
                 f"Q_pay(P*) = {ex_q_pay:.0f} × ({ex_pstar/1000:.0f}/{ex_price/1000:.0f})^(−{ex_eps:.2f}) = <strong>{ex_q_opt_p:.1f}</strong>"),
                ("Крок 3", "Revenue_cur = P_cur × Q_pay",
                 f"TR_cur = {ex_price/1000:.0f}k × {ex_q_pay:.0f} = <strong>{ex_rev_cur/1e6:.2f}M грн</strong>"),
                ("Крок 4", "Revenue_opt = P* × Q_pay(P*)",
                 f"TR_opt = {ex_pstar/1000:.0f}k × {ex_q_opt_p:.1f} = <strong>{ex_rev_opt/1e6:.2f}M грн</strong>"),
                ("Крок 5", "Profit_cur = (P_cur − MC) × Q_pay",
                 f"π_cur = ({ex_price/1000:.0f}k−{ex_mc/1000:.0f}k) × {ex_q_pay:.0f} = <strong>{ex_pr_cur/1e6:.2f}M грн</strong>"),
                ("Крок 6", "Profit_opt = (P* − MC) × Q_pay(P*)",
                 f"π_opt = ({ex_pstar/1000:.0f}k−{ex_mc/1000:.0f}k) × {ex_q_opt_p:.1f} = <strong>{ex_pr_opt/1e6:.2f}M грн</strong>"),
            ]
            for step, formula, calc in steps:
                st.markdown(f"""
                <div style="background:white;border:1px solid #e0e6f5;border-radius:8px;
                     padding:10px 14px;margin:5px 0;">
                <span style="color:{ORANGE};font-weight:700;font-size:12px;">{step}</span>
                <span style="font-family:monospace;font-size:13px;color:{BLUE};margin-left:8px;">{formula}</span><br>
                <span style="font-size:14px;margin-left:0;">{calc}</span>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Підсумок:**")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(kpi("Оптимальна ціна P*", f"{ex_pstar/1000:.0f}k грн/рік",
                           f"+{(ex_pstar-ex_price)/ex_price*100:.0f}% від поточної", "kpi-green"),
                        unsafe_allow_html=True)
        with c2:
            st.markdown(kpi("Δ Виручка", f"{(ex_rev_opt-ex_rev_cur)/1e6:+.2f}M грн/рік",
                           f"{ex_rev_cur/1e6:.2f}M → {ex_rev_opt/1e6:.2f}M",
                           "kpi-green" if ex_rev_opt > ex_rev_cur else "kpi-red"),
                        unsafe_allow_html=True)
        with c3:
            st.markdown(kpi("Δ Прибуток", f"{(ex_pr_opt-ex_pr_cur)/1e6:+.2f}M грн/рік",
                           f"{ex_pr_cur/1e6:.2f}M → {ex_pr_opt/1e6:.2f}M",
                           "kpi-green" if ex_pr_opt > ex_pr_cur else "kpi-red"),
                        unsafe_allow_html=True)
        oob_ex = ex_pstar < SURVEY_MIN or ex_pstar > SURVEY_MAX
        if oob_ex:
            st.markdown(f'<div class="warn">⚠️ P* = {ex_pstar/1000:.0f}k за 4 роки ({ex_pstar/4/1000:.0f}k/рік) виходить за survey-діапазон ({SURVEY_MIN/1000:.0f}k–{SURVEY_MAX/1000:.0f}k за 4 роки). Оцінка базується на екстраполяції кривої.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# РОЗШИРЕНІ ДАНІ
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔧 Розширені дані":
    st.title("🔧 Розширені дані — всі джерела")
    st.caption("Оригінальні таблиці максимально близько до raw вигляду")

    SOURCE_META = {
        "P1 · Зарахування":    ("pivot1_enrolled_by_spec_year.csv",
            "Реально зараховані студенти по роках. INPUT у калькулятор як Q_anchor.",
            "kpi-green"),
        "P2 · Бюджет/контракт":("pivot2_budget_contract.csv",
            "Всі хто з'явився в рейтингу (не лише зараховані). Бюджет + контракт split. "
            "INPUT: contract_2025 для Q_pay розрахунку.", "kpi-green"),
        "P3 · Пріоритети":     ("pivot3_priority_enrolled.csv",
            "Середній пріоритет KSE серед зарахованих. Diagnostic — не входить в калькулятор.", "kpi-gray"),
        "P4 · Бали":           ("pivot4_median_score_enrolled.csv",
            "Медіанний NMT бал зарахованих по роках. Diagnostic — тренд якості вступників.", "kpi-gray"),
        "T1 · Ціни KSE":       ("T1_programs_prices.csv",
            "Поточні ціни, пропозиції, true_price_year (MC proxy), enrollment_plan. "
            "INPUT: price_cur, mc_default, enrollment_plan у калькулятор.", "kpi-green"),
        "T2 · WTP агрегат":    ("T2_survey_agg.csv",
            "Агреговані WTP-частки по ціновим точкам (студенти KSE + абітурієнти). Diagnostic.", "kpi-gray"),
        "T2 · WTP individual": ("T2_survey_individual.csv",
            "Індивідуальні WTP-відповіді з демографікою (дохід, регіон, тип міста). "
            "Корисно для сегментації попиту. Наразі не входить в базову модель.", "kpi-gray"),
        "T3 · WTP абітурієнти":("T3_abit_wtp.csv",
            "WTP абітурієнтів n=108 по 4 ціновим точкам. INPUT: основа для ε в калькуляторі.", "kpi-green"),
    }

    tabs = st.tabs(list(SOURCE_META.keys()))
    base = str(HERE) + "/"

    for tab_obj, (tab_name, (fname, desc, badge_cls)) in zip(tabs, SOURCE_META.items()):
        with tab_obj:
            # Badge
            badge_color = GREEN if badge_cls == "kpi-green" else "#bbb"
            role = "INPUT у модель" if badge_cls == "kpi-green" else "Diagnostic"
            st.markdown(f'<span style="background:{badge_color};color:white;border-radius:4px;padding:2px 10px;font-size:12px;font-weight:600;">{role}</span> &nbsp; <code style="font-size:13px;">{fname}</code>', unsafe_allow_html=True)
            st.markdown(f'<div class="expl">{desc}</div>', unsafe_allow_html=True)

            try:
                df_raw = pd.read_csv(base + fname)
                st.markdown(f"**{len(df_raw)} рядків · {len(df_raw.columns)} колонок**")
                st.dataframe(df_raw, hide_index=True, use_container_width=True)

                # Extra viz for key sources
                if fname == "pivot1_enrolled_by_spec_year.csv":
                    ycols = [c for c in df_raw.columns if c != 'spec_name']
                    pm = df_raw.melt(id_vars='spec_name', value_vars=ycols, var_name='year', value_name='enrolled').dropna()
                    pm['year'] = pm['year'].astype(int)
                    fig = go.Figure()
                    for i, s in enumerate(df_raw['spec_name']):
                        d = pm[pm['spec_name']==s]
                        fig.add_trace(go.Scatter(x=d['year'], y=d['enrolled'], mode='lines+markers',
                            name=s, line=dict(color=PALETTE[i%len(PALETTE)], width=2), marker=dict(size=7)))
                    fig.update_layout(height=380, plot_bgcolor='white',
                        xaxis=dict(title="Рік", showgrid=True, gridcolor='#eee', dtick=1),
                        yaxis=dict(title="Зараховано", showgrid=True, gridcolor='#eee'),
                        legend=dict(orientation='h', y=-0.35, font=dict(size=11)),
                        margin=dict(t=20, b=155, l=65, r=20))
                    st.plotly_chart(fig, use_container_width=True)

                elif fname == "T3_abit_wtp.csv":
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=df_raw['price_year']/1000, y=df_raw['pct_yes']*100,
                        name='«Так»', marker_color=BLUE, width=3.5))
                    fig.add_trace(go.Bar(x=df_raw['price_year']/1000, y=df_raw['pct_maybe']*100,
                        name='«Можливо»', marker_color=LIGHT, width=3.5))
                    fig.add_trace(go.Bar(x=df_raw['price_year']/1000, y=df_raw['pct_no']*100,
                        name='«Ні»', marker_color='#ddd', width=3.5))
                    fig.update_layout(barmode='stack', height=340, plot_bgcolor='white',
                        xaxis=dict(title="Ціна грн/рік (тис.)", showgrid=True, gridcolor='#eee'),
                        yaxis=dict(title="%", showgrid=True, gridcolor='#eee'),
                        legend=dict(x=0.65, y=0.98, bgcolor='rgba(255,255,255,0.9)', bordercolor='#ddd', borderwidth=1),
                        margin=dict(t=15, b=40, l=55, r=15))
                    st.plotly_chart(fig, use_container_width=True)

                elif fname == "T2_survey_individual.csv":
                    st.markdown(f"**Колонки:** {', '.join(df_raw.columns.tolist())}")
                    if 'income' in df_raw.columns and 'wtp_score' in df_raw.columns:
                        inc_agg = df_raw.groupby('income')['wtp_score'].mean().reset_index()
                        inc_agg.columns = ['Рівень доходу', 'Сер. WTP score']
                        inc_agg = inc_agg.dropna()
                        if len(inc_agg) > 0:
                            fig = go.Figure(go.Bar(
                                x=inc_agg['Сер. WTP score'],
                                y=inc_agg['Рівень доходу'].str[:40],
                                orientation='h', marker_color=BLUE,
                                text=inc_agg['Сер. WTP score'].round(2), textposition='outside'))
                            fig.update_layout(height=300, plot_bgcolor='white',
                                xaxis=dict(title="Сер. WTP score", showgrid=True, gridcolor='#eee'),
                                yaxis=dict(showgrid=False),
                                margin=dict(t=15, b=30, l=10, r=60))
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption("WTP score: 1.0 = точно так, 0.5 = можливо, 0.0 = ні")

                elif fname == "pivot3_priority_enrolled.csv":
                    pc = [c for c in df_raw.columns if 'avg_priority' in c]
                    yp = [int(c.split('_')[-1]) for c in pc]
                    rows3 = []
                    for _, r in df_raw.iterrows():
                        for col, yr in zip(pc, yp):
                            v = r[col]
                            if not pd.isna(v):
                                rows3.append({'spec': r['spec_name'], 'year': yr, 'priority': v})
                    p3df = pd.DataFrame(rows3)
                    if len(p3df) > 0:
                        fig = go.Figure()
                        for i, s in enumerate(df_raw['spec_name']):
                            d = p3df[p3df['spec']==s]
                            if len(d) > 0:
                                fig.add_trace(go.Scatter(x=d['year'], y=d['priority'],
                                    mode='lines+markers', name=s,
                                    line=dict(color=PALETTE[i%len(PALETTE)], width=2),
                                    marker=dict(size=7)))
                        fig.add_hline(y=1.5, line_dash='dash', line_color='#aaa',
                                      annotation_text="Переважно 1-й вибір",
                                      annotation_position="bottom right",
                                      annotation_font_size=10)
                        fig.update_layout(height=380, plot_bgcolor='white',
                            xaxis=dict(title="Рік", showgrid=True, gridcolor='#eee', dtick=1),
                            yaxis=dict(title="Сер. пріоритет", showgrid=True, gridcolor='#eee'),
                            legend=dict(orientation='h', y=-0.35, font=dict(size=11)),
                            margin=dict(t=20, b=155, l=60, r=20))
                        st.plotly_chart(fig, use_container_width=True)

                elif fname == "pivot4_median_score_enrolled.csv":
                    sc_cols = [c for c in df_raw.columns if c not in ['spec_name','score_trend_23_25']]
                    pm4 = df_raw.melt(id_vars='spec_name', value_vars=sc_cols,
                                      var_name='year', value_name='score').dropna()
                    pm4['year'] = pm4['year'].astype(int)
                    fig = go.Figure()
                    for i, s in enumerate(df_raw['spec_name']):
                        d = pm4[pm4['spec_name']==s]
                        fig.add_trace(go.Scatter(x=d['year'], y=d['score'], mode='lines+markers',
                            name=s, line=dict(color=PALETTE[i%len(PALETTE)], width=2),
                            marker=dict(size=7)))
                    fig.update_layout(height=380, plot_bgcolor='white',
                        xaxis=dict(title="Рік", showgrid=True, gridcolor='#eee', dtick=1),
                        yaxis=dict(title="Медіана балів NMT", showgrid=True, gridcolor='#eee',
                                   range=[140, 200]),
                        legend=dict(orientation='h', y=-0.35, font=dict(size=11)),
                        margin=dict(t=20, b=155, l=65, r=20))
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Помилка читання {fname}: {e}")