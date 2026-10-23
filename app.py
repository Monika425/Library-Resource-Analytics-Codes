from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
DATA_PATHS = [ROOT / 'output' / 'cleaned_library_circulation.csv', ROOT / 'dataset' / 'library_circulation.csv']
FORECAST_PATH = ROOT / 'output' / 'future_demand_forecast.csv'

st.set_page_config(page_title='Library Intelligence | Resource Analytics', page_icon='📚', layout='wide', initial_sidebar_state='expanded')

st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#18213b; --muted:#68758e; --line:#e8eaf4; --purple:#7657e8; --pink:#ed6aa7; --teal:#16b8a6; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:linear-gradient(135deg,#f7f7ff 0%,#f4fbff 52%,#fff7fb 100%); color:var(--ink); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#241c50 0%,#35266f 52%,#174c69 100%); border-right:0; }
[data-testid="stSidebar"] * { color:#f5f4ff !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#d4d0f4 !important; }
[data-testid="stSidebar"] [data-testid="stButton"] button { width:100%; justify-content:flex-start; text-align:left; background:rgba(255,255,255,.075); border:1px solid rgba(255,255,255,.10); color:#f5f4ff; border-radius:13px; padding:12px 14px; margin:4px 0; min-height:46px; font-weight:700; transition:all .18s ease; }
[data-testid="stSidebar"] [data-testid="stButton"] button:hover { background:rgba(255,255,255,.18); border-color:rgba(255,255,255,.30); color:#fff; transform:translateX(2px); }
[data-testid="stSidebar"] [data-testid="stButton"] button:focus { background:linear-gradient(100deg,#7a63ed,#ba68d8); color:white; border-color:rgba(255,255,255,.35); }
.block-container { padding-top:1.5rem; padding-bottom:2.5rem; max-width:1600px; }
.hero { background:linear-gradient(115deg,#392879 0%,#6753d7 42%,#df69a5 100%); padding:30px 32px; border-radius:24px; color:white; margin-bottom:22px; box-shadow:0 14px 34px rgba(91,66,177,.20); position:relative; overflow:hidden; }
.hero:after { content:'✦'; position:absolute; right:40px; top:-45px; font-size:190px; color:rgba(255,255,255,.10); transform:rotate(12deg); }
.hero h1 { font-family:'Manrope',sans-serif; color:white; font-size:2.1rem; line-height:1.2; margin:5px 0 8px 0; font-weight:800; position:relative; z-index:1; }
.hero p { color:#f1eaff; margin:0; font-size:1rem; max-width:850px; position:relative; z-index:1; }
.eyebrow { text-transform:uppercase; letter-spacing:2px; font-size:.72rem; font-weight:800; color:#ffe1f0; position:relative; z-index:1; }
.kpi { background:linear-gradient(145deg,#fff 0%,#fbfaff 100%); border:1px solid #e9e5fb; border-radius:19px; padding:19px 20px; min-height:124px; box-shadow:0 7px 20px rgba(55,43,112,.055); border-top:4px solid #8c72ed; }
.kpi:nth-child(2) { border-top-color:#21b8ad; }
.kpi-label { font-size:.76rem; font-weight:800; letter-spacing:.45px; color:#75809a; text-transform:uppercase; }
.kpi-value { font-family:'Manrope',sans-serif; font-size:1.8rem; font-weight:800; color:#28214f; margin:7px 0 3px; }
.kpi-foot { font-size:.78rem; color:#8b91a8; }
.section-title { font-family:'Manrope',sans-serif; font-size:1.15rem; font-weight:800; color:#28214f; margin:8px 0 2px; }
.section-sub { color:#7d849e; font-size:.83rem; margin-bottom:12px; }
.panel { background:linear-gradient(145deg,#fff 0%,#fcfbff 100%); border:1px solid #ebe7fb; border-radius:18px; padding:18px 20px; box-shadow:0 5px 18px rgba(55,43,112,.035); }
.pill { display:inline-block; background:#eee9ff; color:#6447d1; border-radius:20px; padding:5px 10px; font-size:.73rem; font-weight:800; }
.note { background:linear-gradient(100deg,#fff7e7,#fff1f7); border:1px solid #f8dce9; padding:13px 15px; border-radius:13px; color:#80506b; font-size:.84rem; }
div[data-testid="stMetric"] { background:white; padding:16px; border:1px solid #e9e5fb; border-radius:15px; box-shadow:0 4px 14px rgba(55,43,112,.04); }
.stButton button, .stDownloadButton button { border-radius:11px; border:0; background:linear-gradient(100deg,#7055df,#a05be4); color:white; font-weight:700; }
.stButton button:hover, .stDownloadButton button:hover { color:white; border:0; box-shadow:0 6px 16px rgba(112,85,223,.25); }
[data-testid="stExpander"] { background:rgba(255,255,255,.82); border:1px solid #e7e2fa; border-radius:15px; }
[data-testid="stDataFrame"] { border:1px solid #e9e5fb; border-radius:13px; overflow:hidden; }
footer {visibility:hidden;}
</style>
''', unsafe_allow_html=True)
@st.cache_data
def load_data():
    path = next((p for p in DATA_PATHS if p.exists()), None)
    if path is None:
        raise FileNotFoundError('Could not find cleaned_library_circulation.csv in output/ or library_circulation.csv in dataset/.')
    df = pd.read_csv(path)
    df['month_date'] = pd.to_datetime(df['month_date'], errors='coerce')
    df['circulation_format_total'] = pd.to_numeric(df['circulation_format_total'], errors='coerce').fillna(0)
    for col in ['resource_type','location_acronym','circulation_format','audience']:
        if col not in df: df[col] = 'Unknown'
        df[col] = df[col].fillna('Unknown').astype(str)
    df = df.dropna(subset=['month_date'])
    df['month_date'] = df['month_date'].dt.to_period('M').dt.to_timestamp()
    return df

@st.cache_data
def load_forecast():
    if not FORECAST_PATH.exists(): return pd.DataFrame(columns=['month_date','resource_type','predicted_demand'])
    fc = pd.read_csv(FORECAST_PATH)
    fc['month_date'] = pd.to_datetime(fc['month_date'], errors='coerce')
    fc['predicted_demand'] = pd.to_numeric(fc['predicted_demand'], errors='coerce')
    return fc.dropna(subset=['month_date','predicted_demand'])

def compact(n):
    n = float(n)
    if abs(n) >= 1_000_000: return f'{n/1_000_000:.2f}M'
    if abs(n) >= 1_000: return f'{n/1_000:.1f}K'
    return f'{n:,.0f}'

def fig_layout(fig, height=330):
    fig.update_layout(height=height, margin=dict(l=8,r=8,t=30,b=8), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family='DM Sans',color='#607086',size=11), legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='left',x=0), hoverlabel=dict(bgcolor='#152743',font_color='white'))
    fig.update_xaxes(showgrid=False, linecolor='#e8edf4', tickfont=dict(size=10))
    fig.update_yaxes(gridcolor='#edf1f6', zeroline=False, tickfont=dict(size=10))
    return fig

try:
    df = load_data()
except Exception as e:
    st.error(str(e)); st.stop()
forecast = load_forecast()

# Sidebar: app-style navigation buttons, not radio controls.
PAGES = {
    'overview': '✨  Overview',
    'history': '📅  Historical analytics',
    'forecast': '🔮  Demand forecast',
    'stock': '🛍️  Purchase & stock planner',
}
if 'active_page' not in st.session_state:
    st.session_state.active_page = 'overview'

def go_to(page_key):
    st.session_state.active_page = page_key

with st.sidebar:
    st.markdown('''<div style="padding:12px 4px 22px"><div style="font-size:2.2rem">📚</div><div style="font-family:Manrope;font-size:1.16rem;font-weight:800;line-height:1.15">LIBRARY<br>INTELLIGENCE</div><div style="font-size:.72rem;color:#d4d0f4;margin-top:8px;letter-spacing:1px">RESOURCE ANALYTICS</div></div>''', unsafe_allow_html=True)
    st.markdown('<div style="font-size:.72rem;letter-spacing:1.5px;font-weight:800;color:#c7c0f5;padding:0 0 8px 4px">WORKSPACE MENU</div>', unsafe_allow_html=True)
    for key, label in PAGES.items():
        if st.session_state.active_page == key:
            st.markdown(f'<div style="display:flex;align-items:center;gap:10px;background:linear-gradient(100deg,#7a63ed,#ba68d8);border:1px solid #ffffff45;border-radius:13px;padding:13px 14px;margin:4px 0;color:white;font-weight:800;box-shadow:0 7px 18px #160e3d55">{label}</div>', unsafe_allow_html=True)
        elif st.button(label, key=f'nav_{key}', use_container_width=True):
            go_to(key)
            st.rerun()
    st.markdown('<div style="margin-top:28px;padding:13px;border-radius:14px;background:rgba(255,255,255,.09);font-size:.78rem;color:#e1ddff;line-height:1.5">Explore circulation patterns, forecast demand, and turn insights into smarter resource planning.</div>', unsafe_allow_html=True)

page_key = st.session_state.active_page
page = PAGES[page_key]

min_date, max_date = df['month_date'].min().date(), df['month_date'].max().date()
resource_options = sorted(df['resource_type'].unique())
location_options = sorted(df['location_acronym'].unique())
audience_options = sorted(df['audience'].unique())
with st.expander('🎛️  Refine your analytics · date, resource, location and audience', expanded=False):
    f1, f2 = st.columns([1.2, 1])
    with f1:
        date_range = st.date_input('Circulation period', value=(min_date,max_date), min_value=min_date, max_value=max_date)
    with f2:
        chosen_resources = st.multiselect('Resource categories', resource_options, default=resource_options, placeholder='Choose resource categories')
    f3, f4 = st.columns(2)
    with f3:
        chosen_locations = st.multiselect('Library locations', location_options, default=location_options, placeholder='Choose locations')
    with f4:
        chosen_audiences = st.multiselect('Audience', audience_options, default=audience_options, placeholder='Choose audiences')
if isinstance(date_range, (tuple,list)) and len(date_range)==2:
    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start_date, end_date = pd.Timestamp(min_date), pd.Timestamp(max_date)

filtered = df[(df['month_date']>=start_date.to_period('M').to_timestamp()) & (df['month_date']<=end_date.to_period('M').to_timestamp())]
if chosen_resources: filtered = filtered[filtered['resource_type'].isin(chosen_resources)]
else: filtered = filtered.iloc[0:0]
if chosen_locations: filtered = filtered[filtered['location_acronym'].isin(chosen_locations)]
else: filtered = filtered.iloc[0:0]
if chosen_audiences: filtered = filtered[filtered['audience'].isin(chosen_audiences)]
else: filtered = filtered.iloc[0:0]

# Header
st.markdown(f'''<div class="hero"><div class="eyebrow">DATA WAREHOUSE · DECISION INTELLIGENCE</div><h1>Your Library, In Focus</h1><p>A colourful command centre for library performance, historical trends, demand forecasts and smarter stock decisions.</p><div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:18px"><span style="background:#ffffff20;border:1px solid #ffffff28;padding:6px 11px;border-radius:20px;font-size:.76rem">◷ {start_date.strftime('%b %Y')} — {end_date.strftime('%b %Y')}</span><span style="background:#ffffff20;border:1px solid #ffffff28;padding:6px 11px;border-radius:20px;font-size:.76rem">⌂ {len(chosen_locations)} locations selected</span><span style="background:#ffffff20;border:1px solid #ffffff28;padding:6px 11px;border-radius:20px;font-size:.76rem">↗ {len(chosen_resources)} resource categories</span></div></div>''', unsafe_allow_html=True)

if filtered.empty:
    st.warning('No records match these filters. Select at least one resource category, location, and audience, or widen the date range.')
    st.stop()

# KPIs
circulation = filtered['circulation_format_total'].sum()
records = len(filtered)
active_types = filtered['resource_type'].nunique()
active_locations = filtered['location_acronym'].nunique()
monthly_total = filtered.groupby('month_date')['circulation_format_total'].sum().sort_index()
if len(monthly_total)>1 and monthly_total.iloc[-2] != 0:
    growth = (monthly_total.iloc[-1]-monthly_total.iloc[-2])/monthly_total.iloc[-2]*100
    growth_text = f'{growth:+.1f}% vs prior month in selected period'
else: growth_text = 'Monthly trend appears below'
cols = st.columns(4)
metrics = [("TOTAL CIRCULATIONS",compact(circulation),'Across current filters'),('MONTHLY AVERAGE',compact(monthly_total.mean() if len(monthly_total) else 0),'Average circulation per month'),('RESOURCE CATEGORIES',str(active_types),'Categories represented'),('LIBRARY LOCATIONS',str(active_locations),'Locations represented')]
for c,(label,value,foot) in zip(cols,metrics):
    c.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-foot">{foot}</div></div>',unsafe_allow_html=True)

st.markdown('<div style="height:10px"></div>',unsafe_allow_html=True)
if page_key == 'overview':
    left,right = st.columns([1.65,1])
    with left:
        st.markdown('<div class="section-title">Circulation momentum</div><div class="section-sub">Monthly demand over the selected period</div>',unsafe_allow_html=True)
        trend = filtered.groupby('month_date',as_index=False)['circulation_format_total'].sum()
        fig=px.area(trend,x='month_date',y='circulation_format_total',markers=True)
        fig.update_traces(line=dict(color='#0d9488',width=3),fillcolor='rgba(13,148,136,.13)',marker=dict(size=5,color='#0d9488'),hovertemplate='%{x|%b %Y}<br>Circulations: %{y:,.0f}<extra></extra>')
        fig.update_layout(xaxis_title='',yaxis_title='Circulations')
        st.plotly_chart(fig_layout(fig,340),use_container_width=True,config={'displayModeBar':False})
    with right:
        st.markdown('<div class="section-title">Demand mix</div><div class="section-sub">Share by resource type</div>',unsafe_allow_html=True)
        mix=filtered.groupby('resource_type',as_index=False)['circulation_format_total'].sum().sort_values('circulation_format_total',ascending=False)
        fig=px.pie(mix,values='circulation_format_total',names='resource_type',hole=.68)
        fig.update_traces(textposition='inside',textinfo='percent',hovertemplate='%{label}<br>%{value:,.0f} circulations<extra></extra>')
        fig.update_layout(showlegend=True,legend=dict(orientation='h',y=-.15,x=0,font=dict(size=9)))
        st.plotly_chart(fig_layout(fig,340),use_container_width=True,config={'displayModeBar':False})
    left,right=st.columns(2)
    with left:
        st.markdown('<div class="section-title">Top resource categories</div><div class="section-sub">Ranked by circulation volume</div>',unsafe_allow_html=True)
        rank=filtered.groupby('resource_type',as_index=False)['circulation_format_total'].sum().sort_values('circulation_format_total',ascending=True)
        fig=px.bar(rank,x='circulation_format_total',y='resource_type',orientation='h',text='circulation_format_total')
        fig.update_traces(marker_color='#4777d5',texttemplate='%{x:,.0f}',textposition='outside',cliponaxis=False)
        fig.update_layout(xaxis_title='Circulations',yaxis_title='')
        st.plotly_chart(fig_layout(fig,330),use_container_width=True,config={'displayModeBar':False})
    with right:
        st.markdown('<div class="section-title">Location performance</div><div class="section-sub">Where resources are being used most</div>',unsafe_allow_html=True)
        loc=filtered.groupby('location_acronym',as_index=False)['circulation_format_total'].sum().sort_values('circulation_format_total',ascending=False).head(10).sort_values('circulation_format_total')
        fig=px.bar(loc,x='circulation_format_total',y='location_acronym',orientation='h',text='circulation_format_total')
        fig.update_traces(marker_color='#8b7be8',texttemplate='%{x:,.0f}',textposition='outside',cliponaxis=False)
        fig.update_layout(xaxis_title='Circulations',yaxis_title='Location')
        st.plotly_chart(fig_layout(fig,330),use_container_width=True,config={'displayModeBar':False})
    st.markdown('<div class="section-title">Signals worth your attention</div><div class="section-sub">Automatically generated from the filtered circulation data</div>',unsafe_allow_html=True)
    top=mix.sort_values('circulation_format_total',ascending=False)
    s1,s2,s3=st.columns(3)
    if len(top):
        s1.markdown(f'<div class="panel"><div class="pill">HIGHEST DEMAND</div><h3 style="margin:12px 0 4px;color:#172b4d">{top.iloc[0].resource_type}</h3><div style="color:#718096;font-size:.85rem">{top.iloc[0].circulation_format_total:,.0f} circulations in selected period</div></div>',unsafe_allow_html=True)
    if len(monthly_total):
        peak_date=monthly_total.idxmax()
        s2.markdown(f'<div class="panel"><div class="pill">PEAK MONTH</div><h3 style="margin:12px 0 4px;color:#172b4d">{peak_date.strftime("%b %Y")}</h3><div style="color:#718096;font-size:.85rem">{monthly_total.max():,.0f} total circulations</div></div>',unsafe_allow_html=True)
    if len(loc):
        s3.markdown(f'<div class="panel"><div class="pill">BUSIEST LOCATION</div><h3 style="margin:12px 0 4px;color:#172b4d">{loc.iloc[-1].location_acronym}</h3><div style="color:#718096;font-size:.85rem">{loc.iloc[-1].circulation_format_total:,.0f} circulations</div></div>',unsafe_allow_html=True)

elif page_key == 'history':
    st.markdown('<div class="section-title">Historical performance explorer</div><div class="section-sub">Compare resource categories, audiences, formats, and locations over time.</div>',unsafe_allow_html=True)
    hist1,hist2=st.columns([1,1])
    with hist1:
        metric_choice=st.selectbox('Break down by', ['Resource type','Location','Audience','Circulation format'])
    col_map={'Resource type':'resource_type','Location':'location_acronym','Audience':'audience','Circulation format':'circulation_format'}
    dim=col_map[metric_choice]
    with hist2:
        top_n=st.slider('Number of series to show',3,12,7)
    totals=filtered.groupby(dim)['circulation_format_total'].sum().sort_values(ascending=False)
    keep=totals.head(top_n).index.tolist()
    h=filtered[filtered[dim].isin(keep)].groupby(['month_date',dim],as_index=False)['circulation_format_total'].sum()
    fig=px.line(h,x='month_date',y='circulation_format_total',color=dim,markers=True)
    fig.update_layout(xaxis_title='',yaxis_title='Circulations',legend_title='')
    st.plotly_chart(fig_layout(fig,420),use_container_width=True,config={'displayModeBar':False})
    st.markdown('<div class="section-title">Period comparison</div><div class="section-sub">Monthly and category-level totals in the current selection</div>',unsafe_allow_html=True)
    pivot=filtered.pivot_table(index=dim,columns='month_date',values='circulation_format_total',aggfunc='sum',fill_value=0)
    pivot['TOTAL'] = pivot.sum(axis=1)
    pivot=pivot.sort_values('TOTAL',ascending=False).reset_index()
    pivot.columns=[c.strftime('%b %Y') if isinstance(c,pd.Timestamp) else c for c in pivot.columns]
    st.dataframe(pivot,use_container_width=True,hide_index=True,height=340)
    st.download_button('⬇ Download historical data (CSV)',filtered.to_csv(index=False).encode('utf-8'),file_name='library_historical_analytics.csv',mime='text/csv')

elif page_key == 'forecast':
    st.markdown('<div class="section-title">Forward demand outlook</div><div class="section-sub">Forecasts produced by the project’s Random Forest regression pipeline.</div>',unsafe_allow_html=True)
    if forecast.empty:
        st.warning('No forecast file found. Run `python ml/prediction.py` from the project folder to generate one.')
    else:
        fc=forecast.copy()
        if chosen_resources: fc=fc[fc['resource_type'].isin(chosen_resources)]
        else: fc=fc.iloc[0:0]
        if not fc.empty:
            fleft,fright=st.columns([1.6,1])
            with fleft:
                fchart=px.line(fc,x='month_date',y='predicted_demand',color='resource_type',markers=True)
                fchart.update_layout(xaxis_title='',yaxis_title='Predicted circulations',legend_title='')
                st.plotly_chart(fig_layout(fchart,360),use_container_width=True,config={'displayModeBar':False})
            with fright:
                future_totals=fc.groupby('month_date',as_index=False)['predicted_demand'].sum()
                fig=px.bar(future_totals,x='month_date',y='predicted_demand',text='predicted_demand')
                fig.update_traces(marker_color='#0d9488',texttemplate='%{y:,.0f}',textposition='outside',cliponaxis=False)
                fig.update_layout(xaxis_title='',yaxis_title='Predicted total')
                st.plotly_chart(fig_layout(fig,360),use_container_width=True,config={'displayModeBar':False})
            st.markdown('<div class="section-title">Forecast detail</div><div class="section-sub">Predicted monthly circulation volume by resource type</div>',unsafe_allow_html=True)
            showfc=fc.sort_values(['month_date','predicted_demand'],ascending=[True,False]).copy()
            showfc['month_date']=showfc['month_date'].dt.strftime('%b %Y')
            showfc['predicted_demand']=showfc['predicted_demand'].round().astype(int)
            st.dataframe(showfc.rename(columns={'month_date':'Forecast month','resource_type':'Resource type','predicted_demand':'Predicted demand'}),use_container_width=True,hide_index=True)
            st.caption('Forecasts are model estimates, not guaranteed outcomes. The included forecast file covers the three months after the historical data available when the model was last run.')
        else: st.info('No forecast rows match the selected resource categories.')

elif page_key == 'stock':
    st.markdown('<div class="section-title">Purchase & stock planning</div><div class="section-sub">Translate predicted circulation into a transparent replenishment priority list.</div>',unsafe_allow_html=True)
    plan1, plan2 = st.columns(2)
    with plan1:
        lead_months = st.slider('Procurement lead time (months)', 1, 6, 2)
    with plan2:
        buffer_pct = st.slider('Safety-stock buffer', 0, 50, 15, step=5)
    st.markdown('<div class="note"><b>Important:</b> the warehouse contains circulation, not actual inventory counts, order quantities, prices, or supplier lead times. Recommendations below are demand-based planning estimates. Enter current stock to estimate a stock gap; the app does not claim to know live stock.</div>',unsafe_allow_html=True)
    if forecast.empty:
        st.warning('Forecast data is missing. Generate forecasts with `python ml/prediction.py` first.')
    else:
        base_fc=forecast.copy()
        if chosen_resources: base_fc=base_fc[base_fc['resource_type'].isin(chosen_resources)]
        recent=filtered.groupby('resource_type')['circulation_format_total'].sum()
        fmonths=sorted(base_fc['month_date'].unique())
        horizon=base_fc[base_fc['month_date'].isin(fmonths[:lead_months])].groupby('resource_type')['predicted_demand'].sum()
        monthly_forecast=base_fc.groupby('resource_type')['predicted_demand'].mean()
        rec=pd.DataFrame({'resource_type':sorted(set(base_fc.resource_type))})
        rec['forecast_monthly_demand']=rec['resource_type'].map(monthly_forecast).fillna(0)
        rec['lead_time_demand']=rec['resource_type'].map(horizon).fillna(0)
        rec['safety_buffer']=rec['lead_time_demand']*buffer_pct/100
        rec['suggested_target_stock']=np.ceil(rec['lead_time_demand']+rec['safety_buffer']).astype(int)
        rec['recent_circulation']=rec['resource_type'].map(recent).fillna(0)
        rec['priority_score']=rec['forecast_monthly_demand']*(1+buffer_pct/100)
        rec['priority']=np.select([rec['forecast_monthly_demand']>=rec['forecast_monthly_demand'].quantile(.75),rec['forecast_monthly_demand']>=rec['forecast_monthly_demand'].median()],['High','Medium'],default='Monitor')
        st.markdown('**Add known on-hand quantities (optional)**')
        st.caption('Enter quantities only if you have verified current stock. Leave blank to keep the demand target as a planning benchmark.')
        stock_values={}
        with st.expander('Enter stock counts by resource type',expanded=False):
            for _,r in rec.iterrows():
                val=st.number_input(f"{r['resource_type']} on hand",min_value=0,step=1,value=0,key=f"stock_{r['resource_type']}")
                stock_values[r['resource_type']]=val
        rec['on_hand']=rec['resource_type'].map(stock_values).fillna(0).astype(int)
        rec['estimated_gap']=(rec['suggested_target_stock']-rec['on_hand']).clip(lower=0)
        rec['planning_action']=np.where(rec['estimated_gap']>0,'Review replenishment','Stock meets target')
        rec=rec.sort_values(['priority_score','forecast_monthly_demand'],ascending=False)
        k1,k2,k3=st.columns(3)
        k1.metric('Highest forecast demand',rec.iloc[0]['resource_type'] if len(rec) else '—')
        k2.metric('Demand target (lead time)',compact(rec['suggested_target_stock'].sum()))
        k3.metric('Estimated gap to target',compact(rec['estimated_gap'].sum()))
        view=rec[['resource_type','forecast_monthly_demand','lead_time_demand','safety_buffer','suggested_target_stock','on_hand','estimated_gap','priority','planning_action']].copy()
        view.columns=['Resource type','Avg monthly forecast','Lead-time demand','Safety buffer','Suggested target stock','On hand entered','Estimated gap','Priority','Planning action']
        for c in ['Avg monthly forecast','Lead-time demand','Safety buffer']: view[c]=view[c].round().astype(int)
        st.dataframe(view,use_container_width=True,hide_index=True)
        st.download_button('⬇ Download purchase planning (CSV)',view.to_csv(index=False).encode('utf-8'),file_name='library_purchase_stock_plan.csv',mime='text/csv')
        st.markdown('<div class="section-title">Recommended next steps</div>',unsafe_allow_html=True)
        high=rec[rec['priority']=='High']
        gap=rec[rec['estimated_gap']>0]
        a,b=st.columns(2)
        with a:
            if len(high): st.markdown('**Prioritise demand review**\n\n'+'\n'.join([f"- **{r.resource_type}** — about {r.forecast_monthly_demand:,.0f} circulations/month forecast" for r in high.itertuples()]))
            else: st.write('No high-priority demand category under the current filter.')
        with b:
            if len(gap): st.markdown('**Validate possible stock gaps**\n\n'+'\n'.join([f"- **{r.resource_type}** — estimated gap {r.estimated_gap:,.0f} units against the planning target" for r in gap.itertuples()]))
            else: st.write('No estimated stock gap using the entered quantities.')
        st.caption('Circulation counts are not necessarily one-to-one with purchasable units. Validate these planning quantities against actual inventory, replacement rates, budget, and vendor constraints before ordering.')

st.markdown('<div style="margin-top:24px;padding:15px 2px;border-top:1px solid #e2e8f0;color:#8a96a8;font-size:.76rem">LIBRARY INTELLIGENCE · Academic data warehouse project · Analytics are based on the included circulation dataset.</div>',unsafe_allow_html=True)
