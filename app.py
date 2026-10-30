import streamlit as st
import pandas as pd
from pathlib import Path
from engine import validate, simulate, sgp_portfolio, american_to_decimal

import requests

@st.cache_data(ttl=1800)
def fetch_live_odds(sport_key):
    response = requests.get(
        "https://api.theoddsapi.com/odds/",
        headers={"x-api-key": st.secrets["ODDS_API_KEY"]},
        params={"sport_key": sport_key},
        timeout=15
    )
    response.raise_for_status()
    return response.json()
st.set_page_config(page_title='EdgeLab | Prop Optimizer', page_icon='📊', layout='wide')
st.markdown('''<style>
.block-container {padding-top:1.2rem; max-width:1100px}
.stMetric {background:#17263a; border:1px solid #33465e; padding:12px; border-radius:12px}
@media(max-width:600px){h1{font-size:1.7rem!important} [data-testid="stMetricValue"]{font-size:1.35rem}}
</style>''', unsafe_allow_html=True)
st.title('📊 EdgeLab — Sports Prop Optimizer')
st.caption('NFL • NBA • MLB • NHL | Educational research tool • No live sportsbook feed')

with st.sidebar:
    st.header('Settings')
    n = st.select_slider('Simulation draws', options=[2000,5000,10000,20000,50000],value=20000)
    seed = st.number_input('Random seed',min_value=0,max_value=999999,value=42)
    st.markdown('**Shared variance factors** (illustrative correlation assumptions)')
    game_rho = st.slider('Same game',0.0,0.35,0.12,0.01)
    team_rho = st.slider('Same team',0.0,0.35,0.16,0.01)
    player_rho = st.slider('Same player',0.0,0.35,0.22,0.01)
    st.caption('These are variance shares, not measured correlations. Calibrate using historical data.')

sample = Path(__file__).parent / 'sample_props.csv'
st.subheader('1 · Load player props')
upload = st.file_uploader('Upload CSV (or use included fictional example)', type='csv')
try:
    raw = pd.read_csv(upload if upload is not None else sample)
    df = validate(raw)
except Exception as e:
    st.error(f'Invalid CSV: {e}')
    st.stop()

sport = st.selectbox('Sport', ['ALL']+sorted(df['sport'].unique().tolist()))
if sport != 'ALL': df = df[df['sport']==sport].copy().reset_index(drop=True)
if df.empty: st.warning('No props for selected sport.'); st.stop()
with st.expander('See loaded input data'):
    st.dataframe(df[['sport','game_id','player','team','market','side','line','american_odds','projection','std_dev']],hide_index=True,use_container_width=True)

if game_rho+team_rho+player_rho>=1:
    st.error('Shared variance shares must total less than 1.'); st.stop()

st.subheader('2 · Simulated player prop analysis')
result,wins = simulate(df,n=int(n),seed=int(seed),game_rho=game_rho,team_rho=team_rho,player_rho=player_rho)
cols=st.columns(3)
cols[0].metric('Props evaluated',len(result))
cols[1].metric('Positive model EV',int((result['ev_per_dollar']>0).sum()))
cols[2].metric('Simulation draws',f'{n:,}')
show = result[['sport','game_id','player','market','side','line','american_odds','model_prob','implied_prob','ev_per_dollar','fair_odds']].sort_values('ev_per_dollar',ascending=False)
# Display percentages in human-readable units.
display = show.copy()
display['model_prob'] *= 100
display['implied_prob'] *= 100
st.dataframe(display,hide_index=True,use_container_width=True,column_config={
 'model_prob':st.column_config.NumberColumn('Model hit %',format='%.1f%%'),
 'implied_prob':st.column_config.NumberColumn('Implied %',format='%.1f%%'),
 'ev_per_dollar':st.column_config.NumberColumn('EV / $1',format='$%.3f')})
st.download_button('Download analysis CSV',result.to_csv(index=False).encode(),'prop_analysis.csv','text/csv')

st.subheader('3 · Correlated same-game parlay builder')
games = result['game_id'].unique().tolist()
game = st.selectbox('Select game',games)
c1,c2 = st.columns(2)
min_legs = c1.selectbox('Minimum legs',[2,3,4],index=0)
max_legs = c2.selectbox('Maximum legs',[2,3,4],index=1)
if min_legs>max_legs:
    st.warning('Minimum legs must not exceed maximum legs.')
else:
    picks = sgp_portfolio(result,wins,game,min_legs,max_legs,top_k=15)
    if picks.empty: st.info('Not enough distinct eligible props for this game.')
    else:
        visible=picks.copy()
        visible['joint_probability'] *= 100
        visible['independent_probability'] *= 100
        st.dataframe(visible,hide_index=True,use_container_width=True,column_config={
            'joint_probability':st.column_config.NumberColumn('Joint hit %',format='%.1f%%'),
            'independent_probability':st.column_config.NumberColumn('Independent %',format='%.1f%%'),
            'correlation_lift':st.column_config.NumberColumn('Correlation lift',format='%.2fx')})
        st.download_button('Download SGP candidates',picks.to_csv(index=False).encode(),'sgp_candidates.csv','text/csv')
        st.markdown('**Check value against the actual sportsbook combined SGP quote**')
        chosen = st.selectbox('Candidate',range(len(picks)),format_func=lambda i: f'#{i+1}: {picks.loc[i,"legs"][:90]}')
        quote = st.number_input('Sportsbook combined American odds (not multiplied leg odds)',value=300,step=10)
        p = float(picks.loc[chosen,'joint_probability'])
        d = american_to_decimal(quote)
        st.metric('Estimated SGP EV per $1',f'${p*d-1:+.3f}')
        st.caption(f'Model joint hit rate {p:.1%}; quote implied probability {1/d:.1%}. EV excludes pushes, voids, taxes and limits.')

st.divider()
st.warning('Important: Sample data is fictional. Normal-distribution models and generic positive correlations are simplistic, especially for MLB/NHL counts, alternate lines, injury changes, and opposing-player relationships. Simulated hit rates are not verified or calibrated predictions. No odds are fetched automatically. Sportsbooks may reject SGP combinations or price them differently. Do not treat this as a guarantee or staking advice.')
