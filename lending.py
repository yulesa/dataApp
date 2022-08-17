import streamlit as st
import pandas as pd
import requests
import asyncio
import aiohttp
import datetime

from datetime import datetime as dt
from subgrounds import Subgrounds
from aiocache import Cache
from aiocache import cached


import plotly.graph_objs as go
from plotly.subplots import make_subplots



#######################################################################################################



###### Main Page ######


st.set_page_config(layout="wide")
st.title('Messari: Protocol Analytics')
st.write('Yule Andrade  using Messari Subgraphs')


subgraphs = requests.get('https://subgraphs.messari.io/deployments.json')
subgraph_json = subgraphs.json()

protocols = ['aave',
             'abracadabra',
             'aurigami',
             'banker-joe',
             'bastion-protocol',
             'benqi',
             'compound',
             'cream-finance',
             'dforce',
             'ellipsis-finance',
             'geist-finance',
             'inverse-finance',
             'iron-bank',
             'liquity',
             'makerdao',
             'maple-finance',
             'moonwell',
             'qidao',
             'rari-fuse',
             'scream',
             'tectonic',
             'venus'
]
             
selection_layout = st.columns([2,3,3,1,1])

with selection_layout[0]: 
    protocol = st.selectbox(
        'Protocol:',
        protocols,
        index = 6,
        key='protocol')

versions = [k for k in list(subgraph_json['lending'].keys()) if protocol in k]

with selection_layout[1]: 
    version = st.multiselect(
        'Versions:',
        versions,
        default = versions, #autoselect all
        key='version')

chains = []
for v in version:
    for c in subgraph_json['lending'][v].keys():
        chains.append(v + ' ' + c)

with selection_layout[2]:
    chain = st.multiselect(
        'Chain:',
        chains,
        default=chains,
        key = 'chain')




with selection_layout[3]: 
    start_date = st.date_input('Start Date:', datetime.date(2022, 1, 1))

with selection_layout[4]: 
    end_date = st.date_input('End Date:', datetime.date(2025, 1, 1))

start_date = int(dt.strptime(str(start_date), "%Y-%m-%d").timestamp())
end_date = int(dt.strptime(str(end_date), "%Y-%m-%d").timestamp())

#inicialize the session state with the default of the protocols
if 'version_updated' not in st.session_state:
    st.session_state['version_updated'] = version
    st.session_state['chain_updated'] = chain
    st.session_state['start_date_updated'] = start_date
    st.session_state['end_date_updated'] = end_date
    st.session_state['protocol_updated'] = protocol

#only update the state if button is pressed
if st.button('Update Data'):
    st.session_state['version_updated'] = version
    st.session_state['chain_updated'] = chain
    st.session_state['start_date_updated'] = start_date
    st.session_state['end_date_updated'] = end_date
    st.session_state['protocol_updated'] = protocol


#create a dict with selected fields from the state by removing what is not being used.
temp_dict = subgraph_json['lending'].copy()
for k in subgraph_json['lending'].keys(): 
    if k not in st.session_state.version_updated:
        temp_dict.pop(k)

subgraph_dict = temp_dict.copy()
for k1 in list(temp_dict.keys()):
    for k2 in list(temp_dict[k1].keys()):
        if (k1 + ' ' + k2) not in st.session_state.chain_updated:
            subgraph_dict[k1].pop(k2)




st.write("## "+st.session_state.protocol_updated.title())




#####################################################################


def choose_granularity(_key:str) -> str:
    choice = st.radio(
        'Data granulariry:',
         ['Day', 'Week', 'Month', 'Quarter'],
        index=1, #Start with weeks.
        horizontal=True,
        key=_key)
    
    match choice:
        case 'Day':
            choice = 'D'
        case 'Week':
            choice =  'W'
        case 'Month':
            choice =  'M'
        case 'Quarter':
            choice =  'Q'
    return choice

def show_details(title:str, details:str, df):
    with st.expander("Chart Datails"):
        st.write(details)
        st.dataframe(df)
        st.download_button(
           "Download CSV",
           df.to_csv().encode('utf-8'),
           protocol + '-' + title + '.csv',
           key=title+'_csv'
        )


@cached(ttl=None, cache=Cache.MEMORY)
async def get_df(_session: aiohttp.ClientSession, _payload:str, BASE_URL:str) -> pd.DataFrame:
    req = await _session.request('POST', url=BASE_URL, json=_payload)
    response = await req.json()
    
    # Error catching
    if "data" not in response:
        st.write(f"warning, no data: {BASE_URL} {response} {_payload}")
        return pd.DataFrame()

    df = pd.DataFrame(response['data'][list(response['data'])[0]])
    if df.empty:
        st.write(f"warning, empty response {BASE_URL} {_payload}")
    return df
    
async def get_chain(_session: aiohttp.ClientSession, _payload:str, _BASE_URL:str) -> pd.DataFrame:
    df = await get_df(_session, _payload, _BASE_URL)
    df['inputToken'] = pd.DataFrame([*df['inputToken']])[['symbol']]
    df.set_index('inputToken', inplace=True)
    return df

async def get_version(_session: aiohttp.ClientSession, _payload, _version_dict):
    chains_df_list = await asyncio.gather(*(get_chain(_session, _payload, _version_dict[k2]) for k2 in list(_version_dict.keys())))
    chainsdf = pd.concat(chains_df_list, keys = list(_version_dict.keys()), names=['Chain'], axis=0)
    return chainsdf

async def get_protocol(_subgraph_dict, _payload)-> pd.DataFrame:
    async with aiohttp.ClientSession() as session:
        versions_df_list = await asyncio.gather(*(get_version(session, _payload, _subgraph_dict[k1]) for k1 in list(_subgraph_dict.keys())))
    df = pd.concat(versions_df_list,keys = list(_subgraph_dict.keys()), names=['Version'], axis=0)
    df['totalDepositBalanceUSD'] = df['totalDepositBalanceUSD'].astype('double')
    df['totalBorrowBalanceUSD'] = df['totalBorrowBalanceUSD'].astype('double')
    df['Available to Borrow (USD)'] = df['totalDepositBalanceUSD']-df['totalBorrowBalanceUSD']    
    return df








@cached(ttl=None, cache=Cache.MEMORY)
async def get_all_schema(_BASE_URL:str, _fieldPath:str, _start_date, _end_date) -> pd.DataFrame:
    sg = Subgrounds()
    protocol = sg.load_subgraph(_BASE_URL)
    query = getattr(protocol.Query, _fieldPath)(first=100000, where= {'timestamp_lte':_end_date, 'timestamp_gte':_start_date})
    df = sg.query_df([query])
    df.rename(columns = {(_fieldPath+'_timestamp'):'Timestamp'}, inplace = True)
    df.set_index('Timestamp', inplace=True)
    df.index = pd.to_datetime(df.index, unit='s').to_period('D').to_timestamp()
    df.sort_index(inplace=True)
    return df

async def get_all_chain_schema(_chain_dict, _fieldPath:str, _start_date, _end_date) -> pd.DataFrame:
    chains_df_list = await asyncio.gather(*(get_all_schema(_chain_dict[k], _fieldPath, _start_date, _end_date) for k in _chain_dict.keys()))
    chainsdf = pd.concat(chains_df_list, keys = list(_chain_dict.keys()), names=['Chain'], axis=0)
    return chainsdf

async def get_all_version_schema(_version_dict, _fieldPath:str, _start_date, _end_date) -> pd.DataFrame:   
    versions_df_list = await asyncio.gather(*(get_all_chain_schema(_version_dict[k], _fieldPath, _start_date, _end_date) for k in _version_dict.keys()))
    versiondf = pd.concat(versions_df_list, keys = list(_version_dict.keys()), names=['Version'], axis=0)
    return versiondf








async def get_market(_session: aiohttp.ClientSession, _payload:str, _BASE_URL:str, _market:str, _start_date:int, _end_date:int) -> pd.DataFrame:
    _skip = 0
    edited_payload = _payload.copy()
    edited_payload['query'] = _payload['query'].format(market=_market, start_date=_start_date, end_date=_end_date, skip=_skip)
    tmp_df = await get_df(_session, edited_payload, _BASE_URL)
    df = tmp_df.copy()
    while len(tmp_df.index) == 1000:
        _skip += 1000
        edited_payload['query'] = _payload['query'].format(market=_market, start_date=_start_date, end_date=_end_date, skip=_skip)
        tmp_df = await get_df(_session, edited_payload, _BASE_URL)
        df = pd.concat([df, tmp_df])
    return df

async def get_all_markets(_session: aiohttp.ClientSession, _payload:str, _BASE_URL:str, _markets, _start_date:int, _end_date:int) -> pd.DataFrame:
    markets_df = await asyncio.gather(*(get_market(_session, _payload, _BASE_URL, market, _start_date, _end_date) for market in _markets['id']))
    df = pd.concat(markets_df)
    df = df.astype({'timestamp': 'int',
                            'totalValueLockedUSD': 'double',
                            'cumulativeSupplySideRevenueUSD': 'double',
                            'dailySupplySideRevenueUSD': 'double',
                            'cumulativeProtocolSideRevenueUSD': 'double',
                            'dailyProtocolSideRevenueUSD': 'double',
                            'cumulativeTotalRevenueUSD': 'double',
                            'dailyTotalRevenueUSD': 'double',
                            'totalDepositBalanceUSD': 'double',
                            'dailyDepositUSD': 'double',
                            'cumulativeDepositUSD': 'double',
                            'totalBorrowBalanceUSD': 'double',
                            'dailyBorrowUSD': 'double',
                            'cumulativeBorrowUSD': 'double',
                            'dailyLiquidateUSD': 'double',
                            'cumulativeLiquidateUSD': 'double',
                            'dailyWithdrawUSD': 'double',
                            'dailyRepayUSD': 'double',
                            'inputTokenBalance': 'double',
                            'inputTokenPriceUSD': 'double',
                            'outputTokenSupply': 'double',
                            'outputTokenPriceUSD': 'double',
                            'exchangeRate': 'double'})
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s').dt.to_period('D').dt.to_timestamp()
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    df = pd.concat([pd.json_normalize(df['market']).set_index(df.index), df.drop('market', axis=1)], axis=1)
    return df

async def get_all_chains_markets(_session: aiohttp.ClientSession, _payload:str, _chain_dict, _markets, _start_date:int, _end_date:int)-> pd.DataFrame:
    chains_df_list = await asyncio.gather(*(get_all_markets(_session, _payload, _chain_dict[k], _markets.loc[(k),['id']], _start_date, _end_date) for k in _chain_dict.keys()))
    chainsdf = pd.concat(chains_df_list, keys = list(_chain_dict.keys()), names=['Chain'], axis=0)
    chainsdf.sort_index()
    return chainsdf

async def get_all_protocol_markets(_payload:str, _version_dict, _markets, _start_date, _end_date)-> pd.DataFrame:
    async with aiohttp.ClientSession() as session:
        versions_df_list = await asyncio.gather(*(get_all_chains_markets(session, _payload, _version_dict[k], _markets.loc[(k),['id']], _start_date, _end_date) for k in _version_dict.keys()))
        versionsdf = pd.concat(versions_df_list,keys = list(_version_dict.keys()), names=['Version'], axis=0)
        versionsdf.sort_index()
        return(versionsdf)

payload1 = {
                'query': 
                '''
                    {
                        markets(first: 1000){
                        id
                        inputToken{
                            symbol
                        }
                        totalDepositBalanceUSD
                        totalBorrowBalanceUSD
                        }
                    }
                '''        
            }

payload2 =  {
        'query':
            ''' {{
                    marketDailySnapshots(first: 1000,
                        skip: {skip},
                        where: {{
                            market: "{market}",
                            timestamp_gte: {start_date},
                            timestamp_lte: {end_date}
                        }}
                    ){{
                        timestamp
                        market{{
                            id
                            inputToken{{
                            symbol
                            }}
                        }}
                        rates{{
                            rate
                            side
                        }}
                        totalValueLockedUSD
                        cumulativeSupplySideRevenueUSD
                        dailySupplySideRevenueUSD
                        cumulativeProtocolSideRevenueUSD
                        dailyProtocolSideRevenueUSD
                        cumulativeTotalRevenueUSD
                        dailyTotalRevenueUSD
                        totalDepositBalanceUSD
                        dailyDepositUSD
                        cumulativeDepositUSD
                        totalBorrowBalanceUSD
                        dailyBorrowUSD
                        cumulativeBorrowUSD
                        dailyLiquidateUSD
                        cumulativeLiquidateUSD
                        dailyWithdrawUSD
                        dailyRepayUSD
                        inputTokenBalance
                        inputTokenPriceUSD
                        outputTokenSupply
                        outputTokenPriceUSD
                        exchangeRate
                        rewardTokenEmissionsAmount
                        rewardTokenEmissionsUSD
                    }}
            }}'''
}

async def main():
    dfs = await asyncio.gather(
        get_all_version_schema(subgraph_dict, 'usageMetricsDailySnapshots', st.session_state.start_date_updated, st.session_state.end_date_updated),
        get_all_version_schema(subgraph_dict, 'financialsDailySnapshots', st.session_state.start_date_updated, st.session_state.end_date_updated),
        get_all_protocol_markets(payload2, subgraph_dict, dfm1.loc[(),['id']], st.session_state.start_date_updated, st.session_state.end_date_updated)
    )
    return dfs

dfm1 = asyncio.run(get_protocol(subgraph_dict, payload1))
dfm2, dfm3, dfm4 = asyncio.run(main())


df_markets = dfm1.copy()
df_usageMetrics = dfm2.copy()
df_financials = dfm3.copy()
df_markets_snapshots = dfm4.copy()



yulesa_template = dict(
    layout=go.Layout(
        # title_font=dict(family="Open Sans", size=26),
        # font=dict(family="Open Sans"),
        xaxis=dict(tickfont = dict(size=12)),
        yaxis=dict(tickfont = dict(size=12)),
        height=600,
        title_x=0.465,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        plot_bgcolor='rgb(24, 40, 53)'
    )
)




#######################################################################################################




#### Charts Row 1 ####
column1, column2, column3 = st.columns(3)

with column1:
    title1 = 'Total Deposited (USD)'
    df1 = df_markets.loc[:, ['totalDepositBalanceUSD']]
    df1 = df1.groupby('inputToken').sum()
    df1.rename(columns = {'totalDepositBalanceUSD':'Total Deposited (USD)'}, inplace = True)
    df1 = df1.sort_values(by=['Total Deposited (USD)'], ascending=False)
    fig1 = go.Figure()
    fig1.add_trace(go.Pie(
        labels=df1.index,
        values=df1.loc[:, 'Total Deposited (USD)']))
    fig1.update_layout(
        title_text='<b>Total Deposited (USD)</b>',
    )
    column1.plotly_chart(fig1, use_container_width=True)
    show_details(title1,
                 """
                    Total deposits across all protocol.
                 """,
                 df1)
        
with column2:
    title2 = 'Total Borrowed (USD)'
    df2 = df_markets.loc[:, ['totalBorrowBalanceUSD']]
    df2 = df2.groupby('inputToken').sum()
    df2.rename(columns = {'totalBorrowBalanceUSD':'Total Borrowed (USD)'}, inplace = True)
    df2 = df2.sort_values(by=['Total Borrowed (USD)'], ascending=False)
    fig2 = make_subplots()
    fig2.add_trace(go.Pie(
        labels=df2.index,
        values=df2.loc[:, 'Total Borrowed (USD)']))
    fig2.update_layout(
        title_text='<b>Total Borrowed (USD)</b>',
    )
    column2.plotly_chart(fig2, use_container_width=True)
    show_details(title2,
                 """
                    Total borrows across all protocol.
                 """,
                 df2)
        
with column3:
    title3 = 'Available to Borrow (USD)'
    df3 = df_markets.loc[:, ['Available to Borrow (USD)']]
    df3 = df3.groupby('inputToken').sum()
    df3 = df3.sort_values(by=['Available to Borrow (USD)'], ascending=False)
    fig3 = make_subplots()
    fig3.add_trace(go.Pie(
        labels=df3.index,
        values=df3['Available to Borrow (USD)']))
    fig3.update_layout(
        title_text='<b>Available to Borrow (USD)</b>',
    )
    column3.plotly_chart(fig3, use_container_width=True)
    
    show_details(title3,
                 """
                    Total available to borrow across all markets. Total available is Total Deposited - Total Borrowed.
                 """,
                 df3)
        
        
        
        
        
st.markdown("""---""")
#### Charts Row 2 ####





column1, column2= st.columns(2)


with column1:
    graph_placeholder4 = column1.empty()
    g4 = choose_granularity('g4') 
    title4 = 'User and Transaction Count'
    df4 = df_usageMetrics.loc[:, ['usageMetricsDailySnapshots_dailyTransactionCount', 'usageMetricsDailySnapshots_dailyActiveUsers']]
    df4 = df4.groupby(level=2).sum()
    df4 = df4.resample(g4).sum()
    df4.rename(columns = {'usageMetricsDailySnapshots_dailyTransactionCount':'Daily Transaction Count'}, inplace = True)
    df4.rename(columns = {'usageMetricsDailySnapshots_dailyActiveUsers':'Daily Unique Active User'}, inplace = True)


    fig4 = go.Figure(data=[
        go.Bar(name='Daily Transaction Count', x=df4.index, y=df4.loc[:,'Daily Transaction Count']),
        go.Bar(name='Daily Unique Active User', x=df4.index, y=df4.loc[:, 'Daily Unique Active User'])
    ])

    fig4.update_layout(template=yulesa_template)
    fig4.update_layout(
        barmode='group',
        title_text='<b>User and Transaction Count</b>',
        xaxis_title='Time'
    )
    
    graph_placeholder4.plotly_chart(fig4, use_container_width=True)
    
      
    show_details(title4,
                 """
                    Total available to borrow across all markets. Total available is Total Deposited - Total Borrowed.
                 """,
                 df4)

with column2:    
    graph_placeholder5 = column2.empty()
    g5 = choose_granularity('g5')
    title5 = 'Actions Count'
    df5 = df_usageMetrics.loc[:, ['usageMetricsDailySnapshots_dailyDepositCount',
                       'usageMetricsDailySnapshots_dailyWithdrawCount',
                       'usageMetricsDailySnapshots_dailyBorrowCount',
                       'usageMetricsDailySnapshots_dailyRepayCount',
                       'usageMetricsDailySnapshots_dailyLiquidateCount']]
    df5 = df5.groupby(level=2).sum()
    df5 = df5.resample(g5).sum()
    df5.rename(columns = {'usageMetricsDailySnapshots_dailyDepositCount':'Deposits',
                         'usageMetricsDailySnapshots_dailyWithdrawCount':'Withdraws',
                          'usageMetricsDailySnapshots_dailyBorrowCount':'Borrows',
                          'usageMetricsDailySnapshots_dailyRepayCount':'Repays',
                          'usageMetricsDailySnapshots_dailyLiquidateCount':'Liquidations'}, inplace = True)


    fig5 = go.Figure(data=[
        go.Bar(name='Deposits', x=df5.index, y=df5.loc[:, 'Deposits']),
        go.Bar(name='Withdraws', x=df5.index, y=df5.loc[:, 'Withdraws']),
        go.Bar(name='Borrows', x=df5.index, y=df5.loc[:, 'Borrows']),
        go.Bar(name='Repays', x=df5.index, y=df5.loc[:, 'Repays']),
        go.Bar(name='Liquidations', x=df5.index, y=df5.loc[:, 'Liquidations']),
    ])

    fig5.update_layout(template=yulesa_template)
    fig5.update_layout(
        barmode='group',
        title_text=f'<b>{title5}</b>',
        xaxis_title='Time'
    )
    
    graph_placeholder5.plotly_chart(fig5, use_container_width=True)
    
    show_details(title5,
                 """
                    Total available to borrow across all markets. Total available is Total Deposited - Total Borrowed.
                 """,
                 df5)


        
        
st.markdown("""---""")
#### Charts Row 3 ####





column1, column2 = st.columns(2)
    
with column1:    
    graph_placeholder6 = column1.empty()
    g6 = choose_granularity('g6')
    title6 = 'Income Statement'
    df6 = df_financials.loc[:, ['financialsDailySnapshots_dailyTotalRevenueUSD',
                     'financialsDailySnapshots_dailySupplySideRevenueUSD',
                     'financialsDailySnapshots_dailyProtocolSideRevenueUSD']]
    df6 = df6.groupby(level=2).sum()
    df6.rename(columns = {
        'financialsDailySnapshots_dailyTotalRevenueUSD':'Total Revenue (USD)',
        'financialsDailySnapshots_dailySupplySideRevenueUSD': 'Supply Side Revenue (USD)',
        'financialsDailySnapshots_dailyProtocolSideRevenueUSD': 'Protocol Side Revenue (USD)'}, inplace = True)
    df6 = df6.resample(g6).sum()

    fig6 = go.Figure(data=[
        go.Bar(name='Protocol Side Revenue (USD)', x=df6.index, y=df6.loc[:, 'Protocol Side Revenue (USD)']),
        go.Bar(name='Supply Side Revenue (USD)', x=df6.index, y=df6.loc[:, 'Supply Side Revenue (USD)']),
        go.Scatter(name='Total Revenue (USD)', x=df6.index, y=df6.loc[:, 'Total Revenue (USD)'])
    ])

    fig6.update_layout(template=yulesa_template)
    fig6.update_layout(
        barmode='stack',
        title_text=f'<b>{title6}</b>',
        xaxis_title='Time'
    )

    graph_placeholder6.plotly_chart(fig6, use_container_width=True)
    
    show_details(title6,
                 """
                    Total available to borrow across all markets. Total available is Total Deposited - Total Borrowed.
                 """,
                 df6)
    
with column2:    
    graph_placeholder7 = column2.empty()
    g7 = choose_granularity('g7')   
    title7 = 'Outstanding Deposits and Loans'
    df7 = df_financials.loc[:, ['financialsDailySnapshots_totalDepositBalanceUSD',
                     'financialsDailySnapshots_totalBorrowBalanceUSD']]
    df7.rename(columns = {
        'financialsDailySnapshots_totalDepositBalanceUSD':'Outstanding Deposits (USD)',
        'financialsDailySnapshots_totalBorrowBalanceUSD': 'Outstanding Loans (USD)'}, inplace = True)
    df7 = df7.groupby(level=2).sum()
    df7['Utilization (%)'] = df7.loc[:,'Outstanding Loans (USD)'] / df7.loc[:,'Outstanding Deposits (USD)']
    df7 = df7.resample(g7).last()

    fig7 = make_subplots(specs=[[{"secondary_y": True}]])
    fig7.add_trace(go.Scatter(name='Outstanding Deposits (USD)', x=df7.index, y=df7['Outstanding Deposits (USD)'], mode='lines', fill='tozeroy'), secondary_y=False)
    fig7.add_trace(go.Scatter(name='Outstanding Loans (USD)', x=df7.index, y=df7['Outstanding Loans (USD)'], mode='lines', fill='tozeroy'), secondary_y=False)
    fig7.add_trace(go.Scatter(name='Utilization (%)', x=df7.index, y=df7['Utilization (%)']), secondary_y=True)

    fig7.update_layout(template=yulesa_template)
    fig7.update_layout(
        title_text=f'<b>{title7}</b>',
        xaxis_title='Time'
    )
    fig7.update_yaxes(title_text='Outstanding Deposits and Loans (USD)', secondary_y=False)
    fig7.update_yaxes(title_text='Utilization (%)', secondary_y=True)

    graph_placeholder7.plotly_chart(fig7, use_container_width=True)
    
    show_details(title7,
                 """
                    Total available to borrow across all markets. Total available is Total Deposited - Total Borrowed.
                 """,
                 df7)



        
        
st.markdown("""---""")
#### Charts Row 4 ####






column1, column2 = st.columns(2)
    
with column1:    
    graph_placeholder8 = column1.empty()
    g8 = choose_granularity('g8')
    title8 = 'Deposits and Withdraws'
    df8 = df_financials.loc[:, ['financialsDailySnapshots_dailyDepositUSD',
                     'financialsDailySnapshots_dailyWithdrawUSD',
                     'financialsDailySnapshots_totalDepositBalanceUSD']]
    df8.rename(columns = {
        'financialsDailySnapshots_dailyDepositUSD':'Deposits (USD)',
        'financialsDailySnapshots_dailyWithdrawUSD': 'Withdraws (USD)',
        'financialsDailySnapshots_totalDepositBalanceUSD': 'Outstanding Deposits (USD)'}, inplace = True)
    df8 = df8.groupby(level=2).sum()
    df8['Net Change (USD)'] = df8.loc[:, 'Deposits (USD)'] - df8.loc[:, 'Withdraws (USD)']
    df8['Withdraws (USD)'] = -df8.loc[:, 'Withdraws (USD)']
    df8a = df8.loc[:, ['Deposits (USD)', 'Withdraws (USD)', 'Net Change (USD)']].resample(g8).sum()
    df8b = df8.loc[:, ['Outstanding Deposits (USD)']].resample(g8).last()
    df8 = pd.concat([df8a, df8b], axis=1)


    fig8 = make_subplots(specs=[[{"secondary_y": True}]])
    fig8.add_trace(go.Bar(name='Deposits (USD)', x=df8.index, y=df8.loc[:, 'Deposits (USD)']), secondary_y=False)
    fig8.add_trace(go.Bar(name='Withdraws (USD)', x=df8.index, y=df8.loc[:, 'Withdraws (USD)']), secondary_y=False)
    fig8.add_trace(go.Scatter(name='Net Change (USD))', x=df8.index, y=df8.loc[:, 'Net Change (USD)'], mode='markers'), secondary_y=False)
    fig8.add_trace(go.Scatter(name='Outstanding Deposits (USD)', x=df8.index, y=df7.loc[:, 'Outstanding Deposits (USD)']), secondary_y=True)

    fig8.update_layout(template=yulesa_template)
    fig8.update_layout(
        barmode='relative',
        title_text=f'<b>{title8}</b>',
        xaxis_title='Time')
    fig8.update_yaxes(title_text='Deposits and Withdraws (USD)', secondary_y=False)
    fig8.update_yaxes(title_text='Outstanding Deposits (USD)', secondary_y=True)

    graph_placeholder8.plotly_chart(fig8, use_container_width=True)
    
    show_details(title8,
                 """
                    Total available to borrow across all markets. Total available is Total Deposited - Total Borrowed.
                 """,
                 df8)

    
with column2:    
    graph_placeholder9 = column2.empty()
    g9 = choose_granularity('g9')
    title9 = 'Borrows and Repays'
    df9 = df_financials.loc[:, ['financialsDailySnapshots_dailyBorrowUSD',
                             'financialsDailySnapshots_dailyRepayUSD',
                             'financialsDailySnapshots_totalBorrowBalanceUSD']]
    df9 = df9.groupby(level=2).sum()
    df9.rename(columns = {
        'financialsDailySnapshots_dailyBorrowUSD':'Borrows (USD)',
        'financialsDailySnapshots_dailyRepayUSD': 'Repays (USD)',
        'financialsDailySnapshots_totalBorrowBalanceUSD': 'Outstanding Loans (USD)'}, inplace = True)
    df9['Net Change (USD)'] = df9.loc[:,'Borrows (USD)'] - df9.loc[:, 'Repays (USD)']
    df9['Repays (USD)'] = -df9.loc[:, 'Repays (USD)']
    df9a = df9.loc[:, ['Borrows (USD)', 'Repays (USD)', 'Net Change (USD)']].resample(g9).sum()
    df9b = df9.loc[:, ['Outstanding Loans (USD)']].resample(g9).last()
    df9 = pd.concat([df9a, df9b], axis=1)



    fig9 = make_subplots(specs=[[{"secondary_y": True}]])
    fig9.add_trace(go.Bar(name='Borrows (USD)', x=df9.index, y=df9.loc[:, 'Borrows (USD)']), secondary_y=False)
    fig9.add_trace(go.Bar(name='Repays (USD)', x=df9.index, y=df9.loc[:, 'Repays (USD)']), secondary_y=False)
    fig9.add_trace(go.Scatter(name='Net Change (USD))', x=df9.index, y=df9.loc[:, 'Net Change (USD)'], mode='markers'), secondary_y=False)
    fig9.add_trace(go.Scatter(name='Outstanding Loans (USD)', x=df9.index, y=df7.loc[:, 'Outstanding Loans (USD)']), secondary_y=True)

    fig9.update_layout(template=yulesa_template)
    fig9.update_layout(
        barmode='relative',
        title_text=f'<b>{title9}</b>',
        xaxis_title='Time')
    fig9.update_yaxes(title_text='Borrows and Repays (USD)', secondary_y=False)
    fig9.update_yaxes(title_text='Outstanding Loans (USD)', secondary_y=True)

    graph_placeholder9.plotly_chart(fig9, use_container_width=True)
    
    show_details(title9,
                 """
                    Total available to borrow across all markets. Total available is Total Deposited - Total Borrowed.
                 """,
                 df9)

        
        
        
        
st.markdown("""---""")
#### Charts Row 5 ####        




column1, column2 = st.columns(2)
    
with column1:    
    graph_placeholder10 = column1.empty()
    g10 = choose_granularity('g10')
    title10 = 'Liquidations by Token'
    df10 = df_markets_snapshots.loc[:, ['inputToken.symbol',
                                      'dailyLiquidateUSD']]
    df10.rename(columns = {'inputToken.symbol': 'Markets',
                           'dailyLiquidateUSD': 'Liquidations (USD)'}, inplace = True)
    df10 = df10.reset_index(level=2).groupby(['timestamp','Markets']).sum().reset_index('Markets')
    df10 = df10.groupby('Markets').resample(g10).sum().loc[:, ['Liquidations (USD)']].reset_index('Markets')
    top_markets = df10.groupby('Markets').max().sort_values('Liquidations (USD)', ascending = False).head()
    df10.loc[~df10.loc[:, 'Markets'].isin(top_markets.index), 'Markets'] = 'Other'
    df10 = df10.groupby(['timestamp','Markets']).sum().swaplevel(axis=0)
    graph_order = df10.groupby('Markets').max().sort_values('Liquidations (USD)', ascending = False)

    fig10 = go.Figure()
    for token in graph_order.index:
        fig10.add_trace(go.Bar(name=token, x=df10.loc[token].index, y=df10.loc[token, 'Liquidations (USD)']))

    fig10.update_layout(template=yulesa_template)
    fig10.update_layout(
        title_text=f'<b>{title10}</b>',
        xaxis_title='Time',
        barmode='stack'
    )
    graph_placeholder10.plotly_chart(fig10, use_container_width=True)
    
    show_details(title10,
                 """
                    Total available to borrow across all markets. Total available is Total Deposited - Total Borrowed.
                 """,
                 df10)
        

with column2:    
    graph_placeholder11 = column2.empty()
    g11 = choose_granularity('g11')
    title11 = 'Cumulative Deposit and Loans'
    df11 = df_financials.loc[:, ['financialsDailySnapshots_cumulativeDepositUSD',
                                 'financialsDailySnapshots_cumulativeBorrowUSD']]
                                 
    df11 = df11.groupby(level=2).sum()
    df11.rename(columns = {
        'financialsDailySnapshots_cumulativeDepositUSD':'Cumulative Deposits (USD)',
        'financialsDailySnapshots_cumulativeBorrowUSD': 'Cumulative Loans (USD)'}, inplace = True)
    df11 = df11.resample(g11).last()

    fig11 = go.Figure(data=[go.Scatter(name='Cumulative Deposits (USD)', x=df11.index, y=df11.loc[:, 'Cumulative Deposits (USD)'], mode='lines', fill='tozeroy'),
                            go.Scatter(name='Cumulative Loans (USD)', x=df11.index, y=df11.loc[:, 'Cumulative Loans (USD)'], mode='lines', fill='tozeroy')])

    fig11.update_layout(template=yulesa_template)
    fig11.update_layout(
        title_text=f'<b>{title11}</b>',
        xaxis_title='Time'
    )

    graph_placeholder11.plotly_chart(fig11, use_container_width=True)
    
    show_details(title11,
                 """
                    Total available to borrow across all markets. Total available is Total Deposited - Total Borrowed.
                 """,
                 df11)


        
        
        
st.markdown("""---""")
#### Charts Row 6 ####        





column1, column2 = st.columns(2)
with column1:        
    graph_placeholder12 = column1.empty()
    g12 = choose_granularity('g12')
    title12 = 'Outstanding Deposit by Token'
    df12 = df_markets_snapshots.loc[:, ['id',
                                     'inputToken.symbol',
                                     'totalDepositBalanceUSD']]
    df12.rename(columns = {'inputToken.symbol': 'Markets',
                            'totalDepositBalanceUSD':'Total Deposited (USD)'}, inplace = True)
    df12 = df12.reset_index(level=2).groupby(['timestamp', 'id', 'Markets']).sum().reset_index(['Markets', 'id'])
    df12 = df12.groupby(['id', 'Markets'], as_index=False).resample(g12).last()
    top_markets = df12.groupby('id').max().sort_values('Total Deposited (USD)', ascending = False).head()
    df12.loc[~df12.loc[:, 'id'].isin(top_markets.index), 'Markets'] = 'Other'
    df12 = df12.groupby(['timestamp','Markets']).sum().swaplevel(axis=0)
    graph_order = df12.groupby('Markets').max().sort_values('Total Deposited (USD)', ascending = False)

    fig12 = go.Figure()
    for token in graph_order.index:
        fig12.add_trace(go.Scatter(name=token, x=df12.loc[token].index, y=df12.loc[token, 'Total Deposited (USD)'], stackgroup='one', hoverinfo='x+y'))

    fig12.update_layout(template=yulesa_template)
    fig12.update_layout(
        title_text=f'<b>{title12}</b>',
        xaxis_title='Time'
    )
    graph_placeholder12.plotly_chart(fig12, use_container_width=True)
    
    show_details(title12,
                 """
                    Total available to borrow across all markets. Total available is Total Deposited - Total Borrowed.
                 """,
                 df12)
        
with column2:    
    graph_placeholder13 = column2.empty()    
    g13 = choose_granularity('g13')
    title13 = 'Outstanding Loans by Token'
    df13 = df_markets_snapshots.loc[:, ['id',
                                     'inputToken.symbol',
                                     'totalBorrowBalanceUSD']]
    df13.rename(columns = {'inputToken.symbol': 'Markets',
                           'totalBorrowBalanceUSD':'Total Loans (USD)'}, inplace = True)
    df13 = df13.reset_index(level=2).groupby(['timestamp', 'id', 'Markets']).sum().reset_index(['Markets', 'id'])
    df13 = df13.groupby(['id', 'Markets'], as_index=False).resample(g13).last()
    # top_markets = df13.groupby('Markets').max().sort_values('Total Loans (USD)', ascending = False).head()
    df13.loc[~df13.loc[:, 'id'].isin(top_markets.index), 'Markets'] = 'Other'
    df13 = df13.groupby(['timestamp','Markets']).sum().swaplevel(axis=0)
    graph_order = df13.groupby('Markets').max().sort_values('Total Loans (USD)', ascending = False)

    fig13 = go.Figure()
    for token in graph_order.index:
        fig13.add_trace(go.Scatter(name=token, x=df13.loc[token,:].index, y=df13.loc[token, 'Total Loans (USD)'], stackgroup='one', hoverinfo='x+y'))

    fig13.update_layout(template=yulesa_template)
    fig13.update_layout(
        title_text=f'<b>{title13}</b>',
        xaxis_title='Time'
    )

    graph_placeholder13.plotly_chart(fig13, use_container_width=True)
    
    show_details(title13,
                 """
                    Total available to borrow across all markets. Total available is Total Deposited - Total Borrowed.
                 """,
                 df13)
        
        
        
        
st.markdown("""---""")
#### Charts Row 7 ####        





graph_placeholder14 = st.empty()  
g14 = choose_granularity('g14')      
title14 = 'Utilization'
df14 = df_markets_snapshots.loc[:, ['id',
                                    'inputToken.symbol',
                                    'totalDepositBalanceUSD',
                                    'totalBorrowBalanceUSD']]
df14.rename(columns = {'inputToken.symbol': 'Markets',
                        'totalDepositBalanceUSD':'Total Deposited (USD)',
                        'totalBorrowBalanceUSD':'Total Borrow (USD)'}, inplace = True)
df14 = df14.reset_index(level=2).groupby(['timestamp', 'id', 'Markets']).sum().reset_index(['Markets', 'id'])
df14 = pd.pivot_table(df14, index=['timestamp'], columns=['id', 'Markets'])
df14 = df14.fillna(method='ffill')
df14 = df14.melt(ignore_index=False).pivot_table(index=['timestamp','id', 'Markets'], columns=[None]).droplevel(level=0, axis=1).reset_index(['id', 'Markets'])
df14['Utilization (%)'] = df14.loc[:, 'Total Borrow (USD)']/df14.loc[:, 'Total Deposited (USD)']
df14 = df14.groupby(['id', 'Markets'], as_index=False).resample(g14).last().droplevel(level=0, axis=0)

fig14 = go.Figure()
for token_address in filter(None, df14['id'].unique()):
    fig14.add_trace(go.Scatter(name=df14.loc[df14['id'] == token_address]['Markets'][0], x=df14.loc[df14['id'] == token_address].index, y=df14.loc[df14['id'] == token_address]['Utilization (%)'].tolist()))

fig14.update_layout(
    title_text=f'<b>{title14}</b>',
    xaxis_title='Time'
)
graph_placeholder14.plotly_chart(fig14, use_container_width=True)

show_details(title14,
                """
                Total available to borrow across all markets. Total available is Total Deposited - Total Borrowed.
                """,
                df14)
        
        
        
        
st.markdown("""---""")
#### Charts Row 7 ####    
        
    
    
        
column1, column2 = st.columns(2)

with column1:
    graph_placeholder15a = st.empty()  
with column2: 
    graph_placeholder15b = st.empty()  
g15 = choose_granularity('g15')

         
title15a = 'Borrow Rate'
df15 = df_markets_snapshots[['inputToken.symbol', 'id', 'rates']]
df15 = df15.reset_index(level=2).groupby(['timestamp', 'id', 'inputToken.symbol']).sum()
df15 = pd.json_normalize(df15.iloc[:,0]).set_index(df15.index)
df15.sort_index(inplace = True)
df_rates = pd.DataFrame()
for x in df15.columns:
    tmp_df = pd.json_normalize(df15[x]).set_index(df15.index)
    tmp_df = tmp_df.rename({'rate': tmp_df.iloc[0,1]}, axis=1).drop('side', axis=1)
    df_rates = pd.concat([df_rates, tmp_df], axis=1)
df15 = df_rates.reset_index(['id', 'inputToken.symbol'])
df15 = pd.pivot_table(df15, index=['timestamp'], columns=['id', 'inputToken.symbol'])
df15 = df15.fillna(method='ffill')
df15 = df15.melt(ignore_index=False).pivot_table(index=['timestamp','id', 'inputToken.symbol'], columns=[None]).droplevel(level=0, axis=1).reset_index(['id', 'inputToken.symbol'])
df15 = df15.groupby(['id', 'inputToken.symbol'], as_index=False).resample(g15).last().droplevel(level=0, axis=0)
df15.sort_index(inplace = True)

fig15a = go.Figure()
for token_address in filter(None, df15['id'].unique().tolist()):
    fig15a.add_trace(go.Scatter(name=df15.loc[df15['id'] == token_address].iloc[0,1], x=df15.index.unique(), y=df15.loc[df15['id'] == token_address].iloc[:,2], mode='lines'))

fig15a.update_layout(
    title_text=f'<b>{title15a}</b>',
    xaxis_title='Time'
    )

graph_placeholder15a.plotly_chart(fig15a, use_container_width=True)


title15b = 'Lender Rate'
fig15b = go.Figure()
for token_address in filter(None, df15['id'].unique().tolist()):
    fig15b.add_trace(go.Scatter(name=df15.loc[df15['id'] == token_address].iloc[0,1], x=df15.index.unique(), y=df15.loc[df15['id'] == token_address].iloc[:,3], mode='lines'))

fig15b.update_layout(
    title_text=f'<b>{title15b}</b>',
    xaxis_title='Time'
)

graph_placeholder15b.plotly_chart(fig15b, use_container_width=True)
    
show_details('Rates',
             """
                Total available to borrow across all markets. Total available is Total Deposited - Total Borrowed.
             """,
             df15)
