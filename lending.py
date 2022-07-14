import streamlit as st
import pandas as pd
import requests
import time
import datetime
from datetime import datetime as dt
from subgrounds import Subgrounds

import plotly.graph_objs as go
from plotly.subplots import make_subplots



#######################################################################################################


st.set_page_config(layout="wide")


###### Side Bar ######

subgraphs = requests.get('https://subgraphs.messari.io/deployments.json')
subgraph_json = subgraphs.json()

protocols = list(subgraph_json['lending-protocols'].keys())
protocol = st.sidebar.selectbox(
    'Select a protocol:',
    protocols,
    key='protocol'
)

chains = list(subgraph_json['lending-protocols'][protocol].keys())
chain = st.sidebar.selectbox(
    'Select a chain:',
    chains,
    key='chain'
)

BASE_URL = subgraph_json['lending-protocols'][protocol][chain]



layout = st.sidebar.columns(2)

with layout[0]: 
    start_date = st.date_input('Start Date:', datetime.date(2022, 1, 1))

with layout[1]: 
    end_date = st.date_input('End Date:', datetime.date(2025, 1, 1))
        
start_date = int(dt.strptime(str(start_date), "%Y-%m-%d").timestamp())
end_date = int(dt.strptime(str(end_date), "%Y-%m-%d").timestamp())

    


#######################################################################################################




# @st.cache
# def get_df(payload) -> pd.DataFrame:
#     response = requests.post(BASE_URL, json=payload).json()
#     df = pd.DataFrame(response['data'][list(response['data'])[0]])
#     return df

@st.cache
def get_df(BASE_URL:str, payload:str) -> pd.DataFrame:
    response = requests.post(BASE_URL, json=payload).json()
    df = pd.DataFrame(response['data'][list(response['data'])[0]])
    return df

@st.cache
def get_all_schema(BASE_URL:str, fieldPath:str, start_date, end_date) -> pd.DataFrame:
    sg = Subgrounds()
    protocol = sg.load_subgraph(BASE_URL)
    query = getattr(protocol.Query, fieldPath)(first=100000, where={'timestamp_gte': start_date, 'timestamp_lte':end_date})
    df = sg.query_df([query])
    return df

def get_custom_query(BASE_URL:str) -> pd.DataFrame:
    sg = Subgrounds()
    protocol = sg.load_subgraph(BASE_URL)
    query2 = protocol.Query.markets.inputToken.symbol()
    query1 = protocol.Query.markets()
    df = sg.query_df([query1, query2])
    return df

# @st.cache
def get_all_markets_ss(BASE_URL:str, markets, start_date:int, end_date:int) -> pd.DataFrame:
    def get_market_ss(BASE_URL:str, market:str, start_date:int, end_date:int) -> pd.DataFrame:
        skip = 0
        payload = {
            'query':
                '''query snapshots($market: String, $skip: Int, $start_date: Int, $end_date: Int){
                    marketDailySnapshots(first: 1000, where: {market: $market, timestamp_gte: $start_date, timestamp_lte: $end_date}, skip: $skip){
                        timestamp
                        market{
                          id
                          inputToken{
                            symbol
                          }
                        }
                        rates{
                          rate
                          side
                        }
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
                    }
                }''',

            'variables' : {
                'market': market,
                'skip': skip,
                'start_date': start_date,
                'end_date': end_date
            }   
        }

        response = requests.post(BASE_URL, json=payload).json()
        tmp_df = pd.DataFrame(response['data'][list(response['data'])[0]])
        df = tmp_df.copy()
        while len(tmp_df.index) == 1000:
            skip += 1000
            payload['variables']['skip'] = skip
            response = requests.post(BASE_URL, json=payload).json()
            tmp_df = pd.DataFrame(response['data'][list(response['data'])[0]])
            df = pd.concat([df, tmp_df])
        return df
    
    df = pd.DataFrame()
    for market in markets['id']:
        tmpdf = get_market_ss(BASE_URL, market, start_date, end_date)
        df = pd.concat([df, tmpdf])
    return df


def choose_granularity(_key:str) -> str:
    choice = st.radio(
        'Data granulariry:',
         ['Day', 'Week', 'Month', 'Quarter'],
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

dfm1 = get_df(BASE_URL, payload1)
df_markets = dfm1.copy()
df_markets['inputToken'] = pd.DataFrame([*df_markets['inputToken']])[['symbol']]
df_markets.set_index('inputToken', inplace=True)
df_markets['totalDepositBalanceUSD'] = df_markets['totalDepositBalanceUSD'].astype('double')
df_markets['totalBorrowBalanceUSD'] = df_markets['totalBorrowBalanceUSD'].astype('double')
df_markets['Available to Borrow (USD)'] = df_markets['totalDepositBalanceUSD']-df_markets['totalBorrowBalanceUSD']


dfm2 = get_all_schema(BASE_URL, 'usageMetricsDailySnapshots', start_date, end_date)
df_usageMetrics = dfm2.copy()
df_usageMetrics.set_index('usageMetricsDailySnapshots_timestamp', inplace=True)
df_usageMetrics.index = pd.to_datetime(df_usageMetrics.index, unit='s').to_period('D').to_timestamp()
df_usageMetrics.sort_index(inplace=True)


dfm3 = get_all_schema(BASE_URL, 'financialsDailySnapshots', start_date, end_date)
df_financials = dfm3.copy()
df_financials.set_index('financialsDailySnapshots_timestamp', inplace=True)
df_financials.index = pd.to_datetime(df_financials.index, unit='s').to_period('D').to_timestamp()
df_financials.sort_index(inplace=True)


dfm4 = get_all_markets_ss(BASE_URL, dfm1[['id']], start_date, end_date)
markets_snapshots = dfm4.copy()
markets_snapshots = markets_snapshots.astype({'timestamp': 'int',
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
markets_snapshots['timestamp'] = pd.to_datetime(markets_snapshots['timestamp'], unit='s').dt.to_period('D').dt.to_timestamp()
markets_snapshots.set_index('timestamp', inplace=True)
markets_snapshots.sort_index(inplace=True)
markets_snapshots = pd.concat([pd.json_normalize(markets_snapshots['market']).set_index(markets_snapshots.index), markets_snapshots.drop('market', axis=1)], axis=1)
df_rates = pd.DataFrame()

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




###### Main Page ######

st.title('Messari: Lending Financial Dashboard')
st.write("## "+protocol.title())

#### Charts Row 1 ####
column1, column2, column3 = st.columns(3)

with column1:
    df1 = df_markets.loc[:, ['totalDepositBalanceUSD']]
    df1.rename(columns = {'totalDepositBalanceUSD':'Total Deposited (USD)'}, inplace = True)
    df1 = df1.sort_values(by=['Total Deposited (USD)'], ascending=False)
    fig1 = make_subplots()
    fig1.add_trace(go.Pie(
        labels=df1.index,
        values=df1['Total Deposited (USD)']))
    fig1.update_layout(
        title_text='<b>Total Deposited (USD)</b>',
    )
    column1.plotly_chart(fig1, use_container_width=True)
    with st.expander("Chart Datails"):
        st.write("""
            Total deposits across all protocol.
         """)
        st.dataframe(df1)
        
with column2:
    df2 = df_markets[['totalBorrowBalanceUSD']]
    df2.rename(columns = {'totalBorrowBalanceUSD':'Total Borrowed (USD)'}, inplace = True)
    
    
    df2 = df2.sort_values(by=['Total Borrowed (USD)'], ascending=False)
    fig2 = make_subplots()
    fig2.add_trace(go.Pie(
        labels=df2.index,
        values=df2['Total Borrowed (USD)']))
    fig2.update_layout(
        title_text='<b>Total Borrowed (USD)</b>',
    )
    column2.plotly_chart(fig2, use_container_width=True)
    with st.expander("Chart Datails"):
        st.write("""
            Total borrows across all protocol.
         """)
        st.dataframe(df2)
        
with column3:
    df3 = df_markets[['Available to Borrow (USD)']]
    df3 = df3.sort_values(by=['Available to Borrow (USD)'], ascending=False)
    fig3 = make_subplots()
    fig3.add_trace(go.Pie(
        labels=df3.index,
        values=df3['Available to Borrow (USD)']))
    fig3.update_layout(
        title_text='<b>Available to Borrow (USD)</b>',
    )
    column3.plotly_chart(fig3, use_container_width=True)
    with st.expander("Chart Datails"):
        st.write("""
            Total available to borrow across all markets. Total available is Total Deposited - Total Borrowed.
         """)
        st.dataframe(df3)
        
        
        
        
        
st.markdown("""---""")
#### Charts Row 2 ####





column1, column2, column3, column4 = st.columns([4, 1, 4, 1])

with column2:
    g4 = choose_granularity('g4')    
    
with column1:    
    df4 = df_usageMetrics[['usageMetricsDailySnapshots_dailyTransactionCount', 'usageMetricsDailySnapshots_dailyActiveUsers']]
    df4 = df4.resample(g4).sum()
    df4.rename(columns = {'usageMetricsDailySnapshots_dailyTransactionCount':'Daily Transaction Count'}, inplace = True)
    df4.rename(columns = {'usageMetricsDailySnapshots_dailyActiveUsers':'Daily Unique Active User'}, inplace = True)


    fig4 = go.Figure(data=[
        go.Bar(name='Daily Transaction Count', x=df4.index, y=df4['Daily Transaction Count'].tolist()),
        go.Bar(name='Daily Unique Active User', x=df4.index, y=df4['Daily Unique Active User'].tolist())
    ])

    fig4.update_layout(template=yulesa_template)
    fig4.update_layout(
        barmode='group',
        title_text='<b>User and Transaction Count</b>',
        xaxis_title='Time'
    )
    column1.plotly_chart(fig4, use_container_width=True)
    
    with st.expander("Chart Datails"):
        st.write("""
            Total deposits across all protocol.
         """)
        st.dataframe(df4)

with column4:    
    g5 = choose_granularity('g5')

with column3:
    df5 = df_usageMetrics[['usageMetricsDailySnapshots_dailyDepositCount',
                           'usageMetricsDailySnapshots_dailyWithdrawCount',
                           'usageMetricsDailySnapshots_dailyBorrowCount',
                           'usageMetricsDailySnapshots_dailyRepayCount',
                           'usageMetricsDailySnapshots_dailyLiquidateCount']]
    df5 = df5.resample(g5).sum()
    df5.rename(columns = {'usageMetricsDailySnapshots_dailyDepositCount':'Deposits',
                         'usageMetricsDailySnapshots_dailyWithdrawCount':'Withdraws',
                          'usageMetricsDailySnapshots_dailyBorrowCount':'Borrows',
                          'usageMetricsDailySnapshots_dailyRepayCount':'Repays',
                          'usageMetricsDailySnapshots_dailyLiquidateCount':'Liquidations'}, inplace = True)


    fig5 = go.Figure(data=[
        go.Bar(name='Deposits', x=df5.index, y=df5['Deposits'].tolist()),
        go.Bar(name='Withdraws', x=df5.index, y=df5['Withdraws'].tolist()),
        go.Bar(name='Borrows', x=df5.index, y=df5['Borrows'].tolist()),
        go.Bar(name='Repays', x=df5.index, y=df5['Repays'].tolist()),
        go.Bar(name='Liquidations', x=df5.index, y=df5['Liquidations'].tolist()),
    ])
    
    fig5.update_layout(template=yulesa_template)
    fig5.update_layout(
        barmode='group',
        title_text='<b>Actions Count</b>',
        xaxis_title='Time'
    )
    column3.plotly_chart(fig5, use_container_width=True)
    with st.expander("Chart Datails"):
        st.write("""
            Total deposits across all protocol.
         """)
        st.dataframe(df5)


        
        
st.markdown("""---""")
#### Charts Row 3 ####





column1, column2, column3, column4 = st.columns([4, 1, 4, 1])

with column2:
    g6 = choose_granularity('g6')
    
with column1:
    df6 = df_financials[['financialsDailySnapshots_dailyTotalRevenueUSD',
                     'financialsDailySnapshots_dailySupplySideRevenueUSD',
                     'financialsDailySnapshots_dailyProtocolSideRevenueUSD']]
    df6.rename(columns = {
        'financialsDailySnapshots_dailyTotalRevenueUSD':'Total Revenue (USD)',
        'financialsDailySnapshots_dailySupplySideRevenueUSD': 'Supply Side Revenue (USD)',
        'financialsDailySnapshots_dailyProtocolSideRevenueUSD': 'Protocol Side Revenue (USD)'}, inplace = True)

    df6 = df6.resample(g6).sum()

    fig6 = go.Figure(data=[
        go.Bar(name='Protocol Side Revenue (USD)', x=df6.index, y=df6['Protocol Side Revenue (USD)'].tolist()),
        go.Bar(name='Supply Side Revenue (USD)', x=df6.index, y=df6['Supply Side Revenue (USD)'].tolist()),
        go.Scatter(name='Total Revenue (USD)', x=df6.index, y=df6['Total Revenue (USD)'].tolist())
    ])

    fig6.update_layout(template=yulesa_template)
    fig6.update_layout(
        barmode='stack',
        title_text='<b>Income Statement</b>',
        xaxis_title='Time'
    )

    st.plotly_chart(fig6, use_container_width=True)
    with st.expander("Chart Datails"):
        st.write("""
            Total deposits across all protocol.
         """)
        st.dataframe(df6)

with column4:
    g7 = choose_granularity('g7')
    
with column3:   
    df7 = df_financials[['financialsDailySnapshots_totalDepositBalanceUSD',
                         'financialsDailySnapshots_totalBorrowBalanceUSD']]
    df7.rename(columns = {
        'financialsDailySnapshots_totalDepositBalanceUSD':'Outstanding Deposits (USD)',
        'financialsDailySnapshots_totalBorrowBalanceUSD': 'Outstanding Loans (USD)'}, inplace = True)
    df7['Utilization (%)'] = df7['Outstanding Loans (USD)'] / df7['Outstanding Deposits (USD)']
    df7 = df7.resample(g7).last()
    
    fig7 = make_subplots(specs=[[{"secondary_y": True}]])
    fig7.add_trace(go.Scatter(name='Outstanding Deposits (USD)', x=df7.index, y=df7['Outstanding Deposits (USD)'].tolist(), mode='lines', fill='tozeroy'), secondary_y=False)
    fig7.add_trace(go.Scatter(name='Outstanding Loans (USD)', x=df7.index, y=df7['Outstanding Loans (USD)'].tolist(), mode='lines', fill='tozeroy'), secondary_y=False)
    fig7.add_trace(go.Scatter(name='Utilization (%)', x=df7.index, y=df7['Utilization (%)'].tolist()), secondary_y=True)

    fig7.update_layout(template=yulesa_template)
    fig7.update_layout(
        title_text='<b>Outstanding Deposits and Loans</b>',
        xaxis_title='Time'
    )
    fig7.update_yaxes(title_text='Outstanding Deposits and Loans (USD)', secondary_y=False)
    fig7.update_yaxes(title_text='Utilization (%)', secondary_y=True)

    st.plotly_chart(fig7, use_container_width=True)
    with st.expander("Chart Datails"):
        st.write("""
            Total deposits across all protocol.
         """)
        st.dataframe(df7)

        
        
        
        
        
st.markdown("""---""")
#### Charts Row 4 ####






column1, column2, column3, column4 = st.columns([4, 1, 4, 1])
with column2:
    g8 = choose_granularity('g8')
    
with column1:
    df8 = df_financials[['financialsDailySnapshots_dailyDepositUSD',
                     'financialsDailySnapshots_dailyWithdrawUSD',
                     'financialsDailySnapshots_totalDepositBalanceUSD']]
    df8.rename(columns = {
        'financialsDailySnapshots_dailyDepositUSD':'Deposits (USD)',
        'financialsDailySnapshots_dailyWithdrawUSD': 'Withdraws (USD)',
        'financialsDailySnapshots_totalDepositBalanceUSD': 'Outstanding Deposits (USD)'}, inplace = True)
    df8['Net Change (USD)'] = df8['Deposits (USD)'] - df8['Withdraws (USD)']
    df8['Withdraws (USD)'] = -df8['Withdraws (USD)']
    df8a = df8[['Deposits (USD)', 'Withdraws (USD)', 'Net Change (USD)']].resample(g8).sum()
    df8b = df8[['Outstanding Deposits (USD)']].resample(g8).last()
    df8 = pd.concat([df8a, df8b], axis=1)
    

    fig8 = make_subplots(specs=[[{"secondary_y": True}]])
    fig8.add_trace(go.Bar(name='Deposits (USD)', x=df8.index, y=df8['Deposits (USD)'].tolist()), secondary_y=False)
    fig8.add_trace(go.Bar(name='Withdraws (USD)', x=df8.index, y=df8['Withdraws (USD)'].tolist()), secondary_y=False)
    fig8.add_trace(go.Scatter(name='Net Change (USD))', x=df8.index, y=df8['Net Change (USD)'].tolist(), mode='markers'), secondary_y=False)
    fig8.add_trace(go.Scatter(name='Outstanding Deposits (USD)', x=df8.index, y=df7['Outstanding Deposits (USD)'].tolist()), secondary_y=True)

    fig8.update_layout(template=yulesa_template)
    fig8.update_layout(
        barmode='relative',
        title_text='<b>Deposits and Withdraws</b>',
        xaxis_title='Time')
    fig8.update_yaxes(title_text='Deposits and Withdraws (USD)', secondary_y=False)
    fig8.update_yaxes(title_text='Outstanding Deposits (USD)', secondary_y=True)

    st.plotly_chart(fig8, use_container_width=True)
    with st.expander("Chart Datails"):
        st.write("""
            Total deposits across all protocol.
         """)
        st.dataframe(df8)

with column4:
    g9 = choose_granularity('g9')
    
with column3:
    df9 = df_financials[['financialsDailySnapshots_dailyBorrowUSD',
                     'financialsDailySnapshots_dailyRepayUSD',
                     'financialsDailySnapshots_totalBorrowBalanceUSD']]
    df9.rename(columns = {
        'financialsDailySnapshots_dailyBorrowUSD':'Borrows (USD)',
        'financialsDailySnapshots_dailyRepayUSD': 'Repays (USD)',
        'financialsDailySnapshots_totalBorrowBalanceUSD': 'Outstanding Loans (USD)'}, inplace = True)
    df9['Net Change (USD)'] = df9['Borrows (USD)'] - df9['Repays (USD)']
    df9['Repays (USD)'] = -df9['Repays (USD)']
    df9a = df9[['Borrows (USD)', 'Repays (USD)', 'Net Change (USD)']].resample(g9).sum()
    df9b = df9[['Outstanding Loans (USD)']].resample(g9).last()
    df9 = pd.concat([df9a, df9b], axis=1)
    
    

    fig9 = make_subplots(specs=[[{"secondary_y": True}]])
    fig9.add_trace(go.Bar(name='Borrows (USD)', x=df9.index, y=df9['Borrows (USD)'].tolist()), secondary_y=False)
    fig9.add_trace(go.Bar(name='Repays (USD)', x=df9.index, y=df9['Repays (USD)'].tolist()), secondary_y=False)
    fig9.add_trace(go.Scatter(name='Net Change (USD))', x=df9.index, y=df9['Net Change (USD)'].tolist(), mode='markers'), secondary_y=False)
    fig9.add_trace(go.Scatter(name='Outstanding Loans (USD)', x=df9.index, y=df7['Outstanding Loans (USD)'].tolist()), secondary_y=True)

    fig9.update_layout(template=yulesa_template)
    fig9.update_layout(
        barmode='relative',
        title_text='<b>Borrows and Repays</b>',
        xaxis_title='Time')
    fig9.update_yaxes(title_text='Borrows and Repays (USD)', secondary_y=False)
    fig9.update_yaxes(title_text='Outstanding Loans (USD)', secondary_y=True)

    st.plotly_chart(fig9, use_container_width=True)
    with st.expander("Chart Datails"):
        st.write("""
            Total deposits across all protocol.
         """)
        st.dataframe(df9)

        
        
        
        
st.markdown("""---""")
#### Charts Row 5 ####        




column1, column2, column3, column4 = st.columns([4, 1, 4, 1])

with column2:
    g10 = choose_granularity('g10')
    
with column1:
    df10 = markets_snapshots[['inputToken.symbol',
                              'dailyLiquidateUSD']]
    df10.rename(columns = {'inputToken.symbol': 'Markets',
                           'dailyLiquidateUSD': 'Liquidations (USD)'}, inplace = True)
    df10 = df10.groupby('Markets').resample(g10).sum()[['Liquidations (USD)']].reset_index('Markets')
    top_markets = df10.groupby('Markets').max().sort_values('Liquidations (USD)', ascending = False).head()
    df10.loc[~df10['Markets'].isin(top_markets.index.tolist()), 'Markets'] = 'Other'
    df10 = df10.groupby(['timestamp','Markets']).sum().swaplevel(axis=0)
    graph_order = df10.groupby('Markets').max().sort_values('Liquidations (USD)', ascending = False)

    fig10 = go.Figure()
    for token in graph_order.index:
        fig10.add_trace(go.Bar(name=token, x=df10.loc[token].index, y=df10.loc[token]['Liquidations (USD)'].tolist()))

    fig10.update_layout(template=yulesa_template)
    fig10.update_layout(
        title_text='<b>Liquidations by Token</b>',
        xaxis_title='Time',
        barmode='stack'
    )
    st.plotly_chart(fig10, use_container_width=True)
    with st.expander("Chart Datails"):
        st.write("""
            Total deposits across all protocol.
         """)
        st.dataframe(df10)
        
        
with column4:
    g11 = choose_granularity('g11')
    
with column3:
    df11 = df_financials[['financialsDailySnapshots_cumulativeDepositUSD',
                         'financialsDailySnapshots_cumulativeBorrowUSD']]
    df11.rename(columns = {
        'financialsDailySnapshots_cumulativeDepositUSD':'Cumulative Deposits (USD)',
        'financialsDailySnapshots_cumulativeBorrowUSD': 'Cumulative Loans (USD)'}, inplace = True)
    df11 = df11.resample(g11).last()
    
    fig11 = go.Figure(data=[go.Scatter(name='Cumulative Deposits (USD)', x=df11.index, y=df11['Cumulative Deposits (USD)'].tolist(), mode='lines', fill='tozeroy'),
                            go.Scatter(name='Cumulative Loans (USD)', x=df11.index, y=df11['Cumulative Loans (USD)'].tolist(), mode='lines', fill='tozeroy')])

    fig11.update_layout(template=yulesa_template)
    fig11.update_layout(
        title_text='<b>Cumulative Deposit and Loans</b>',
        xaxis_title='Time'
    )

    st.plotly_chart(fig11, use_container_width=True)
    with st.expander("Chart Datails"):
        st.write("""
            Total deposits across all protocol.
         """)
        st.dataframe(df11)


        
        
        
st.markdown("""---""")
#### Charts Row 6 ####        





column1, column2, column3, column4 = st.columns([4,1,4,1])
with column2: 
    g12 = choose_granularity('g12')
    
with column1:    
    df12 = markets_snapshots[['id',
                              'inputToken.symbol',
                              'totalDepositBalanceUSD']]
    df12.rename(columns = {'inputToken.symbol': 'Markets',
                            'totalDepositBalanceUSD':'Total Deposited (USD)'}, inplace = True)
    df12 = df12.groupby(['id', 'Markets'], as_index=False).resample(g12).last()
    top_markets = df12.groupby('id').max().sort_values('Total Deposited (USD)', ascending = False).head()
    df12.loc[~df12['id'].isin(top_markets.index.tolist()), 'Markets'] = 'Other'
    df12 = df12.groupby(['timestamp','Markets']).sum().swaplevel(axis=0)
    graph_order = df12.groupby('Markets').max().sort_values('Total Deposited (USD)', ascending = False)

    fig12 = go.Figure()
    for token in graph_order.index:
        fig12.add_trace(go.Scatter(name=token, x=df12.loc[token].index, y=df12.loc[token]['Total Deposited (USD)'].tolist(), stackgroup='one', hoverinfo='x+y'))

    fig12.update_layout(template=yulesa_template)
    fig12.update_layout(
        title_text='<b>Outstanding Deposit by Token</b>',
        xaxis_title='Time'
    )
    st.plotly_chart(fig12, use_container_width=True)
    with st.expander("Chart Datails"):
        st.write("""
            Total deposits across all protocol.
         """)
        st.dataframe(df12)
        
        
with column4: 
    g13 = choose_granularity('g13')
    
with column3:    
    df13 = markets_snapshots[['id',
                              'inputToken.symbol',
                              'totalBorrowBalanceUSD']]
    df13.rename(columns = {'inputToken.symbol': 'Markets',
                           'totalBorrowBalanceUSD':'Total Loans (USD)'}, inplace = True)
    df13 = df13.groupby(['id', 'Markets'], as_index=False).resample(g13).last()
    # top_markets = df13.groupby('Markets').max().sort_values('Total Loans (USD)', ascending = False).head()  # Comment line to keep the same top_markets as the graph above.
    df13.loc[~df13['id'].isin(top_markets.index.tolist()), 'Markets'] = 'Other'
    df13 = df13.groupby(['timestamp','Markets']).sum().swaplevel(axis=0)
    graph_order = df13.groupby('Markets').max().sort_values('Total Loans (USD)', ascending = False) # Comment line to keep the same order as the graph above.

    fig13 = go.Figure()
    for token in graph_order.index:
        fig13.add_trace(go.Scatter(name=token, x=df13.loc[token,:].index, y=df13.loc[token,:]['Total Loans (USD)'].tolist(), stackgroup='one', hoverinfo='x+y'))

    fig13.update_layout(template=yulesa_template)
    fig13.update_layout(
        title_text='<b>Outstanding Loans by Token</b>',
        xaxis_title='Time'
    )

    st.plotly_chart(fig13, use_container_width=True)
    with st.expander("Chart Datails"):
        st.write("""
            Total deposits across all protocol.
         """)
        st.dataframe(df13)
        
        
        
        
st.markdown("""---""")
#### Charts Row 7 ####        





column1, column2 = st.columns([4,1])
with column2: 
    g14 = choose_granularity('g14')
    
with column1:    
    df14 = markets_snapshots[['id',
                              'inputToken.symbol',
                              'totalDepositBalanceUSD',
                              'totalBorrowBalanceUSD']]
    df14.rename(columns = {'inputToken.symbol': 'Markets',
                            'totalDepositBalanceUSD':'Total Deposited (USD)',
                            'totalBorrowBalanceUSD':'Total Borrow (USD)'}, inplace = True)
    df14 = pd.pivot_table(df14, index=['timestamp'], columns=['id', 'Markets'])
    df14 = df14.fillna(method='ffill')
    df14 = df14.melt(ignore_index=False).pivot_table(index=['timestamp','id', 'Markets'], columns=[None]).droplevel(level=0, axis=1).reset_index(['id', 'Markets'])
    df14['Utilization (%)'] = df14['Total Borrow (USD)']/df14['Total Deposited (USD)']
    df14 = df14.groupby(['id', 'Markets'], as_index=False).resample(g14).last().droplevel(level=0, axis=0)
    
    fig14 = go.Figure()
    for token_address in filter(None, df14['id'].unique()):
        fig14.add_trace(go.Scatter(name=df14.loc[df14['id'] == token_address]['Markets'][0], x=df14.loc[df14['id'] == token_address].index, y=df14.loc[df14['id'] == token_address]['Utilization (%)'].tolist()))

    fig14.update_layout(
        title_text='<b>Utilization</b>',
        xaxis_title='Time'
    )

    st.plotly_chart(fig14, use_container_width=True)
    with st.expander("Chart Datails"):
        st.write("""
            Total deposits across all protocol.
         """)
        st.dataframe(df14)
        
        
        
        
st.markdown("""---""")
#### Charts Row 7 ####    
        
    
    
        
column1, column2, column3 = st.columns([4,4,1])
with column3: 
    g15 = choose_granularity('g15')
    
with column1:       
    df15 = markets_snapshots[['inputToken.symbol', 'id', 'rates']].set_index(['inputToken.symbol', 'id'], append=True)
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
        fig15a.add_trace(go.Scatter(name=df15.loc[df15['id'] == token_address].iloc[0,1], x=df15.index, y=df15.loc[df15['id'] == token_address].iloc[:,2], mode='lines'))

    fig15a.update_layout(
        title_text='<b>Borrow Rate</b>',
        xaxis_title='Time'
        )

    st.plotly_chart(fig15a, use_container_width=True)
    
with column2:     
    fig15b = go.Figure()
    for token_address in filter(None, df15['id'].unique().tolist()):
        fig15b.add_trace(go.Scatter(name=df15.loc[df15['id'] == token_address].iloc[0,1], x=df15.index, y=df15.loc[df15['id'] == token_address].iloc[:,3], mode='lines'))

    fig15b.update_layout(
        title_text='<b>Lender Rate</b>',
        xaxis_title='Time'
    )

    st.plotly_chart(fig15b, use_container_width=True)
    
    
with st.expander("Chart Datails"):
        st.write("""
            Total deposits across all protocol.
         """)
        st.dataframe(df15)