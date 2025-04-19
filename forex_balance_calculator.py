import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Forex Balance Calculator",
    page_icon="💹",
    layout="wide"
)

# Add custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-bottom: 1rem;
    }
    .result-text {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2E7D32;
    }
    .stApp {
        background-color: #F5F7FA;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown("<h1 class='main-header'>Forex Account Balance Calculator</h1>", unsafe_allow_html=True)

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    # Input for initial balance
    initial_balance = st.number_input("Initial Balance ($)", min_value=1.0, value=1000.0, step=100.0)
    
    # Input for minimum daily profit percentage
    min_profit_percentage = st.number_input("Minimum Daily Profit (%)", min_value=0.0, max_value=100.0, value=0.5, step=0.1)
    
    # Input for maximum daily profit percentage
    max_profit_percentage = st.number_input("Maximum Daily Profit (%)", min_value=0.0, max_value=100.0, value=2.0, step=0.1)

with col2:
    # Input for number of days
    days = st.number_input("Number of Days", min_value=1, max_value=365, value=30, step=1)
    
    # Input for calculation method
    calculation_method = st.selectbox(
        "Calculation Method",
        ["Random (between min and max)", "Average (fixed middle value)", "Compound with fixed min", "Compound with fixed max"]
    )
    
    # Option to exclude weekends
    exclude_weekends = st.checkbox("Exclude Weekends", value=True)

# Calculate button
calculate_button = st.button("Calculate", use_container_width=True)

if calculate_button:
    # Create date range
    start_date = datetime.now().date()
    date_range = []
    balance_values = []
    daily_profits = []
    
    current_balance = initial_balance
    
    for i in range(days):
        current_date = start_date + timedelta(days=i)
        
        # Skip weekends if option is selected
        if exclude_weekends and current_date.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
            continue
            
        date_range.append(current_date)
        
        # Calculate daily profit based on selected method
        if calculation_method == "Random (between min and max)":
            daily_profit_percentage = np.random.uniform(min_profit_percentage, max_profit_percentage)
        elif calculation_method == "Average (fixed middle value)":
            daily_profit_percentage = (min_profit_percentage + max_profit_percentage) / 2
        elif calculation_method == "Compound with fixed min":
            daily_profit_percentage = min_profit_percentage
        else:  # "Compound with fixed max"
            daily_profit_percentage = max_profit_percentage
            
        daily_profit = current_balance * (daily_profit_percentage / 100)
        daily_profits.append(daily_profit)
        
        current_balance += daily_profit
        balance_values.append(current_balance)
    
    # Create DataFrame for display
    df = pd.DataFrame({
        'Date': date_range,
        'Balance': balance_values,
        'Daily Profit': daily_profits,
        'Daily Profit %': [p / (b - p) * 100 for p, b in zip(daily_profits, balance_values)]
    })
    
    # Display results
    st.markdown("<h2 class='sub-header'>Results</h2>", unsafe_allow_html=True)
    
    # Create metrics for key information
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Final Balance", f"${balance_values[-1]:.2f}", f"{balance_values[-1] - initial_balance:.2f}")
    with metric_cols[1]:
        st.metric("Total Profit", f"${balance_values[-1] - initial_balance:.2f}", f"{(balance_values[-1] - initial_balance) / initial_balance * 100:.2f}%")
    with metric_cols[2]:
        st.metric("Trading Days", f"{len(date_range)}")
    with metric_cols[3]:
        st.metric("Avg. Daily Profit", f"${np.mean(daily_profits):.2f}", f"{np.mean(daily_profits) / initial_balance * 100:.2f}%")
    
    # Create beautiful chart with Plotly
    fig = go.Figure()
    
    # Add balance line
    fig.add_trace(go.Scatter(
        x=date_range,
        y=balance_values,
        mode='lines+markers',
        name='Account Balance',
        line=dict(color='#1E88E5', width=3),
        marker=dict(size=8, color='#0D47A1'),
        hovertemplate='Date: %{x}<br>Balance: $%{y:.2f}<extra></extra>'
    ))
    
    # Add initial balance reference line
    fig.add_trace(go.Scatter(
        x=[date_range[0], date_range[-1]],
        y=[initial_balance, initial_balance],
        mode='lines',
        name='Initial Balance',
        line=dict(color='#E53935', width=2, dash='dash'),
        hovertemplate='Initial Balance: $%{y:.2f}<extra></extra>'
    ))
    
    # Customize layout
    fig.update_layout(
        title={
            'text': 'Forex Account Balance Growth Projection',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24, color='#0D47A1')
        },
        xaxis_title='Date',
        yaxis_title='Balance ($)',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=600,
        template='plotly_white',
        margin=dict(l=20, r=20, t=100, b=20),
    )
    
    # Add range slider
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=14, label="2w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Display detailed data table
    st.markdown("<h2 class='sub-header'>Detailed Data</h2>", unsafe_allow_html=True)
    
    # Format the DataFrame for display
    display_df = df.copy()
    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
    display_df['Balance'] = display_df['Balance'].map('${:,.2f}'.format)
    display_df['Daily Profit'] = display_df['Daily Profit'].map('${:,.2f}'.format)
    display_df['Daily Profit %'] = display_df['Daily Profit %'].map('{:,.2f}%'.format)
    
    st.dataframe(display_df, use_container_width=True)
    
    # Add download button for CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name=f"forex_balance_projection_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )
