import streamlit as st
import sys
import subprocess

# Try to import plotly, install if not available
try:
    import plotly.graph_objects as go
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
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

# Create tabs for different scenarios
tab1, tab2 = st.tabs(["Ideal Scenario", "Realistic Scenario"])

with tab1:
    # Create two columns for inputs
    col1, col2 = st.columns(2)

    with col1:
        # Input for initial balance
        initial_balance = st.number_input("Initial Balance ($)", min_value=1.0, value=1000.0, step=100.0, key="ideal_balance")
        
        # Input for minimum daily profit percentage
        min_profit_percentage = st.number_input("Minimum Daily Profit (%)", min_value=0.0, max_value=100.0, value=0.5, step=0.1, key="ideal_min_profit")
        
        # Input for maximum daily profit percentage
        max_profit_percentage = st.number_input("Maximum Daily Profit (%)", min_value=0.0, max_value=100.0, value=2.0, step=0.1, key="ideal_max_profit")

    with col2:
        # Input for number of days
        days = st.number_input("Number of Days", min_value=1, max_value=365, value=30, step=1, key="ideal_days")
        
        # Input for calculation method
        calculation_method = st.selectbox(
            "Calculation Method",
            ["Random (between min and max)", "Average (fixed middle value)", "Compound with fixed min", "Compound with fixed max"],
            key="ideal_method"
        )
        
        # Option to exclude weekends
        exclude_weekends = st.checkbox("Exclude Weekends", value=True, key="ideal_weekends")

    # Calculate button
    calculate_button = st.button("Calculate Ideal Scenario", use_container_width=True, key="ideal_calculate")

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
        st.markdown("<h2 class='sub-header'>Ideal Scenario Results</h2>", unsafe_allow_html=True)
        
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
                'text': 'Ideal Forex Account Balance Growth Projection',
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
        # Format date column safely
        if pd.api.types.is_datetime64_any_dtype(display_df['Date']):
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
        else:
            display_df['Date'] = pd.to_datetime(display_df['Date']).dt.strftime('%Y-%m-%d')
        display_df['Balance'] = display_df['Balance'].map('${:,.2f}'.format)
        display_df['Daily Profit'] = display_df['Daily Profit'].map('${:,.2f}'.format)
        display_df['Daily Profit %'] = display_df['Daily Profit %'].map('{:,.2f}%'.format)
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download button removed
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name=f"forex_ideal_projection_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

with tab2:
    st.markdown("<h2 class='sub-header'>Realistic Trading Scenario</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
    <p>در دنیای واقعی معامله‌گری فارکس، تریدرها با چالش‌های متعددی روبرو می‌شوند:</p>
    <ul>
        <li>روزهای زیان‌ده در کنار روزهای سودده</li>
        <li>احتمال ضررهای متوالی (Drawdown)</li>
        <li>تأثیر احساسات و روانشناسی بر تصمیم‌گیری</li>
        <li>مدیریت ریسک ناکافی</li>
    </ul>
    <p>این بخش یک شبیه‌سازی واقع‌بینانه‌تر از نتایج احتمالی معاملات را نشان می‌دهد.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for inputs
    col1, col2 = st.columns(2)

    with col1:
        # Input for initial balance
        realistic_initial_balance = st.number_input("Initial Balance ($)", min_value=1.0, value=1000.0, step=100.0, key="real_balance")
        
        # Input for win rate
        win_rate = st.slider("Win Rate (%)", min_value=30, max_value=70, value=45, step=5, 
                            help="Percentage of profitable trades. Professional traders typically have 40-60% win rates.")
        
        # Input for risk per trade
        risk_per_trade = st.slider("Risk Per Trade (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5,
                                 help="Percentage of account risked on each trade. Professional traders typically risk 1-2% per trade.")

    with col2:
        # Input for number of days
        realistic_days = st.number_input("Number of Days", min_value=1, max_value=365, value=30, step=1, key="real_days")
        
        # Input for risk-reward ratio
        risk_reward = st.slider("Risk-Reward Ratio", min_value=1.0, max_value=3.0, value=1.5, step=0.1,
                               help="Ratio of potential profit to potential loss. Higher values mean larger profits when winning.")
        
        # Option to exclude weekends
        realistic_exclude_weekends = st.checkbox("Exclude Weekends", value=True, key="real_weekends")
        
        # Option to include drawdown periods
        include_drawdown = st.checkbox("Include Realistic Drawdown Periods", value=True,
                                     help="Simulate periods of consecutive losses that occur in real trading")

    # Calculate button
    realistic_calculate_button = st.button("Calculate Realistic Scenario", use_container_width=True, key="real_calculate")

    if realistic_calculate_button:
        # Create date range
        start_date = datetime.now().date()
        realistic_date_range = []
        realistic_balance_values = []
        realistic_daily_changes = []
        realistic_daily_percentages = []
        trade_results = []  # Win or Loss
        
        current_balance = realistic_initial_balance
        
        # Simulate drawdown periods
        if include_drawdown:
            # Create a more realistic pattern with streaks of wins and losses
            drawdown_probability = 0.2  # 20% chance of entering a drawdown period
            in_drawdown = False
            drawdown_length = 0
            max_drawdown_length = 5  # Maximum consecutive losing days
        
        for i in range(realistic_days):
            current_date = start_date + timedelta(days=i)
            
            # Skip weekends if option is selected
            if realistic_exclude_weekends and current_date.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
                continue
                
            realistic_date_range.append(current_date)
            
            # Determine if trade is a win or loss
            if include_drawdown:
                # Check if we should enter a drawdown period
                if not in_drawdown and np.random.random() < drawdown_probability:
                    in_drawdown = True
                    drawdown_length = np.random.randint(2, max_drawdown_length + 1)
                
                # If in drawdown, force losses until drawdown period ends
                if in_drawdown:
                    is_win = False
                    drawdown_length -= 1
                    if drawdown_length <= 0:
                        in_drawdown = False
                else:
                    # Normal win/loss probability
                    is_win = np.random.random() < (win_rate / 100)
            else:
                # Simple random win/loss based on win rate
                is_win = np.random.random() < (win_rate / 100)
            
            # Calculate daily change based on win/loss
            if is_win:
                daily_change_percentage = risk_per_trade * risk_reward  # Win gives R:R times the risk
                trade_results.append("Win")
            else:
                daily_change_percentage = -risk_per_trade  # Loss is just the risk amount
                trade_results.append("Loss")
                
            daily_change = current_balance * (daily_change_percentage / 100)
            realistic_daily_changes.append(daily_change)
            realistic_daily_percentages.append(daily_change_percentage)
            
            current_balance += daily_change
            realistic_balance_values.append(current_balance)
        
        # Create DataFrame for display
        realistic_df = pd.DataFrame({
            'Date': realistic_date_range,
            'Balance': realistic_balance_values,
            'Daily Change': realistic_daily_changes,
            'Daily Change %': realistic_daily_percentages,
            'Trade Result': trade_results
        })
        
        # Calculate drawdown
        realistic_df['Drawdown'] = realistic_initial_balance - realistic_df['Balance'].cummin()
        realistic_df['Drawdown %'] = (realistic_df['Drawdown'] / realistic_initial_balance) * 100
        max_drawdown = realistic_df['Drawdown'].max()
        max_drawdown_percent = realistic_df['Drawdown %'].max()
        
        # Display results
        st.markdown("<h2 class='sub-header'>Realistic Scenario Results</h2>", unsafe_allow_html=True)
        
        # Create metrics for key information
        metric_cols = st.columns(4)
        with metric_cols[0]:
            final_balance = realistic_balance_values[-1]
            delta = final_balance - realistic_initial_balance
            delta_color = "normal" if delta >= 0 else "inverse"
            st.metric("Final Balance", f"${final_balance:.2f}", f"{delta:.2f}", delta_color=delta_color)
        with metric_cols[1]:
            profit = final_balance - realistic_initial_balance
            profit_percent = (profit / realistic_initial_balance) * 100
            delta_color = "normal" if profit >= 0 else "inverse"
            st.metric("Total Profit/Loss", f"${profit:.2f}", f"{profit_percent:.2f}%", delta_color=delta_color)
        with metric_cols[2]:
            win_count = trade_results.count("Win")
            actual_win_rate = (win_count / len(trade_results)) * 100
            st.metric("Actual Win Rate", f"{actual_win_rate:.1f}%", f"{len(trade_results)} trades")
        with metric_cols[3]:
            st.metric("Max Drawdown", f"${max_drawdown:.2f}", f"{max_drawdown_percent:.2f}%", delta_color="inverse")
        
        # Create beautiful chart with Plotly
        fig = go.Figure()
        
        # Add balance line
        fig.add_trace(go.Scatter(
            x=realistic_date_range,
            y=realistic_balance_values,
            mode='lines',
            name='Account Balance',
            line=dict(color='#1E88E5', width=3),
            hovertemplate='Date: %{x}<br>Balance: $%{y:.2f}<extra></extra>'
        ))
        
        # Add win/loss markers
        for result in ["Win", "Loss"]:
            mask = [r == result for r in trade_results]
            marker_color = '#2E7D32' if result == "Win" else '#C62828'
            marker_symbol = 'triangle-up' if result == "Win" else 'triangle-down'
            
            fig.add_trace(go.Scatter(
                x=[realistic_date_range[i] for i in range(len(mask)) if mask[i]],
                y=[realistic_balance_values[i] for i in range(len(mask)) if mask[i]],
                mode='markers',
                name=result,
                marker=dict(
                    color=marker_color,
                    size=10,
                    symbol=marker_symbol
                ),
                hovertemplate='Date: %{x}<br>Balance: $%{y:.2f}<br>Result: ' + result + '<extra></extra>'
            ))
        
        # Add initial balance reference line
        fig.add_trace(go.Scatter(
            x=[realistic_date_range[0], realistic_date_range[-1]],
            y=[realistic_initial_balance, realistic_initial_balance],
            mode='lines',
            name='Initial Balance',
            line=dict(color='#E53935', width=2, dash='dash'),
            hovertemplate='Initial Balance: $%{y:.2f}<extra></extra>'
        ))
        
        # Customize layout
        fig.update_layout(
            title={
                'text': 'Realistic Forex Account Balance Projection',
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
        
        # Add drawdown chart
        fig_drawdown = go.Figure()
        
        fig_drawdown.add_trace(go.Scatter(
            x=realistic_date_range,
            y=realistic_df['Drawdown %'],
            mode='lines',
            name='Drawdown %',
            fill='tozeroy',
            line=dict(color='#C62828', width=2),
            hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
        ))
        
        fig_drawdown.update_layout(
            title={
                'text': 'Account Drawdown Over Time',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=24, color='#C62828')
            },
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            yaxis=dict(autorange="reversed"),  # Invert y-axis for better visualization
            hovermode='x unified',
            height=400,
            template='plotly_white',
            margin=dict(l=20, r=20, t=100, b=20),
        )
        
        st.plotly_chart(fig_drawdown, use_container_width=True)
        
        # Display statistics
        st.markdown("<h2 class='sub-header'>Trading Statistics</h2>", unsafe_allow_html=True)
        
        stat_cols = st.columns(3)
        with stat_cols[0]:
            st.metric("Win/Loss Ratio", f"{win_count}:{len(trade_results)-win_count}")
            st.metric("Profit Factor", f"{abs(sum([c for c in realistic_daily_changes if c > 0]) / sum([c for c in realistic_daily_changes if c < 0])):.2f}")
        
        with stat_cols[1]:
            st.metric("Average Win", f"${np.mean([c for c in realistic_daily_changes if c > 0]):.2f}")
            st.metric("Average Loss", f"${np.mean([c for c in realistic_daily_changes if c < 0]):.2f}")
        
        with stat_cols[2]:
            st.metric("Largest Win", f"${max([c for c in realistic_daily_changes if c > 0], default=0):.2f}")
            st.metric("Largest Loss", f"${min([c for c in realistic_daily_changes if c < 0], default=0):.2f}")
        
        # Display realistic trading advice
        st.markdown("""
        <div style="background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0;">
        <h3 style="color: #0D47A1;">واقعیت‌های معامله‌گری فارکس</h3>
        <ul>
            <li><strong>مدیریت ریسک:</strong> تریدرهای موفق معمولاً در هر معامله فقط 1-2% از سرمایه خود را ریسک می‌کنند.</li>
            <li><strong>نرخ موفقیت:</strong> حتی تریدرهای حرفه‌ای معمولاً نرخ موفقیت 40-60% دارند، نه 90-100%.</li>
            <li><strong>دوره‌های افت (Drawdown):</strong> همه تریدرها دوره‌های افت را تجربه می‌کنند. مدیریت این دوره‌ها کلید موفقیت است.</li>
            <li><strong>نسبت ریسک به پاداش:</strong> داشتن نسبت ریسک به پاداش مناسب (معمولاً 1:2 یا بالاتر) به شما امکان می‌دهد حتی با نرخ موفقیت کمتر از 50% سودآور باشید.</li>
            <li><strong>انتظارات واقع‌بینانه:</strong> رشد پایدار 5-15% ماهانه در بلندمدت بسیار عالی است، نه 100% یا بیشتر.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Display detailed data table
        st.markdown("<h2 class='sub-header'>Detailed Data</h2>", unsafe_allow_html=True)
        
        # Format the DataFrame for display
        display_df = realistic_df.copy()
        # Format date column safely
        if pd.api.types.is_datetime64_any_dtype(display_df['Date']):
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
        else:
            display_df['Date'] = pd.to_datetime(display_df['Date']).dt.strftime('%Y-%m-%d')
        display_df['Balance'] = display_df['Balance'].map('${:,.2f}'.format)
        display_df['Daily Change'] = display_df['Daily Change'].map('${:,.2f}'.format)
        display_df['Daily Change %'] = display_df['Daily Change %'].map('{:,.2f}%'.format)
        display_df['Drawdown'] = display_df['Drawdown'].map('${:,.2f}'.format)
        display_df['Drawdown %'] = display_df['Drawdown %'].map('{:,.2f}%'.format)
        
        st.dataframe(display_df, use_container_width=True)
        
        # Add download button for CSV
        csv = realistic_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Realistic Data as CSV",
            data=csv,
            file_name=f"forex_realistic_projection_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
