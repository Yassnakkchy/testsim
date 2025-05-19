# -*- coding: utf-8 -*-
"""
Created on Mon May 19 09:45:06 2025

@author: nakkachy
"""

# payout_curve_simulator_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pandas.api.types import CategoricalDtype

class PayoutCurveSimulator:
    def __init__(self):
        self.data = None
        self.levels = 6
        self.x_values = [95, 100, 105, 110, 115, 120, 200]  # Attainment breakpoints (%)
        self.y_values = [50, 100, 120, 140, 160, 180, 200]  # Payout breakpoints (%)
        
        st.set_page_config(layout="wide", page_title="Payout Curve Simulator")
        st.title("Payout Curve Simulator")
        
        self.init_sidebar()
        self.init_main_content()
    
    def init_sidebar(self):
        st.sidebar.title("Configuration")
        
        # File upload
        st.sidebar.subheader("Data Source")
        uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx", "xls"])
        if uploaded_file is not None:
            try:
                self.data = pd.read_excel(uploaded_file, sheet_name="Base Work _ Simulation")
                
                required_columns = [
                    'CompPlanID 2025', 'ComponentName 2025', 'Region ', 'BU',
                    'Attainment', 'FY TIN (USD)', 'Earn Scenario 2025', 'PayeeID'
                ]
                
                has_scenario1 = 'Earn Scenario 1' in self.data.columns
                has_scenario2 = 'Earn Scenario 2' in self.data.columns
                has_scenario2A = 'Earn Scenario 2A' in self.data.columns
                has_scenario3 = 'Earn Scenario 3' in self.data.columns
                
                missing_columns = [col for col in required_columns if col not in self.data.columns]
                if missing_columns:
                    st.sidebar.error(f"Missing required columns: {', '.join(missing_columns)}")
                else:
                    st.sidebar.success(f"Loaded {len(self.data)} records")
                    if has_scenario1:
                        st.sidebar.info("Includes Scenario 1 data")
                    if has_scenario2:
                        st.sidebar.info("Includes Scenario 2 data")
                    if has_scenario2A:
                        st.sidebar.info("Includes Scenario 2A data")
                    if has_scenario3:
                        st.sidebar.info("Includes Scenario 3 data")
                    
                    self.update_filter_options()
            except Exception as e:
                st.sidebar.error(f"Failed to load Excel file: {str(e)}")
                self.data = None
        
        # Payout curve configuration
        st.sidebar.subheader("Payout Curve Configuration")
        self.levels = st.sidebar.number_input("Number of Levels", min_value=1, max_value=10, value=self.levels, step=1)
        
        # Update level values if count changes
        if len(self.x_values) < self.levels + 1:
            last_x = self.x_values[-1]
            last_y = self.y_values[-1]
            for i in range(len(self.x_values), self.levels + 1):
                self.x_values.append(last_x + 5)
                self.y_values.append(last_y + 20)
        elif len(self.x_values) > self.levels + 1:
            self.x_values = self.x_values[:self.levels + 1]
            self.y_values = self.y_values[:self.levels + 1]
        
        # Create editable table for levels
        level_data = pd.DataFrame({
            "Attainment (%)": self.x_values,
            "Payout (%)": self.y_values
        })
        edited_level_data = st.sidebar.data_editor(level_data, num_rows="fixed")
        
        # Update values from edited table
        self.x_values = edited_level_data["Attainment (%)"].tolist()
        self.y_values = edited_level_data["Payout (%)"].tolist()
        
        # Data filters
        st.sidebar.subheader("Data Filters")
        
        self.region_filter = st.sidebar.selectbox(
            "Region",
            ["All"] + sorted(self.data['Region '].astype(str).unique()) if self.data is not None else ["All"]
        )
        
        self.bu_filter = st.sidebar.selectbox(
            "BU",
            ["All"] + sorted(self.data['BU'].astype(str).unique()) if self.data is not None else ["All"]
        )
        
        self.component_filter = st.sidebar.selectbox(
            "FrequencyPayout",
            ["All"] + sorted(self.data['FrequencyPayout'].astype(str).unique()) if self.data is not None else ["All"]
        )
        
        if st.sidebar.button("Reset All Filters"):
            self.region_filter = "All"
            self.bu_filter = "All"
            self.component_filter = "All"
    
    def init_main_content(self):
        tab1, tab2, tab3, tab4 = st.tabs(["Payout Curve", "Data Preview", "Analysis", "Pivot Analysis"])
        
        with tab1:
            self.plot_payout_curve()
        
        with tab2:
            if self.data is not None:
                st.subheader("Data Preview")
                st.dataframe(self.data, height=400)
        
        with tab3:
            if self.data is not None:
                self.plot_analysis_chart()
        
        with tab4:
            if self.data is not None:
                self.generate_pivot_analysis()
    
    def update_filter_options(self):
        pass  # Handled directly in Streamlit widgets
    
    def calculate_payout(self, attainment_decimal, x_values, y_values):
        attainment_percentage = attainment_decimal * 100
        
        if attainment_percentage >= 200:
            return 300
        
        if attainment_percentage < 95:
            return 0
        
        for i in range(len(x_values)):
            if abs(attainment_percentage - x_values[i]) < 0.0001: 
                return y_values[i]
        
        for i in range(len(x_values) - 1):
            if x_values[i] < attainment_percentage < x_values[i+1]:
                return y_values[i] + (attainment_percentage - x_values[i]) * \
                       (y_values[i+1] - y_values[i]) / (x_values[i+1] - x_values[i])
        
        return y_values[-1]
    
    def run_simulation(self):
        if self.data is None:
            st.error("Please load an Excel file first")
            return
        
        filtered_data = self.data.copy()
        
        if self.region_filter != "All":
            filtered_data = filtered_data[filtered_data['Region '].astype(str) == self.region_filter]
        
        if self.bu_filter != "All":
            filtered_data = filtered_data[filtered_data['BU'].astype(str) == self.bu_filter]
        
        if self.component_filter != "All":
            filtered_data = filtered_data[filtered_data['FrequencyPayout'].astype(str) == self.component_filter]
        
        if filtered_data.empty:
            st.error("No records match the selected filters")
            return
        
        filtered_data['Payout_Percentage'] = filtered_data['Attainment'].apply(
            lambda x: self.calculate_payout(x, self.x_values, self.y_values))
        
        filtered_data['Simulated_Earn'] = filtered_data['FY TIN (USD)'] * (filtered_data['Payout_Percentage'] / 100)
        
        self.data = filtered_data
        
        return filtered_data
    
    def plot_payout_curve(self):
        if st.button("Run Simulation"):
            filtered_data = self.run_simulation()
            if filtered_data is None:
                return
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Generate points for smooth curve
            x_plot = [0, 94.999]  # Before threshold
            y_plot = [0, 0]
            
            # Add breakpoints
            for i in range(len(self.x_values)):
                x_plot.append(self.x_values[i])
                y_plot.append(self.y_values[i])
                
                # Add midpoint for smooth curve
                if i < len(self.x_values) - 1:
                    x_plot.append((self.x_values[i] + self.x_values[i+1]) / 2)
                    y_plot.append((self.y_values[i] + self.y_values[i+1]) / 2)
            
            # Add cap
            x_plot.extend([200, 250])
            y_plot.extend([300, 300])
            
            # Plot main curve
            ax.plot(x_plot, y_plot, color='#4CAF50', linewidth=2.5, label='Payout Curve')
            
            # Plot breakpoints
            ax.scatter(self.x_values, self.y_values, color='red', s=80, label='Breakpoints', zorder=5)
            
            # Add markers
            ax.axvline(x=95, color='blue', linestyle=':', alpha=0.7)
            ax.text(96, 10, 'Minimum 95%', color='blue')
            
            ax.axhline(y=300, color='orange', linestyle='--', alpha=0.7)
            ax.text(150, 310, '300% Cap', color='orange')
            
            # Styling
            ax.set_xlabel('Attainment (%)')
            ax.set_ylabel('Payout (%)')
            ax.set_title('Payout Curve Simulation')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='upper left')
            ax.set_xlim(0, 220)
            ax.set_ylim(0, 320)
            
            st.pyplot(fig)
    
    def plot_analysis_chart(self):
        filtered_data = self.run_simulation()
        if filtered_data is None:
            return
        
        total_simulated = filtered_data['Simulated_Earn'].sum()
        total_actual = filtered_data['Earn Scenario 2025'].sum()
        
        has_scenario1 = 'Earn Scenario 1' in filtered_data.columns
        has_scenario2 = 'Earn Scenario 2' in filtered_data.columns
        has_scenario2A = 'Earn Scenario 2A' in filtered_data.columns
        has_scenario3 = 'Earn Scenario 3' in filtered_data.columns
        
        total_scenario1 = filtered_data['Earn Scenario 1'].sum() if has_scenario1 else 0
        total_scenario2 = filtered_data['Earn Scenario 2'].sum() if has_scenario2 else 0
        total_scenario2A = filtered_data['Earn Scenario 2A'].sum() if has_scenario2A else 0
        total_scenario3 = filtered_data['Earn Scenario 3'].sum() if has_scenario3 else 0
        
        delta_actual = total_simulated - total_actual
        delta_scenario1 = total_simulated - total_scenario1 if has_scenario1 else 0
        delta_scenario2 = total_simulated - total_scenario2 if has_scenario2 else 0
        delta_scenario2A = total_simulated - total_scenario2A if has_scenario2A else 0
        delta_scenario3 = total_simulated - total_scenario3 if has_scenario3 else 0
        
        # Display results
        st.subheader("Simulation Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Simulated Earnings", f"${total_simulated:,.2f}")
        with col2:
            st.metric("2025 Actual Earnings", f"${total_actual:,.2f}", 
                     f"{delta_actual:,.2f} ({(delta_actual/total_actual*100 if total_actual != 0 else 0):+.2f}%)")
        
        if has_scenario1:
            with col3:
                st.metric("Scenario 1 Earnings", f"${total_scenario1:,.2f}", 
                         f"{delta_scenario1:,.2f} ({(delta_scenario1/total_scenario1*100 if total_scenario1 != 0 else 0):+.2f}%)")
        
        if has_scenario2:
            with col1:
                st.metric("Scenario 2 Earnings", f"${total_scenario2:,.2f}", 
                         f"{delta_scenario2:,.2f} ({(delta_scenario2/total_scenario2*100 if total_scenario2 != 0 else 0):+.2f}%)")
        
        if has_scenario2A:
            with col2:
                st.metric("Scenario 2A Earnings", f"${total_scenario2A:,.2f}", 
                         f"{delta_scenario2A:,.2f} ({(delta_scenario2A/total_scenario2A*100 if total_scenario2A != 0 else 0):+.2f}%)")
        
        if has_scenario3:
            with col3:
                st.metric("Scenario 3 Earnings", f"${total_scenario3:,.2f}", 
                         f"{delta_scenario3:,.2f} ({(delta_scenario3/total_scenario3*100 if total_scenario3 != 0 else 0):+.2f}%)")
        
        # Plot comparison chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scenarios = ['Simulated', 'Actual 2025']
        values = [total_simulated, total_actual]
        deltas = [0, delta_actual]
        colors = ['#4CAF50', '#2196F3']  # Green, Blue
        
        if has_scenario1:
            scenarios.append('Scenario 1')
            values.append(total_scenario1)
            deltas.append(delta_scenario1)
            colors.append('#FF9800')  # Orange
        
        if has_scenario2:
            scenarios.append('Scenario 2')
            values.append(total_scenario2)
            deltas.append(delta_scenario2)
            colors.append('#9C27B0')  # Purple
            
        if has_scenario2A:
            scenarios.append('Scenario 2A')
            values.append(total_scenario2A)
            deltas.append(delta_scenario2A)
            colors.append('#607D8B')  # Blue Grey
            
        if has_scenario3:
            scenarios.append('Scenario 3')
            values.append(total_scenario3)
            deltas.append(delta_scenario3)
            colors.append('#E91E63')  # Pink
        
        # Create bars
        x = range(len(scenarios))
        bars = ax.bar(x, values, color=colors)
        
        # Add delta annotations
        for i, (val, delta) in enumerate(zip(values, deltas)):
            if delta != 0:  # Skip simulated (no delta)
                ax.text(i, val + (0.05 * max(values)),
                       f"Δ: ${delta:,.2f}",
                       ha='center', va='bottom', fontsize=10,
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios)
        ax.set_ylabel('Earnings (USD)')
        ax.set_title('Earnings Scenario Comparison')
        ax.grid(True, linestyle='--', alpha=0.6, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.2f}',
                    ha='center', va='bottom', fontsize=10)
        
        if len(scenarios) > 2:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        st.pyplot(fig)
    
    def generate_pivot_analysis(self):
        filtered_data = self.run_simulation()
        if filtered_data is None:
            return
        
        x_values_sorted = sorted(self.x_values)
        
        bins = [0] + x_values_sorted + [float('inf')]
        labels = [f"<{x_values_sorted[0]}%"] + \
                 [f"{x_values_sorted[i]}-{x_values_sorted[i+1]}%" for i in range(len(x_values_sorted)-1)] + \
                 [f"≥{x_values_sorted[-1]}%"]
        
        filtered_data['Attainment Tier'] = pd.cut(filtered_data['Attainment'] * 100,
                                                bins=bins,
                                                labels=labels,
                                                right=False)
        
        # 1. Earnings comparison matrix
        st.subheader("Earnings Comparison by Tier")
        
        earnings_cols = ['Earn Scenario 2025', 'Simulated_Earn']
        if 'Earn Scenario 1' in filtered_data.columns:
            earnings_cols.append('Earn Scenario 1')
        if 'Earn Scenario 2' in filtered_data.columns:
            earnings_cols.append('Earn Scenario 2')
        if 'Earn Scenario 2A' in filtered_data.columns:
            earnings_cols.append('Earn Scenario 2A')
        if 'Earn Scenario 3' in filtered_data.columns:
            earnings_cols.append('Earn Scenario 3')
        
        earnings_pivot = filtered_data.groupby('Attainment Tier')[earnings_cols].sum().reset_index()
        
        # Calculate deltas
        for col in earnings_cols[1:]:
            earnings_pivot[f'Delta vs {col}'] = earnings_pivot['Simulated_Earn'] - earnings_pivot[col]
        
        # Format currency
        for col in earnings_cols + [c for c in earnings_pivot.columns if 'Delta' in c]:
            earnings_pivot[col] = earnings_pivot[col].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")
        
        st.dataframe(earnings_pivot)
        
        # 2. Payee distribution matrix
        st.subheader("Payee Distribution by Tier")
        
        payee_cols = ['PayeeID']
        if 'Earn Scenario 1' in filtered_data.columns:
            payee_cols.append('Earn Scenario 1')
        if 'Earn Scenario 2' in filtered_data.columns:
            payee_cols.append('Earn Scenario 2')
        if 'Earn Scenario 2A' in filtered_data.columns:
            payee_cols.append('Earn Scenario 2A')
        if 'Earn Scenario 3' in filtered_data.columns:
            payee_cols.append('Earn Scenario 3')
        
        # Create a function to count distinct payees per tier
        def count_distinct_payees(group):
            result = {'PayeeID': group['PayeeID'].nunique()}
            
            # For each scenario, count payees with earnings > 0
            if 'Earn Scenario 1' in group.columns:
                result['Scenario 1 Payees'] = group[group['Earn Scenario 1'] > 0]['PayeeID'].nunique()
            if 'Earn Scenario 2' in group.columns:
                result['Scenario 2 Payees'] = group[group['Earn Scenario 2'] > 0]['PayeeID'].nunique()
            if 'Earn Scenario 2A' in group.columns:
                result['Scenario 2A Payees'] = group[group['Earn Scenario 2A'] > 0]['PayeeID'].nunique()
            if 'Earn Scenario 3' in group.columns:
                result['Scenario 3 Payees'] = group[group['Earn Scenario 3'] > 0]['PayeeID'].nunique()
            
            return pd.Series(result)
        
        payee_pivot = filtered_data.groupby('Attainment Tier').apply(count_distinct_payees).reset_index()
        
        # Calculate total distinct payees (should be 800)
        total_payees = payee_pivot['PayeeID'].sum()
        st.info(f"Total distinct PayeeIDs: {total_payees}")
        
        st.dataframe(payee_pivot)

if __name__ == "__main__":
    app = PayoutCurveSimulator()
