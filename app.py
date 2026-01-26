"""
MarketPulse AI - Demo Version

Standalone Streamlit app for showcasing MarketPulse AI capabilities.
Perfect for deployment on Streamlit Cloud, Railway, Render, or Heroku.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import numpy as np
from typing import Dict, List, Any
import json

# Page configuration
st.set_page_config(
    page_title="MarketPulse AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .insight-card {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .recommendation-card {
        background: #e8f5e8;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 MarketPulse AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Retail Decision Support System</p>', unsafe_allow_html=True)
    
    # Demo notice
    st.info("🎯 **Demo Version** - This showcases MarketPulse AI capabilities with simulated data and AI responses!")
    
    # Sidebar navigation
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["📊 Dashboard", "📈 Data Upload", "🔍 AI Insights", "💡 Recommendations", "📊 Scenario Analysis", "⚠️ Risk Assessment"]
    )
    
    # Main content based on selected page
    if page == "📊 Dashboard":
        show_dashboard()
    elif page == "📈 Data Upload":
        show_data_upload()
    elif page == "🔍 AI Insights":
        show_insights()
    elif page == "💡 Recommendations":
        show_recommendations()
    elif page == "📊 Scenario Analysis":
        show_scenario_analysis()
    elif page == "⚠️ Risk Assessment":
        show_risk_assessment()

def show_dashboard():
    """Display the main dashboard."""
    st.markdown('<h2 class="section-header">📊 System Dashboard</h2>', unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔧 Components", "12")
    with col2:
        st.metric("📊 API Version", "1.0.0")
    with col3:
        st.metric("🟢 Status", "Demo Mode")
    with col4:
        st.metric("⏰ Uptime", "Running")
    
    # System overview
    st.markdown('<h3 class="section-header">🎯 MarketPulse AI Features</h3>', unsafe_allow_html=True)
    
    features = {
        "🔍 AI Insights": "Generate intelligent analysis of sales patterns and trends",
        "💡 Smart Recommendations": "Get AI-powered business suggestions for optimization",
        "📊 Scenario Analysis": "Compare different business strategies and outcomes",
        "⚠️ Risk Assessment": "Identify potential inventory and demand risks",
        "🛡️ Compliance Validation": "Ensure MRP compliance for all pricing strategies",
        "📈 Data Processing": "Real-time sales data analysis and pattern recognition"
    }
    
    for feature, description in features.items():
        with st.container():
            st.markdown(f"### {feature}")
            st.write(description)
            st.markdown("---")

def show_data_upload():
    """Display the data upload interface."""
    st.markdown('<h2 class="section-header">📈 Upload Sales Data</h2>', unsafe_allow_html=True)
    
    st.info("💡 In the full version, you can upload real sales data. This demo shows the interface and generates sample insights!")
    
    # Upload method selection
    upload_method = st.radio(
        "Choose upload method:",
        ["📝 Manual Entry", "📁 CSV Upload", "📊 Sample Data"]
    )
    
    if upload_method == "📝 Manual Entry":
        show_manual_data_entry()
    elif upload_method == "📁 CSV Upload":
        show_csv_upload()
    elif upload_method == "📊 Sample Data":
        show_sample_data_upload()

def show_manual_data_entry():
    """Show manual data entry form."""
    st.markdown("### 📝 Enter Sales Data Manually")
    
    with st.form("manual_data_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            product_id = st.text_input("🏷️ Product ID", placeholder="e.g., PROD001")
            quantity_sold = st.number_input("📦 Quantity Sold", min_value=0, value=0)
            inventory_level = st.number_input("📊 Current Inventory", min_value=0, value=0)
        
        with col2:
            sale_date = st.date_input("📅 Sale Date", value=date.today())
            price = st.number_input("💰 Price (₹)", min_value=0.0, value=0.0, format="%.2f")
            revenue = st.number_input("💵 Revenue (₹)", min_value=0.0, value=0.0, format="%.2f")
        
        submitted = st.form_submit_button("📤 Upload Data (Demo)", type="primary")
        
        if submitted:
            if product_id and quantity_sold >= 0 and price > 0:
                st.success("✅ Demo: Data uploaded successfully!")
                st.balloons()
                
                # Show what the data would look like
                demo_data = {
                    "Product ID": product_id,
                    "Date": sale_date.strftime("%Y-%m-%d"),
                    "Quantity": quantity_sold,
                    "Price": f"₹{price:,.2f}",
                    "Revenue": f"₹{revenue if revenue > 0 else quantity_sold * price:,.2f}",
                    "Inventory": inventory_level
                }
                
                st.markdown("#### 📊 Data Preview")
                st.json(demo_data)
            else:
                st.error("❌ Please fill in all required fields")

def show_csv_upload():
    """Show CSV upload interface."""
    st.markdown("### 📁 Upload CSV File")
    
    # Show expected format
    st.info("📋 Your CSV should have these columns: product_id, date, quantity_sold, revenue, inventory_level, price")
    
    # Sample CSV format
    sample_df = pd.DataFrame({
        "product_id": ["PROD001", "PROD002", "PROD003"],
        "date": ["2024-01-15", "2024-01-15", "2024-01-16"],
        "quantity_sold": [150, 75, 200],
        "revenue": [15000.0, 7500.0, 20000.0],
        "inventory_level": [500, 200, 800],
        "price": [100.0, 100.0, 100.0]
    })
    
    with st.expander("📋 See Expected CSV Format"):
        st.dataframe(sample_df)
    
    # File upload
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File loaded successfully! Found {len(df)} records.")
            
            # Show preview
            st.markdown("#### 👀 Data Preview")
            st.dataframe(df.head())
            
            # Upload button
            if st.button("📤 Upload to MarketPulse AI (Demo)", type="primary"):
                st.success(f"✅ Demo: Successfully processed {len(df)} records!")
                st.balloons()
        
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")

def show_sample_data_upload():
    """Show sample data upload."""
    st.markdown("### 📊 Use Sample Data")
    
    st.info("💡 This demo generates sample retail data to show how MarketPulse AI works!")
    
    # Generate and show sample data
    sample_data = generate_sample_data()
    sample_df = pd.DataFrame(sample_data)
    
    st.markdown("#### 👀 Sample Data Preview")
    st.dataframe(sample_df)
    
    if st.button("📤 Load Sample Data", type="primary"):
        st.success("✅ Sample data loaded successfully!")
        st.info("🎯 Now you can try the AI Insights and Recommendations features!")
        st.balloons()
        
        # Store in session state for other pages
        st.session_state['sample_data'] = sample_df

def generate_sample_data():
    """Generate sample sales data."""
    products = ["MOBILE_001", "LAPTOP_001", "TABLET_001", "HEADPHONES_001", "CAMERA_001"]
    
    sample_data = []
    base_date = datetime.now() - timedelta(days=30)
    
    for i in range(20):
        product = np.random.choice(products)
        date = base_date + timedelta(days=np.random.randint(0, 30))
        quantity = np.random.randint(1, 20)
        price = np.random.uniform(5000, 50000)
        
        sample_data.append({
            "product_id": product,
            "date": date.strftime("%Y-%m-%d"),
            "quantity_sold": quantity,
            "revenue": quantity * price,
            "inventory_level": np.random.randint(50, 500),
            "price": price
        })
    
    return sample_data

def show_insights():
    """Display AI insights interface."""
    st.markdown('<h2 class="section-header">🔍 AI Insights</h2>', unsafe_allow_html=True)
    
    st.info("🧠 This demo shows how MarketPulse AI generates intelligent insights from your sales data!")
    
    with st.form("insights_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            product_ids = st.text_area(
                "🏷️ Product IDs (one per line)",
                placeholder="MOBILE_001\nLAPTOP_001\nTABLET_001",
                value="MOBILE_001\nLAPTOP_001"
            )
            include_seasonal = st.checkbox("🌟 Include Seasonal Analysis", value=True)
        
        with col2:
            confidence_threshold = st.slider("🎯 Confidence Threshold", 0.0, 1.0, 0.7, 0.1)
            max_insights = st.number_input("📊 Maximum Insights", min_value=1, max_value=50, value=10)
        
        submitted = st.form_submit_button("🔍 Generate Insights (Demo)", type="primary")
        
        if submitted:
            if product_ids.strip():
                product_list = [pid.strip() for pid in product_ids.split('\n') if pid.strip()]
                display_demo_insights(product_list, confidence_threshold, include_seasonal)
            else:
                st.error("❌ Please enter at least one product ID")

def display_demo_insights(product_ids: List[str], confidence: float, seasonal: bool):
    """Display demo insights results."""
    st.markdown('<h3 class="section-header">📊 AI Insights Results</h3>', unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔍 Insights Generated", len(product_ids) * 2)
    with col2:
        st.metric("📊 Patterns Analyzed", 15)
    with col3:
        st.metric("🎯 Confidence Threshold", f"{confidence:.1%}")
    with col4:
        st.metric("🌟 Seasonal Analysis", "✅" if seasonal else "❌")
    
    # Generate demo insights
    demo_insights = []
    for product in product_ids:
        demo_insights.extend([
            {
                "insight_text": f"Product {product} shows strong weekend sales pattern",
                "confidence_level": np.random.uniform(0.7, 0.95),
                "key_factors": ["day_of_week", "customer_behavior"],
                "supporting_evidence": "Sales increase by 35% on Saturdays and Sundays"
            },
            {
                "insight_text": f"Inventory turnover for {product} is optimal during festival seasons",
                "confidence_level": np.random.uniform(0.6, 0.9),
                "key_factors": ["seasonal_events", "demand_patterns"],
                "supporting_evidence": "Diwali and Christmas periods show 60% higher turnover"
            }
        ])
    
    # Display insights
    st.markdown('<h3 class="section-header">💡 Key Insights</h3>', unsafe_allow_html=True)
    
    for i, insight in enumerate(demo_insights, 1):
        with st.expander(f"🔍 Insight #{i}", expanded=True):
            st.write(f"**Finding:** {insight['insight_text']}")
            st.write(f"**Confidence:** {insight['confidence_level']:.1%}")
            st.write(f"**Key Factors:** {', '.join(insight['key_factors'])}")
            st.write(f"**Evidence:** {insight['supporting_evidence']}")
            st.markdown("---")
    
    # Create visualization
    create_demo_insights_visualization(demo_insights)

def create_demo_insights_visualization(insights: List[Dict]):
    """Create demo visualizations for insights."""
    st.markdown('<h3 class="section-header">📈 Insights Visualization</h3>', unsafe_allow_html=True)
    
    # Confidence distribution
    confidences = [insight['confidence_level'] for insight in insights]
    
    fig = px.histogram(
        x=confidences,
        nbins=5,
        title="📊 Confidence Level Distribution",
        labels={'x': 'Confidence Level', 'y': 'Number of Insights'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Key factors analysis
    all_factors = []
    for insight in insights:
        all_factors.extend(insight['key_factors'])
    
    factor_counts = pd.Series(all_factors).value_counts()
    
    fig = px.bar(
        x=factor_counts.values,
        y=factor_counts.index,
        orientation='h',
        title="🔑 Most Important Factors",
        labels={'x': 'Frequency', 'y': 'Factors'}
    )
    st.plotly_chart(fig, use_container_width=True)

def show_recommendations():
    """Display recommendations interface."""
    st.markdown('<h2 class="section-header">💡 Smart Recommendations</h2>', unsafe_allow_html=True)
    
    st.info("🎯 This demo shows how MarketPulse AI generates intelligent business recommendations!")
    
    with st.form("recommendations_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            product_ids = st.text_area(
                "🏷️ Product IDs (one per line)",
                placeholder="MOBILE_001\nLAPTOP_001",
                value="MOBILE_001\nLAPTOP_001"
            )
            priority_filter = st.selectbox("⭐ Priority Filter", ["All", "High", "Medium", "Low"])
        
        with col2:
            target_margin = st.slider("🎯 Target Margin (%)", 0, 50, 25)
            inventory_turnover = st.number_input("🔄 Inventory Turnover Target", min_value=1, max_value=24, value=12)
        
        submitted = st.form_submit_button("💡 Generate Recommendations (Demo)", type="primary")
        
        if submitted:
            if product_ids.strip():
                product_list = [pid.strip() for pid in product_ids.split('\n') if pid.strip()]
                display_demo_recommendations(product_list, target_margin, priority_filter)
            else:
                st.error("❌ Please enter at least one product ID")

def display_demo_recommendations(product_ids: List[str], margin: int, priority: str):
    """Display demo recommendations results."""
    st.markdown('<h3 class="section-header">🎯 Recommendation Results</h3>', unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💡 Recommendations", len(product_ids) * 3)
    with col2:
        st.metric("📊 Products Analyzed", len(product_ids))
    with col3:
        st.metric("🛡️ Compliance Checked", "✅")
    with col4:
        st.metric("⭐ High Priority", len(product_ids))
    
    # Generate demo recommendations
    demo_recommendations = []
    priorities = ["high", "medium", "low"]
    
    for product in product_ids:
        demo_recommendations.extend([
            {
                "title": f"Optimize inventory levels for {product}",
                "description": f"Increase stock by 20% before peak season to meet demand",
                "priority": np.random.choice(priorities),
                "confidence_score": np.random.uniform(0.7, 0.95),
                "expected_impact": {"revenue_increase": np.random.randint(5000, 25000)},
                "compliance_status": "approved"
            },
            {
                "title": f"Adjust pricing strategy for {product}",
                "description": f"Apply 15% discount during weekends for maximum impact",
                "priority": np.random.choice(priorities),
                "confidence_score": np.random.uniform(0.6, 0.9),
                "expected_impact": {"revenue_increase": np.random.randint(3000, 15000)},
                "compliance_status": "approved"
            }
        ])
    
    # Display recommendations
    st.markdown('<h3 class="section-header">💡 Smart Recommendations</h3>', unsafe_allow_html=True)
    
    for i, rec in enumerate(demo_recommendations, 1):
        priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(rec["priority"], "🟡")
        
        with st.expander(f"{priority_color} Recommendation #{i}", expanded=True):
            st.write(f"**Title:** {rec['title']}")
            st.write(f"**Description:** {rec['description']}")
            st.write(f"**Priority:** {rec['priority'].title()}")
            st.write(f"**Confidence:** {rec['confidence_score']:.1%}")
            st.write(f"**Expected Revenue Impact:** ₹{rec['expected_impact']['revenue_increase']:,}")
            st.write(f"**Compliance:** {rec['compliance_status'].title()}")
            st.markdown("---")
    
    # Create visualization
    create_demo_recommendations_visualization(demo_recommendations)

def create_demo_recommendations_visualization(recommendations: List[Dict]):
    """Create demo visualizations for recommendations."""
    st.markdown('<h3 class="section-header">📊 Recommendations Analysis</h3>', unsafe_allow_html=True)
    
    # Priority distribution
    priorities = [rec['priority'] for rec in recommendations]
    priority_counts = pd.Series(priorities).value_counts()
    
    fig = px.pie(
        values=priority_counts.values,
        names=priority_counts.index,
        title="⭐ Priority Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

def show_scenario_analysis():
    """Display scenario analysis interface."""
    st.markdown('<h2 class="section-header">📊 Scenario Analysis</h2>', unsafe_allow_html=True)
    
    st.info("🔮 This demo shows how to test different business strategies and see predicted outcomes!")
    
    with st.form("scenario_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            product_id = st.text_input("🏷️ Product ID", value="MOBILE_001")
            current_inventory = st.number_input("📦 Current Inventory", min_value=0, value=1000)
            demand_forecast = st.number_input("📈 Demand Forecast", min_value=0, value=200)
        
        with col2:
            discount_range = st.slider("💰 Discount Range (%)", 0, 50, (10, 30))
            marketing_budget = st.number_input("📢 Marketing Budget (₹)", min_value=0, value=5000)
        
        submitted = st.form_submit_button("📊 Analyze Scenarios (Demo)", type="primary")
        
        if submitted:
            if product_id:
                display_demo_scenarios(product_id, current_inventory, demand_forecast, discount_range, marketing_budget)
            else:
                st.error("❌ Please enter a product ID")

def display_demo_scenarios(product_id: str, inventory: int, demand: int, discount_range: tuple, budget: int):
    """Display demo scenario analysis results."""
    st.markdown('<h3 class="section-header">🎭 Scenario Analysis Results</h3>', unsafe_allow_html=True)
    
    # Generate demo scenarios
    scenarios = [
        {
            "scenario_name": "Optimistic",
            "expected_revenue": 45000 + np.random.randint(-5000, 5000),
            "risk_score": 0.2,
            "inventory_turnover": 8.5,
            "confidence": 0.82
        },
        {
            "scenario_name": "Base Case",
            "expected_revenue": 35000 + np.random.randint(-3000, 3000),
            "risk_score": 0.4,
            "inventory_turnover": 6.2,
            "confidence": 0.91
        },
        {
            "scenario_name": "Pessimistic",
            "expected_revenue": 25000 + np.random.randint(-2000, 2000),
            "risk_score": 0.7,
            "inventory_turnover": 4.1,
            "confidence": 0.78
        }
    ]
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Scenarios Analyzed", len(scenarios))
    with col2:
        st.metric("🏆 Best Scenario", "Optimistic")
    with col3:
        st.metric("🌟 Seasonal Modeling", "✅")
    with col4:
        avg_revenue = np.mean([s["expected_revenue"] for s in scenarios])
        st.metric("💰 Avg Revenue", f"₹{avg_revenue:,.0f}")
    
    # Display scenarios table
    st.markdown('<h3 class="section-header">🎭 Scenario Comparison</h3>', unsafe_allow_html=True)
    
    scenario_df = pd.DataFrame([
        {
            "Scenario": s["scenario_name"],
            "Expected Revenue (₹)": f"₹{s['expected_revenue']:,.0f}",
            "Risk Score": f"{s['risk_score']:.2f}",
            "Inventory Turnover": f"{s['inventory_turnover']:.1f}",
            "Confidence": f"{s['confidence']:.1%}"
        }
        for s in scenarios
    ])
    
    st.dataframe(scenario_df, use_container_width=True)
    
    # Create visualizations
    create_demo_scenario_visualizations(scenarios)

def create_demo_scenario_visualizations(scenarios: List[Dict]):
    """Create demo visualizations for scenario analysis."""
    st.markdown('<h3 class="section-header">📈 Scenario Visualizations</h3>', unsafe_allow_html=True)
    
    scenario_names = [s["scenario_name"] for s in scenarios]
    revenues = [s["expected_revenue"] for s in scenarios]
    risk_scores = [s["risk_score"] for s in scenarios]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            x=scenario_names,
            y=revenues,
            title="💰 Expected Revenue by Scenario",
            labels={'x': 'Scenario', 'y': 'Expected Revenue (₹)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            x=scenario_names,
            y=risk_scores,
            title="⚠️ Risk Score by Scenario",
            labels={'x': 'Scenario', 'y': 'Risk Score'}
        )
        st.plotly_chart(fig, use_container_width=True)

def show_risk_assessment():
    """Display risk assessment interface."""
    st.markdown('<h2 class="section-header">⚠️ Risk Assessment</h2>', unsafe_allow_html=True)
    
    st.info("🛡️ This demo shows how to identify potential risks in inventory and demand patterns!")
    
    with st.form("risk_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            product_id = st.text_input("🏷️ Product ID", value="MOBILE_001")
            current_inventory = st.number_input("📦 Current Inventory", min_value=0, value=500)
        
        with col2:
            assessment_type = st.selectbox("🔍 Assessment Type", ["Both", "Overstock", "Understock"])
            include_seasonal = st.checkbox("🌟 Include Seasonal Adjustment", value=True)
        
        submitted = st.form_submit_button("⚠️ Assess Risk (Demo)", type="primary")
        
        if submitted:
            if product_id:
                display_demo_risk_results(product_id, current_inventory, assessment_type, include_seasonal)
            else:
                st.error("❌ Please enter a product ID")

def display_demo_risk_results(product_id: str, inventory: int, assessment_type: str, seasonal: bool):
    """Display demo risk assessment results."""
    st.markdown('<h3 class="section-header">🛡️ Risk Assessment Results</h3>', unsafe_allow_html=True)
    
    # Generate demo risk data
    overstock_risk = np.random.uniform(0.2, 0.8)
    understock_risk = np.random.uniform(0.1, 0.7)
    overall_risk = "medium" if (overstock_risk + understock_risk) / 2 < 0.6 else "high"
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏷️ Product ID", product_id)
    with col2:
        st.metric("📦 Current Inventory", inventory)
    with col3:
        st.metric("📊 Demand Volatility", f"{np.random.uniform(0.1, 0.5):.2f}")
    with col4:
        st.metric("🌟 Seasonal Adjusted", "✅" if seasonal else "❌")
    
    # Risk scores
    st.markdown('<h3 class="section-header">⚠️ Risk Analysis</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_color = "🔴" if overstock_risk > 0.7 else "🟡" if overstock_risk > 0.4 else "🟢"
        st.metric("📈 Overstock Risk", f"{overstock_risk:.1%}", delta=risk_color)
    
    with col2:
        risk_color = "🔴" if understock_risk > 0.7 else "🟡" if understock_risk > 0.4 else "🟢"
        st.metric("📉 Understock Risk", f"{understock_risk:.1%}", delta=risk_color)
    
    with col3:
        risk_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(overall_risk, "⚪")
        st.metric("🎯 Overall Risk", f"{risk_emoji} {overall_risk.title()}")
    
    # Risk visualization
    create_demo_risk_visualization(overstock_risk, understock_risk)
    
    # Demo recommendations
    st.markdown('<h3 class="section-header">💡 Risk Mitigation Recommendations</h3>', unsafe_allow_html=True)
    
    recommendations = [
        "Consider reducing inventory by 15% to minimize overstock risk",
        "Monitor demand patterns closely during upcoming festival season",
        "Implement dynamic pricing strategy to optimize inventory turnover",
        "Set up automated alerts for inventory levels below safety stock"
    ]
    
    for i, rec in enumerate(recommendations, 1):
        with st.container():
            st.write(f"💡 **Recommendation #{i}**")
            st.write(rec)
            st.markdown("---")

def create_demo_risk_visualization(overstock_risk: float, understock_risk: float):
    """Create demo risk visualization."""
    st.markdown('<h3 class="section-header">📊 Risk Visualization</h3>', unsafe_allow_html=True)
    
    # Risk gauge chart
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = overstock_risk * 100,
        domain = {'x': [0, 0.5], 'y': [0, 1]},
        title = {'text': "Overstock Risk (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = understock_risk * 100,
        domain = {'x': [0.5, 1], 'y': [0, 1]},
        title = {'text': "Understock Risk (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkgreen"},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()