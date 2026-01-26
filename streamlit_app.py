"""
MarketPulse AI - Streamlit Web Interface

Beautiful, user-friendly web interface for MarketPulse AI system.
No JSON required - just clean forms and visual results!
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, date, timedelta
import time
from typing import Dict, List, Any
import numpy as np

# Page configuration
st.set_page_config(
    page_title="MarketPulse AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
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

# API Configuration
API_BASE_URL = "http://localhost:8001"

def check_api_connection():
    """Check if the MarketPulse AI API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def call_api(endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
    """Make API calls to MarketPulse AI backend."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "POST":
            response = requests.post(url, json=data, timeout=30)
        else:
            response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": f"Connection Error: {str(e)}"}

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 MarketPulse AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Retail Decision Support System</p>', unsafe_allow_html=True)
    
    # Check API connection
    if not check_api_connection():
        st.error("⚠️ MarketPulse AI API is not running. Please start the server with `python run.py`")
        st.info("💡 Make sure the API server is running on http://localhost:8001")
        return
    
    # Sidebar navigation
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["📊 Dashboard", "📈 Data Upload", "🔍 AI Insights", "💡 Recommendations", "📊 Scenario Analysis", "⚠️ Risk Assessment", "🛡️ Compliance Check"]
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
    elif page == "🛡️ Compliance Check":
        show_compliance_check()

def show_dashboard():
    """Display the main dashboard."""
    st.markdown('<h2 class="section-header">📊 System Dashboard</h2>', unsafe_allow_html=True)
    
    # Get system status
    status_response = call_api("/api/v1/system/status")
    
    if status_response.get("success"):
        status_data = status_response.get("data", {})
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🔧 Components", status_data.get("components_initialized", 0))
        
        with col2:
            st.metric("📊 API Version", status_data.get("api_version", "1.0.0"))
        
        with col3:
            st.metric("🟢 Status", "Healthy")
        
        with col4:
            st.metric("⏰ Uptime", "Running")
        
        # Component health
        st.markdown('<h3 class="section-header">🔧 Component Health</h3>', unsafe_allow_html=True)
        
        component_health = status_data.get("component_health", {})
        if component_health:
            health_df = pd.DataFrame([
                {"Component": comp.replace("_", " ").title(), "Status": status}
                for comp, status in component_health.items()
            ])
            
            # Create a color-coded status display
            for _, row in health_df.iterrows():
                status_color = "🟢" if "healthy" in row["Status"].lower() else "🟡"
                st.markdown(f"{status_color} **{row['Component']}**: {row['Status']}")
        
        # Available endpoints
        st.markdown('<h3 class="section-header">🔗 Available Features</h3>', unsafe_allow_html=True)
        
        endpoints = status_data.get("available_endpoints", {})
        for category, endpoint_list in endpoints.items():
            with st.expander(f"📋 {category.replace('_', ' ').title()}"):
                for endpoint in endpoint_list:
                    st.code(endpoint)
    
    else:
        st.error("Failed to fetch system status")

def show_data_upload():
    """Display the data upload interface."""
    st.markdown('<h2 class="section-header">📈 Upload Sales Data</h2>', unsafe_allow_html=True)
    
    st.info("💡 Upload your sales data to start getting AI insights and recommendations!")
    
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
        
        # Additional fields
        st.markdown("#### 📋 Additional Information (Optional)")
        col3, col4 = st.columns(2)
        
        with col3:
            seasonal_event = st.text_input("🎉 Seasonal Event", placeholder="e.g., Diwali, Christmas")
            category = st.selectbox("📂 Category", ["Electronics", "Clothing", "Food", "Books", "Other"])
        
        with col4:
            demand_level = st.selectbox("📈 Demand Level", ["Low", "Medium", "High"])
            competition = st.selectbox("🏪 Competition", ["Low", "Medium", "High"])
        
        submitted = st.form_submit_button("📤 Upload Data", type="primary")
        
        if submitted:
            if product_id and quantity_sold >= 0 and price > 0:
                # Prepare data
                sales_data = {
                    "data": [{
                        "product_id": product_id,
                        "date": sale_date.strftime("%Y-%m-%d"),
                        "quantity_sold": quantity_sold,
                        "revenue": revenue if revenue > 0 else quantity_sold * price,
                        "inventory_level": inventory_level,
                        "price": price,
                        "seasonal_event": seasonal_event if seasonal_event else None,
                        "market_conditions": {
                            "demand_level": demand_level.lower(),
                            "competition": competition.lower()
                        }
                    }],
                    "validate_data": True,
                    "store_patterns": True
                }
                
                # Upload data
                response = call_api("/api/v1/data/ingest", "POST", sales_data)
                
                if response.get("success"):
                    st.success("✅ Data uploaded successfully!")
                    st.balloons()
                else:
                    st.error(f"❌ Upload failed: {response.get('error', 'Unknown error')}")
            else:
                st.error("❌ Please fill in all required fields (Product ID, Quantity, Price)")

def show_csv_upload():
    """Show CSV upload interface."""
    st.markdown("### 📁 Upload CSV File")
    
    # Show expected format
    st.info("📋 Your CSV should have these columns: product_id, date, quantity_sold, revenue, inventory_level, price")
    
    # Sample CSV format
    sample_df = pd.DataFrame({
        "product_id": ["PROD001", "PROD002"],
        "date": ["2024-01-15", "2024-01-15"],
        "quantity_sold": [150, 75],
        "revenue": [15000.0, 7500.0],
        "inventory_level": [500, 200],
        "price": [100.0, 100.0]
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
            if st.button("📤 Upload to MarketPulse AI", type="primary"):
                # Convert to API format
                sales_data = {
                    "data": df.to_dict('records'),
                    "validate_data": True,
                    "store_patterns": True
                }
                
                # Upload data
                with st.spinner("Uploading data..."):
                    response = call_api("/api/v1/data/ingest", "POST", sales_data)
                
                if response.get("success"):
                    st.success(f"✅ Successfully uploaded {len(df)} records!")
                    st.balloons()
                else:
                    st.error(f"❌ Upload failed: {response.get('error', 'Unknown error')}")
        
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")

def show_sample_data_upload():
    """Show sample data upload."""
    st.markdown("### 📊 Use Sample Data")
    
    st.info("💡 Try MarketPulse AI with sample retail data to see how it works!")
    
    # Generate sample data
    sample_data = generate_sample_data()
    
    # Show sample data preview
    st.markdown("#### 👀 Sample Data Preview")
    sample_df = pd.DataFrame(sample_data["data"])
    st.dataframe(sample_df)
    
    if st.button("📤 Upload Sample Data", type="primary"):
        with st.spinner("Uploading sample data..."):
            response = call_api("/api/v1/data/ingest", "POST", sample_data)
        
        if response.get("success"):
            st.success("✅ Sample data uploaded successfully!")
            st.info("🎯 Now you can try the AI Insights and Recommendations features!")
            st.balloons()
        else:
            st.error(f"❌ Upload failed: {response.get('error', 'Unknown error')}")

def generate_sample_data():
    """Generate sample sales data."""
    products = ["MOBILE_001", "LAPTOP_001", "TABLET_001", "HEADPHONES_001", "CAMERA_001"]
    
    sample_data = []
    base_date = datetime.now() - timedelta(days=30)
    
    for i in range(50):
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
            "price": price,
            "seasonal_event": np.random.choice([None, "Diwali", "Christmas", "New Year"], p=[0.7, 0.1, 0.1, 0.1]),
            "market_conditions": {
                "demand_level": np.random.choice(["low", "medium", "high"]),
                "competition": np.random.choice(["low", "medium", "high"])
            }
        })
    
    return {
        "data": sample_data,
        "validate_data": True,
        "store_patterns": True
    }

def show_insights():
    """Display AI insights interface."""
    st.markdown('<h2 class="section-header">🔍 AI Insights</h2>', unsafe_allow_html=True)
    
    st.info("🧠 Get intelligent analysis of your sales data and discover hidden patterns!")
    
    with st.form("insights_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            product_ids = st.text_area(
                "🏷️ Product IDs (one per line)",
                placeholder="PROD001\nPROD002\nPROD003",
                help="Enter product IDs you want to analyze, one per line"
            )
            include_seasonal = st.checkbox("🌟 Include Seasonal Analysis", value=True)
        
        with col2:
            confidence_threshold = st.slider("🎯 Confidence Threshold", 0.0, 1.0, 0.7, 0.1)
            max_insights = st.number_input("📊 Maximum Insights", min_value=1, max_value=50, value=10)
        
        submitted = st.form_submit_button("🔍 Generate Insights", type="primary")
        
        if submitted:
            if product_ids.strip():
                # Parse product IDs
                product_list = [pid.strip() for pid in product_ids.split('\n') if pid.strip()]
                
                # Prepare request
                insights_request = {
                    "product_ids": product_list,
                    "include_seasonal": include_seasonal,
                    "confidence_threshold": confidence_threshold,
                    "max_insights": max_insights
                }
                
                # Get insights
                with st.spinner("🧠 Generating AI insights..."):
                    response = call_api("/api/v1/insights/generate", "POST", insights_request)
                
                if response.get("success"):
                    display_insights_results(response.get("data", {}))
                else:
                    st.error(f"❌ Failed to generate insights: {response.get('error', 'Unknown error')}")
            else:
                st.error("❌ Please enter at least one product ID")

def display_insights_results(data: Dict):
    """Display insights results with visualizations."""
    st.markdown('<h3 class="section-header">📊 Analysis Results</h3>', unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔍 Insights Generated", data.get("insights_generated", 0))
    with col2:
        st.metric("📊 Patterns Analyzed", data.get("total_patterns_analyzed", 0))
    with col3:
        st.metric("🎯 Confidence Threshold", f"{data.get('confidence_threshold', 0):.1%}")
    with col4:
        st.metric("🌟 Seasonal Analysis", "✅" if data.get("seasonal_analysis_included") else "❌")
    
    # Display insights
    insights = data.get("insights", [])
    if insights:
        st.markdown('<h3 class="section-header">💡 Key Insights</h3>', unsafe_allow_html=True)
        
        for i, insight in enumerate(insights, 1):
            with st.container():
                st.markdown(f"""
                <div class="insight-card">
                    <h4>🔍 Insight #{i}</h4>
                    <p><strong>Finding:</strong> {insight.get('insight_text', 'N/A')}</p>
                    <p><strong>Confidence:</strong> {insight.get('confidence_level', 'N/A')}</p>
                    <p><strong>Key Factors:</strong> {', '.join(insight.get('key_factors', []))}</p>
                    <p><strong>Evidence:</strong> {insight.get('supporting_evidence', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Create visualization
        create_insights_visualization(insights)
    else:
        st.warning("⚠️ No insights generated. Try uploading more data or lowering the confidence threshold.")

def create_insights_visualization(insights: List[Dict]):
    """Create visualizations for insights."""
    st.markdown('<h3 class="section-header">📈 Insights Visualization</h3>', unsafe_allow_html=True)
    
    # Confidence distribution
    confidences = [float(insight.get('confidence_level', 0)) for insight in insights if insight.get('confidence_level')]
    
    if confidences:
        fig = px.histogram(
            x=confidences,
            nbins=10,
            title="📊 Confidence Level Distribution",
            labels={'x': 'Confidence Level', 'y': 'Number of Insights'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Key factors analysis
    all_factors = []
    for insight in insights:
        factors = insight.get('key_factors', [])
        all_factors.extend(factors)
    
    if all_factors:
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
    
    st.info("🎯 Get AI-powered business recommendations to optimize your retail operations!")
    
    with st.form("recommendations_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            product_ids = st.text_area(
                "🏷️ Product IDs (one per line)",
                placeholder="PROD001\nPROD002",
                help="Enter product IDs for recommendations"
            )
            priority_filter = st.selectbox("⭐ Priority Filter", ["All", "High", "Medium", "Low"])
            include_compliance = st.checkbox("🛡️ Include Compliance Check", value=True)
        
        with col2:
            target_margin = st.slider("🎯 Target Margin (%)", 0, 50, 25) / 100
            inventory_turnover = st.number_input("🔄 Inventory Turnover Target", min_value=1, max_value=24, value=12)
            max_recommendations = st.number_input("📊 Max Recommendations", min_value=1, max_value=50, value=20)
        
        # Business context
        st.markdown("#### 🏪 Business Context")
        col3, col4 = st.columns(2)
        
        with col3:
            seasonal_events = st.multiselect(
                "🎉 Upcoming Events",
                ["Diwali", "Christmas", "New Year", "Holi", "Eid", "Independence Day", "Republic Day"]
            )
        
        with col4:
            focus_area = st.selectbox(
                "🎯 Focus Area",
                ["Revenue Growth", "Cost Reduction", "Inventory Optimization", "Customer Satisfaction"]
            )
        
        submitted = st.form_submit_button("💡 Generate Recommendations", type="primary")
        
        if submitted:
            if product_ids.strip():
                # Parse product IDs
                product_list = [pid.strip() for pid in product_ids.split('\n') if pid.strip()]
                
                # Prepare request
                rec_request = {
                    "product_ids": product_list,
                    "business_context": {
                        "target_margin": target_margin,
                        "inventory_turnover_target": inventory_turnover,
                        "seasonal_events": seasonal_events,
                        "focus_area": focus_area
                    },
                    "priority_filter": priority_filter.lower() if priority_filter != "All" else None,
                    "include_compliance_check": include_compliance,
                    "max_recommendations": max_recommendations
                }
                
                # Get recommendations
                with st.spinner("💡 Generating smart recommendations..."):
                    response = call_api("/api/v1/recommendations/generate", "POST", rec_request)
                
                if response.get("success"):
                    display_recommendations_results(response.get("data", {}))
                else:
                    st.error(f"❌ Failed to generate recommendations: {response.get('error', 'Unknown error')}")
            else:
                st.error("❌ Please enter at least one product ID")

def display_recommendations_results(data: Dict):
    """Display recommendations results."""
    st.markdown('<h3 class="section-header">🎯 Recommendation Results</h3>', unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💡 Recommendations", len(data.get("recommendations", [])))
    with col2:
        st.metric("📊 Products Analyzed", data.get("total_products", 0))
    with col3:
        st.metric("🛡️ Compliance Checked", "✅" if data.get("compliance_validated") else "❌")
    with col4:
        priority_dist = data.get("priority_distribution", {})
        high_priority = priority_dist.get("high", 0)
        st.metric("⭐ High Priority", high_priority)
    
    # Display recommendations
    recommendations = data.get("recommendations", [])
    if recommendations:
        st.markdown('<h3 class="section-header">💡 Smart Recommendations</h3>', unsafe_allow_html=True)
        
        for i, rec in enumerate(recommendations, 1):
            priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(rec.get("priority", "medium"), "🟡")
            
            with st.container():
                st.markdown(f"""
                <div class="recommendation-card">
                    <h4>{priority_color} Recommendation #{i}</h4>
                    <p><strong>Title:</strong> {rec.get('title', 'N/A')}</p>
                    <p><strong>Description:</strong> {rec.get('description', 'N/A')}</p>
                    <p><strong>Priority:</strong> {rec.get('priority', 'N/A').title()}</p>
                    <p><strong>Confidence:</strong> {rec.get('confidence_score', 'N/A')}</p>
                    <p><strong>Expected Impact:</strong> {rec.get('expected_impact', {})}</p>
                    <p><strong>Compliance:</strong> {rec.get('compliance_status', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Create visualization
        create_recommendations_visualization(recommendations)
    else:
        st.warning("⚠️ No recommendations generated. Try with different parameters or upload more data.")

def create_recommendations_visualization(recommendations: List[Dict]):
    """Create visualizations for recommendations."""
    st.markdown('<h3 class="section-header">📊 Recommendations Analysis</h3>', unsafe_allow_html=True)
    
    # Priority distribution
    priorities = [rec.get('priority', 'medium') for rec in recommendations]
    priority_counts = pd.Series(priorities).value_counts()
    
    fig = px.pie(
        values=priority_counts.values,
        names=priority_counts.index,
        title="⭐ Priority Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Confidence scores
    confidences = [float(rec.get('confidence_score', 0)) for rec in recommendations if rec.get('confidence_score')]
    
    if confidences:
        fig = px.box(
            y=confidences,
            title="🎯 Confidence Score Distribution"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def show_scenario_analysis():
    """Display scenario analysis interface."""
    st.markdown('<h2 class="section-header">📊 Scenario Analysis</h2>', unsafe_allow_html=True)
    
    st.info("🔮 Test different business strategies and see predicted outcomes!")
    
    with st.form("scenario_form"):
        st.markdown("#### 📋 Base Scenario Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            product_id = st.text_input("🏷️ Product ID", placeholder="PROD001")
            current_inventory = st.number_input("📦 Current Inventory", min_value=0, value=1000)
            demand_forecast = st.number_input("📈 Demand Forecast", min_value=0, value=200)
        
        with col2:
            discount_min = st.slider("💰 Min Discount (%)", 0, 50, 10) / 100
            discount_max = st.slider("💰 Max Discount (%)", 0, 50, 30) / 100
            marketing_budget = st.number_input("📢 Marketing Budget (₹)", min_value=0, value=5000)
        
        # Scenario types
        st.markdown("#### 🎭 Scenario Types")
        scenario_types = st.multiselect(
            "Select scenarios to analyze:",
            ["optimistic", "pessimistic", "base", "seasonal_peak", "off_season", "competitive"],
            default=["optimistic", "pessimistic", "base"]
        )
        
        # Additional options
        col3, col4 = st.columns(2)
        with col3:
            include_seasonal = st.checkbox("🌟 Include Seasonal Modeling", value=True)
        with col4:
            max_scenarios = st.number_input("📊 Max Scenarios", min_value=1, max_value=20, value=5)
        
        submitted = st.form_submit_button("📊 Analyze Scenarios", type="primary")
        
        if submitted:
            if product_id and scenario_types:
                # Prepare request
                scenario_request = {
                    "base_parameters": {
                        "product_id": product_id,
                        "current_inventory": current_inventory,
                        "demand_forecast": demand_forecast,
                        "discount_range": [discount_min, discount_max],
                        "marketing_budget": marketing_budget
                    },
                    "scenario_types": scenario_types,
                    "include_seasonal": include_seasonal,
                    "max_scenarios": max_scenarios
                }
                
                # Analyze scenarios
                with st.spinner("🔮 Analyzing scenarios..."):
                    response = call_api("/api/v1/scenarios/analyze", "POST", scenario_request)
                
                if response.get("success"):
                    display_scenario_results(response.get("data", {}))
                else:
                    st.error(f"❌ Scenario analysis failed: {response.get('error', 'Unknown error')}")
            else:
                st.error("❌ Please enter product ID and select at least one scenario type")

def display_scenario_results(data: Dict):
    """Display scenario analysis results."""
    st.markdown('<h3 class="section-header">🎭 Scenario Analysis Results</h3>', unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Scenarios Analyzed", data.get("total_scenarios", 0))
    with col2:
        st.metric("🏆 Best Scenario", data.get("best_scenario", "N/A"))
    with col3:
        st.metric("🌟 Seasonal Modeling", "✅" if data.get("seasonal_modeling_included") else "❌")
    with col4:
        scenarios = data.get("scenarios", [])
        avg_revenue = np.mean([s.get("expected_revenue", 0) for s in scenarios]) if scenarios else 0
        st.metric("💰 Avg Revenue", f"₹{avg_revenue:,.0f}")
    
    # Display scenarios
    scenarios = data.get("scenarios", [])
    if scenarios:
        st.markdown('<h3 class="section-header">🎭 Scenario Comparison</h3>', unsafe_allow_html=True)
        
        # Create comparison table
        scenario_df = pd.DataFrame([
            {
                "Scenario": s.get("scenario_name", "Unknown"),
                "Expected Revenue (₹)": f"₹{s.get('expected_revenue', 0):,.0f}",
                "Risk Score": f"{s.get('risk_score', 0):.2f}",
                "Inventory Turnover": f"{s.get('inventory_turnover', 0):.1f}",
                "Confidence": f"{s.get('confidence', 0):.1%}"
            }
            for s in scenarios
        ])
        
        st.dataframe(scenario_df, use_container_width=True)
        
        # Create visualizations
        create_scenario_visualizations(scenarios)
        
        # Show detailed recommendations for each scenario
        st.markdown('<h3 class="section-header">💡 Scenario Recommendations</h3>', unsafe_allow_html=True)
        
        for scenario in scenarios:
            with st.expander(f"🎭 {scenario.get('scenario_name', 'Unknown').title()} Scenario"):
                recommendations = scenario.get('recommendations', [])
                if recommendations:
                    for rec in recommendations:
                        st.markdown(f"• {rec}")
                else:
                    st.info("No specific recommendations for this scenario")
    else:
        st.warning("⚠️ No scenarios generated. Please check your parameters and try again.")

def create_scenario_visualizations(scenarios: List[Dict]):
    """Create visualizations for scenario analysis."""
    st.markdown('<h3 class="section-header">📈 Scenario Visualizations</h3>', unsafe_allow_html=True)
    
    # Revenue comparison
    scenario_names = [s.get("scenario_name", "Unknown") for s in scenarios]
    revenues = [s.get("expected_revenue", 0) for s in scenarios]
    risk_scores = [s.get("risk_score", 0) for s in scenarios]
    
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
    
    # Risk vs Revenue scatter plot
    fig = px.scatter(
        x=risk_scores,
        y=revenues,
        text=scenario_names,
        title="🎯 Risk vs Revenue Analysis",
        labels={'x': 'Risk Score', 'y': 'Expected Revenue (₹)'}
    )
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

def show_risk_assessment():
    """Display risk assessment interface."""
    st.markdown('<h2 class="section-header">⚠️ Risk Assessment</h2>', unsafe_allow_html=True)
    
    st.info("🛡️ Identify potential risks in your inventory and demand patterns!")
    
    with st.form("risk_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            product_id = st.text_input("🏷️ Product ID", placeholder="PROD001")
            current_inventory = st.number_input("📦 Current Inventory", min_value=0, value=500)
            assessment_type = st.selectbox("🔍 Assessment Type", ["Both", "Overstock", "Understock"])
        
        with col2:
            include_seasonal = st.checkbox("🌟 Include Seasonal Adjustment", value=True)
            upcoming_events = st.multiselect(
                "🎉 Upcoming Events",
                ["Diwali", "Christmas", "New Year", "Holi", "Eid", "Summer Sale", "Winter Sale"]
            )
        
        submitted = st.form_submit_button("⚠️ Assess Risk", type="primary")
        
        if submitted:
            if product_id:
                # Prepare request
                risk_request = {
                    "product_id": product_id,
                    "current_inventory": current_inventory,
                    "assessment_type": assessment_type.lower(),
                    "include_seasonal_adjustment": include_seasonal,
                    "upcoming_events": upcoming_events
                }
                
                # Assess risk
                with st.spinner("🛡️ Assessing risks..."):
                    response = call_api("/api/v1/risk/assess", "POST", risk_request)
                
                if response.get("success"):
                    display_risk_results(response.get("data", {}))
                else:
                    st.error(f"❌ Risk assessment failed: {response.get('error', 'Unknown error')}")
            else:
                st.error("❌ Please enter a product ID")

def display_risk_results(data: Dict):
    """Display risk assessment results."""
    st.markdown('<h3 class="section-header">🛡️ Risk Assessment Results</h3>', unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏷️ Product ID", data.get("product_id", "N/A"))
    with col2:
        st.metric("📦 Current Inventory", data.get("current_inventory", 0))
    with col3:
        st.metric("📊 Demand Volatility", f"{data.get('demand_volatility', 0):.2f}")
    with col4:
        st.metric("🌟 Seasonal Adjusted", "✅" if data.get("seasonal_adjustments_applied") else "❌")
    
    # Risk assessments
    risk_assessments = data.get("risk_assessments", {})
    if risk_assessments:
        st.markdown('<h3 class="section-header">⚠️ Risk Analysis</h3>', unsafe_allow_html=True)
        
        # Risk scores
        overstock_risk = risk_assessments.get("overstock_risk", 0)
        understock_risk = risk_assessments.get("understock_risk", 0)
        overall_risk = risk_assessments.get("overall_risk", "unknown")
        
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
        create_risk_visualization(risk_assessments)
        
        # Recommendations
        recommendations = data.get("recommendations", [])
        if recommendations:
            st.markdown('<h3 class="section-header">💡 Risk Mitigation Recommendations</h3>', unsafe_allow_html=True)
            
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"""
                <div class="recommendation-card">
                    <h4>💡 Recommendation #{i}</h4>
                    <p>{rec}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No risk assessment data available.")

def create_risk_visualization(risk_assessments: Dict):
    """Create risk visualization."""
    st.markdown('<h3 class="section-header">📊 Risk Visualization</h3>', unsafe_allow_html=True)
    
    # Risk gauge chart
    overstock_risk = risk_assessments.get("overstock_risk", 0)
    understock_risk = risk_assessments.get("understock_risk", 0)
    
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode = "gauge+number+delta",
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
        mode = "gauge+number+delta",
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

def show_compliance_check():
    """Display compliance check interface."""
    st.markdown('<h2 class="section-header">🛡️ Compliance Check</h2>', unsafe_allow_html=True)
    
    st.info("⚖️ Validate your pricing strategies against MRP regulations!")
    
    with st.form("compliance_form"):
        st.markdown("#### 📋 Recommendation to Validate")
        
        col1, col2 = st.columns(2)
        
        with col1:
            product_id = st.text_input("🏷️ Product ID", placeholder="PROD001")
            original_price = st.number_input("💰 Original Price (₹)", min_value=0.0, value=100.0, format="%.2f")
            discount_percentage = st.slider("💸 Discount Percentage", 0, 50, 15) / 100
        
        with col2:
            new_price = original_price * (1 - discount_percentage)
            st.number_input("💵 New Price (₹)", value=new_price, format="%.2f", disabled=True)
            strategy = st.selectbox("📈 Strategy Type", ["seasonal_discount", "clearance_sale", "bulk_discount", "loyalty_discount"])
            product_category = st.selectbox("📂 Product Category", ["Electronics", "Clothing", "Food", "Books", "Other"])
        
        include_explanations = st.checkbox("📝 Include Explanations", value=True)
        
        submitted = st.form_submit_button("🛡️ Check Compliance", type="primary")
        
        if submitted:
            if product_id and original_price > 0:
                # Prepare request
                compliance_request = {
                    "recommendation": {
                        "product_id": product_id,
                        "discount_percentage": discount_percentage,
                        "new_price": new_price,
                        "original_price": original_price,
                        "strategy": strategy
                    },
                    "product_category": product_category.lower(),
                    "include_explanations": include_explanations
                }
                
                # Check compliance
                with st.spinner("🛡️ Checking compliance..."):
                    response = call_api("/api/v1/compliance/validate", "POST", compliance_request)
                
                if response.get("success"):
                    display_compliance_results(response.get("data", {}))
                else:
                    st.error(f"❌ Compliance check failed: {response.get('error', 'Unknown error')}")
            else:
                st.error("❌ Please enter product ID and valid price")

def display_compliance_results(data: Dict):
    """Display compliance check results."""
    st.markdown('<h3 class="section-header">⚖️ Compliance Results</h3>', unsafe_allow_html=True)
    
    compliance_result = data.get("compliance_result", {})
    is_compliant = compliance_result.get("is_compliant", False)
    confidence = compliance_result.get("confidence", 0)
    violations = compliance_result.get("violations", [])
    
    # Compliance status
    if is_compliant:
        st.success(f"✅ **COMPLIANT** - Confidence: {confidence:.1%}")
    else:
        st.error(f"❌ **NON-COMPLIANT** - Confidence: {confidence:.1%}")
    
    # Violations
    if violations:
        st.markdown('<h4 class="section-header">⚠️ Violations Found</h4>', unsafe_allow_html=True)
        for violation in violations:
            st.error(f"❌ {violation}")
    
    # Regulatory constraints
    constraints = data.get("regulatory_constraints", {})
    if constraints:
        st.markdown('<h4 class="section-header">📋 Regulatory Constraints</h4>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_discount = constraints.get("max_discount", 0)
            st.metric("💸 Max Discount Allowed", f"{max_discount:.1%}")
        
        with col2:
            mrp_compliance = constraints.get("mrp_compliance", "unknown")
            st.metric("⚖️ MRP Compliance", mrp_compliance.title())
    
    # Explanations
    explanations = data.get("constraint_explanations", {})
    if explanations:
        st.markdown('<h4 class="section-header">📝 Explanations</h4>', unsafe_allow_html=True)
        
        for key, explanation in explanations.items():
            st.info(f"**{key.replace('_', ' ').title()}:** {explanation}")
    
    # Original recommendation
    recommendation = data.get("recommendation", {})
    if recommendation:
        st.markdown('<h4 class="section-header">📊 Validated Recommendation</h4>', unsafe_allow_html=True)
        
        rec_df = pd.DataFrame([{
            "Field": key.replace('_', ' ').title(),
            "Value": value
        } for key, value in recommendation.items()])
        
        st.dataframe(rec_df, use_container_width=True)

if __name__ == "__main__":
    main()