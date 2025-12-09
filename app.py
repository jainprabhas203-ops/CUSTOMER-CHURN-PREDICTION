import streamlit as st
import pandas as pd
import numpy as np
import time
import logging
from io import StringIO

# Import our modules
from data_loader import load_data
from eda_engine import (
    plot_churn_distribution, 
    plot_numerical_features, 
    plot_categorical_features,
    correlation_heatmap,
    plot_churn_by_tenure,
    plot_monthly_charges_distribution,
    plot_contract_churn_heatmap,
    plot_churn_trend_over_time,
    plot_payment_method_analysis,
    plot_tenure_histogram,
    plot_churn_by_multiple_features,
    plot_numerical_distribution_comparison,
    plot_service_adoption_heatmap,
    plot_churn_probability_by_tenure,
    plot_customer_value_segments,
    plot_customer_journey,
    plot_cohort_analysis,
    plot_geographical_distribution,
    plot_predictive_analytics_dashboard
)

# Try to import performance_monitor, but handle if psutil is not available
try:
    from performance_monitor import monitor_resources
    PERFORMANCE_MONITOR_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITOR_AVAILABLE = False
    def monitor_resources():
        pass  # Dummy function if monitoring is not available

from ml_engine import train_models, predict_churn, get_feature_importance, load_or_train_models
from ml_pipeline_cache import list_cached_models, clear_cache, get_cache_size, model_exists
from recommendation_engine import generate_recommendations
from utils import find_similar_column, safe_get_column
from performance_monitor import monitor_resources, get_system_resources

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure the page
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced design
st.markdown("""
<style>
    /* Main background */
    .reportview-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
    }
    
    /* Sidebar text */
    .css-1d391kg .css-1rs6os {
        color: white;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #e9ecef;
        border-radius: 8px 8px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #495057;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        font-weight: bold;
        border-top: 3px solid #ff4b4b;
        color: #212529;
    }
    
    /* Expander styling */
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2c3e50;
    }
    
    /* Header styling */
    header[data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
    }
    
    /* Alert styling */
    .stAlert {
        background-color: #e1f5fe;
        border: 1px solid #01579b;
        border-radius: 10px;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #ffffff;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid #e0e0e0;
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
    }
    
    .metric-value {
        font-size: 2em;
        font-weight: 700;
        color: #1f77b4;
        margin: 5px 0;
    }
    
    .metric-label {
        font-size: 1em;
        color: #666;
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }
    
    /* Selectbox styling */
    .stSelectbox div[data-baseweb="select"] {
        border-radius: 8px;
    }
    
    /* Slider styling */
    .stSlider div[data-baseweb="slider"] {
        color: #667eea;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Main title styling */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* Subheaders */
    h2, h3 {
        color: #2c3e50;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Recommendation cards */
    .recommendation-card {
        background-color: #ffffff;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        padding: 20px;
        margin-bottom: 15px;
        border-left: 5px solid #3498db;
        transition: all 0.3s ease;
    }
    
    .recommendation-card:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.12);
    }
    
    .recommendation-card.high-priority {
        border-left-color: #e74c3c;
    }
    
    .recommendation-card.medium-priority {
        border-left-color: #f39c12;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def cached_load_data():
    """Cached version of data loading."""
    with st.spinner("Loading data..."):
        return load_data()

@st.cache_data(show_spinner=False)
def cached_train_models(data, _force_retrain=False):
    """Cached version of model training with optional cache loading."""
    if _force_retrain:
        with st.spinner("Retraining models from scratch..."):
            return load_or_train_models(data, force_retrain=True)
    else:
        with st.spinner("Loading models..."):
            return load_or_train_models(data, force_retrain=False)

# Title and description
st.title("📉 Customer Churn Prediction Dashboard")
st.markdown("---")

# Load data with error handling
try:
    data = cached_load_data()
    if data is None or data.empty:
        st.error("Failed to load data. Please check the data source.")
        st.stop()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    logger.error(f"Error in data loading: {e}")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("---")
    
    # Model selection
    model_choice = st.selectbox(
        "Select Model:",
        ("Random Forest", "XGBoost", "Ensemble"),
        index=0
    )
    
    # Data sampling option
    sample_size = st.slider(
        "Sample Size for Visualizations:",
        min_value=100,
        max_value=min(len(data), 10000),
        value=min(len(data), 5000),
        step=100
    )
    
    # Feature selection for analysis
    st.subheader("Analysis Features")
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    
    selected_num_feature = st.selectbox(
        "Numerical Feature for Analysis:",
        numerical_cols,
        index=min(2, len(numerical_cols)-1) if numerical_cols else 0
    )
    
    selected_cat_feature = st.selectbox(
        "Categorical Feature for Analysis:",
        categorical_cols,
        index=min(1, len(categorical_cols)-1) if categorical_cols else 0
    )
    
    st.markdown("---")
    
    # Cache management section
    st.subheader("🔧 Model Cache")
    
    # Display cache status
    cached_models = list_cached_models()
    if cached_models:
        st.success(f"✅ {len(cached_models)} model(s) cached")
        with st.expander("View Cache Details", expanded=False):
            for model in cached_models:
                meta = model.get('metadata', {})
                if meta:
                    st.write(f"**{model['name'].title()}**")
                    if 'training_date' in meta:
                        st.write(f"- Cached: {meta['training_date'][:19]}")
                    if 'accuracy' in meta:
                        st.write(f"- Accuracy: {meta['accuracy']:.3f}")
                    st.markdown("---")
        
        cache_size = get_cache_size()
        st.caption(f"Cache size: {cache_size:.2f} MB")
        
        # Force retrain button
        if st.button("🔄 Retrain Models", help="Clear cache and retrain all models"):
            clear_cache()
            st.cache_data.clear()
            st.success("Cache cleared! Models will retrain on next use.")
            st.rerun()
    else:
        st.info("No cached models. Will train on first use.")
    
    st.markdown("---")
    st.markdown("<div class='info-box'>💡 Tip: First run trains models, subsequent runs load instantly from cache!</div>", unsafe_allow_html=True)

# Main dashboard with professional tab organization
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Executive Summary", 
    "📊 Business Overview", 
    "📈 Advanced Analytics", 
    "🤖 Predictions & ML", 
    "💡 Insights & Actions"
])

# Tab 1: Executive Summary (NEW - Professional Dashboard)
with tab1:
    st.header("Executive Summary")
    st.markdown("**Critical metrics and insights at a glance**")
    st.markdown("---")
    
    # Top-level KPIs in prominent display
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Calculate key metrics
    total_customers = len(data)
    churn_rate = (data['Churn'] == 'Yes').mean() if 'Churn' in data.columns else 0
    retention_rate = 1 - churn_rate
    predicted_monthly_churn = int(total_customers * churn_rate)
    
    # Revenue metrics
    avg_monthly_charges = data['MonthlyCharges'].mean() if 'MonthlyCharges' in data.columns else 0
    revenue_at_risk = predicted_monthly_churn * avg_monthly_charges
    
    with col1:
        st.metric(
            label="📊 Total Customers",
            value=f"{total_customers:,}",
            delta=None
        )
    
    with col2:
        churn_delta = "High Risk" if churn_rate > 0.25 else "Normal"
        st.metric(
            label="⚠️ Churn Rate",
            value=f"{churn_rate:.1%}",
            delta=churn_delta,
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="✅ Retention Rate",
            value=f"{retention_rate:.1%}",
            delta="Target: 80%",
            delta_color="normal"
        )
    
    with col4:
        st.metric(
            label="📉 Monthly Churn",
            value=f"{predicted_monthly_churn:,}",
            delta=f"{predicted_monthly_churn/total_customers:.1%} of base"
        )
    
    with col5:
        st.metric(
            label="💰 Revenue at Risk",
            value=f"${revenue_at_risk:,.0f}",
            delta="Monthly"
        )
    
    st.markdown("---")
    
    # Critical Alerts Section
    st.subheader("🚨 Critical Alerts")
    col1, col2 = st.columns(2)
    
    with col1:
        if churn_rate > 0.30:
            st.error("⚠️ **High Churn Alert**: Churn rate exceeds 30% threshold!")
        elif churn_rate > 0.25:
            st.warning("⚠️ **Elevated Churn**: Churn rate elevated above 25%")
        else:
            st.success("✅ **Churn Status**: Within acceptable limits")
        
        # High risk segment identification
        if 'Contract' in data.columns:
            contract_churn = data.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean())
            highest_risk = contract_churn.idxmax()
            highest_risk_rate = contract_churn.max()
            st.info(f"🎯 **Top Risk Segment**: {highest_risk} customers ({highest_risk_rate:.1%} churn rate)")
    
    with col2:
        # Trend analysis (simplified)
        if 'tenure' in data.columns:
            new_customers = data[data['tenure'] <= 12]
            new_customer_churn = (new_customers['Churn'] == 'Yes').mean() if len(new_customers) > 0 else 0
            
            if new_customer_churn > 0.40:
                st.error(f"🆕 **New Customer Alert**: {new_customer_churn:.1%} churn in first year!")
            else:
                st.info(f"🆕 **New Customer Churn**: {new_customer_churn:.1%} in first 12 months")
        
        # Revenue insights
        st.success(f"💵 **Avg Monthly Revenue**: ${avg_monthly_charges:.2f} per customer")
    
    st.markdown("---")
    
    # Quick Insights Grid
    st.subheader("⚡ Quick Insights")
    
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        st.markdown("""
        <div class="recommendation-card">
            <h4>📊 Customer Base</h4>
            <ul>
                <li><strong>Total:</strong> {:,} customers</li>
                <li><strong>At Risk:</strong> {:,} customers</li>
                <li><strong>Stable:</strong> {:,} customers</li>
            </ul>
        </div>
        """.format(
            total_customers,
            predicted_monthly_churn,
            total_customers - predicted_monthly_churn
        ), unsafe_allow_html=True)
    
    with insight_col2:
        avg_tenure = data['tenure'].mean() if 'tenure' in data.columns else 0
        st.markdown("""
        <div class="recommendation-card">
            <h4>⏱️ Customer Lifecycle</h4>
            <ul>
                <li><strong>Avg Tenure:</strong> {:.1f} months</li>
                <li><strong>LTV Estimate:</strong> ${:,.0f}</li>
                <li><strong>Payback:</strong> ~{:.0f} months</li>
            </ul>
        </div>
        """.format(
            avg_tenure,
            avg_monthly_charges * avg_tenure,
            avg_tenure * 0.3
        ), unsafe_allow_html=True)
    
    with insight_col3:
        st.markdown("""
        <div class="recommendation-card high-priority">
            <h4>🎯 Top Actions</h4>
            <ul>
                <li>Focus on month-to-month contracts</li>
                <li>Improve first-year experience</li>
                <li>Target high-risk segments</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Visual Summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Churn Distribution")
        churn_fig = plot_churn_distribution(data)
        st.plotly_chart(churn_fig, use_container_width=True, key="exec_churn_dist")
    
    with col2:
        st.subheader("Revenue Analysis")
        if 'MonthlyCharges' in data.columns:
            revenue_fig = plot_monthly_charges_distribution(data)
            st.plotly_chart(revenue_fig, use_container_width=True, key="exec_revenue")

# Tab 2: Business Overview (Enhanced from Overview)
with tab2:
    st.header("Business Overview")
    st.markdown("**Comprehensive view of customer base and churn patterns**")
    
    # Key metrics in professional cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Customers</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{len(data):,}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    churn_rate = (data['Churn'] == 'Yes').mean() if 'Churn' in data.columns else 0
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Churn Rate</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{churn_rate:.1%}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    avg_tenure = data['tenure'].mean() if 'tenure' in data.columns else 0
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Avg. Tenure (months)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{avg_tenure:.1f}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    avg_monthly_charges = data['MonthlyCharges'].mean() if 'MonthlyCharges' in data.columns else 0
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Avg. Monthly Charges ($)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">${avg_monthly_charges:.2f}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Churn Distribution")
        churn_fig = plot_churn_distribution(data)
        st.plotly_chart(churn_fig, use_container_width=True, key="biz_churn_dist")
    
    with col2:
        st.subheader("Customer Tenure Distribution")
        tenure_hist_fig = plot_tenure_histogram(data)
        st.plotly_chart(tenure_hist_fig, use_container_width=True, key="biz_tenure_hist")

# Tab 3: Advanced Analytics (Enhanced from Analytics)
with tab3:
    st.header("Advanced Analytics")
    st.markdown("**Deep dive into customer behavior and churn patterns**")
    st.subheader("Churn Distribution")
    churn_fig = plot_churn_distribution(data)
    st.plotly_chart(churn_fig, use_container_width=True, key="churn_dist_chart")
    
    # Tenure histogram
    st.subheader("Customer Tenure Distribution")
    tenure_hist_fig = plot_tenure_histogram(data)
    st.plotly_chart(tenure_hist_fig, use_container_width=True, key="tenure_hist_chart")

# Tab 2: Analytics
with tab2:
    st.header("Advanced Analytics")
    
    # Train models for feature importance (if not already trained)
    with st.spinner("Preparing analytics..."):
        try:
            # Get trained models for feature importance
            result = cached_train_models(data)
            
            # Handle both old and new return value formats
            if isinstance(result, tuple) and len(result) >= 3:
                # New format: (pipelines, feature_columns, trained_models)
                pipelines, feature_columns, trained_models = result[:3]
                # Extract individual pipelines
                rf_pipeline, xgb_pipeline, ensemble_pipeline = pipelines
                rf_features, xgb_features, ensemble_features = feature_columns
                rf_model, xgb_model, ensemble_model = trained_models
            else:
                # Old format: just the pipelines
                rf_pipeline, xgb_pipeline, ensemble_pipeline = result
                rf_features = xgb_features = ensemble_features = None
                rf_model = xgb_model = ensemble_model = None
        except Exception as e:
            st.warning(f"Could not prepare analytics: {str(e)}")
            rf_model = xgb_model = ensemble_model = None
            rf_features = xgb_features = ensemble_features = None

    # Expanders for different chart categories
    with st.expander("🔍 Feature Analysis", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"Distribution of {selected_num_feature}")
            num_fig = plot_numerical_features(data, selected_num_feature)
            st.plotly_chart(num_fig, use_container_width=True, key=f"num_feature_{selected_num_feature}")
        
        with col2:
            st.subheader(f"Churn by {selected_cat_feature}")
            cat_fig = plot_categorical_features(data, selected_cat_feature)
            st.plotly_chart(cat_fig, use_container_width=True, key=f"cat_feature_{selected_cat_feature}")
    
    with st.expander("📊 Comparative Analysis", expanded=False):
        st.subheader("Churn Rate by Tenure")
        tenure_fig = plot_churn_by_tenure(data)
        st.plotly_chart(tenure_fig, use_container_width=True, key="churn_tenure_chart")
        
        st.subheader("Monthly Charges Distribution")
        mc_fig = plot_monthly_charges_distribution(data)
        st.plotly_chart(mc_fig, use_container_width=True, key="monthly_charges_chart")
        
        st.subheader("Churn Trend Over Time")
        trend_fig = plot_churn_trend_over_time(data)
        st.plotly_chart(trend_fig, use_container_width=True, key="churn_trend_chart")
    
    with st.expander("🔗 Relationship Analysis", expanded=False):
        st.subheader("Feature Correlation Heatmap")
        corr_fig = correlation_heatmap(data)
        st.plotly_chart(corr_fig, use_container_width=True, key="correlation_heatmap")
        
        st.subheader("Contract vs Internet Service Churn Heatmap")
        contract_fig = plot_contract_churn_heatmap(data)
        st.plotly_chart(contract_fig, use_container_width=True, key="contract_heatmap")
        
        st.subheader("Payment Method Analysis")
        payment_fig = plot_payment_method_analysis(data)
        st.plotly_chart(payment_fig, use_container_width=True, key="payment_method_chart")
    
    with st.expander("🎯 Advanced Insights", expanded=False):
        # Feature Importance
        st.subheader("Feature Importance")
        col1, col2, col3 = st.columns(3)
        with col1:
            model_option = st.radio("Select Model:", ["Random Forest", "XGBoost"], key="feature_imp_model")
        
        try:
            if model_option == "Random Forest" and rf_model is not None and rf_features is not None:
                feature_imp = get_feature_importance(rf_model, rf_features)
                if feature_imp is not None:
                    st.bar_chart(feature_imp.set_index('feature')['importance'][:15])
            elif model_option == "XGBoost" and xgb_model is not None and xgb_features is not None:
                feature_imp = get_feature_importance(xgb_model, xgb_features)
                if feature_imp is not None:
                    st.bar_chart(feature_imp.set_index('feature')['importance'][:15])
            else:
                st.info("Feature importance data not available for the selected model.")
        except Exception as e:
            st.warning(f"Could not display feature importance: {str(e)}")
        
        st.subheader("Multi-feature Churn Analysis")
        col1, col2 = st.columns(2)
        with col1:
            feature1 = st.selectbox("Select First Feature:", categorical_cols, index=0, key="multi_feature1")
        with col2:
            feature2 = st.selectbox("Select Second Feature:", categorical_cols, index=min(1, len(categorical_cols)-1), key="multi_feature2")
        
        multi_feature_fig = plot_churn_by_multiple_features(data, feature1, feature2)
        st.plotly_chart(multi_feature_fig, use_container_width=True, key="multi_feature_chart")
        
        st.subheader("Numerical Feature Distribution Comparison")
        dist_feature = st.selectbox("Select Numerical Feature:", numerical_cols, index=0, key="dist_feature")
        dist_comp_fig = plot_numerical_distribution_comparison(data, dist_feature)
        st.plotly_chart(dist_comp_fig, use_container_width=True, key="dist_comp_chart")
        
        st.subheader("Service Adoption Rates")
        service_fig = plot_service_adoption_heatmap(data)
        st.plotly_chart(service_fig, use_container_width=True, key="service_adoption_chart")
        
        st.subheader("Churn Probability Analysis")
        prob_fig = plot_churn_probability_by_tenure(data)
        st.plotly_chart(prob_fig, use_container_width=True, key="churn_probability_chart")
        
        st.subheader("Customer Value Segmentation")
        value_seg_fig = plot_customer_value_segments(data)
        st.plotly_chart(value_seg_fig, use_container_width=True, key="customer_value_chart")
    
    with st.expander("🧭 Customer Journey & Cohort Analysis", expanded=False):
        st.subheader("Customer Journey Visualization")
        journey_fig = plot_customer_journey(data)
        st.plotly_chart(journey_fig, use_container_width=True, key="journey_chart")
        
        st.subheader("Cohort Analysis")
        cohort_fig = plot_cohort_analysis(data)
        st.plotly_chart(cohort_fig, use_container_width=True, key="cohort_chart")
    
    with st.expander("🌍 Geographical & Predictive Analytics", expanded=False):
        st.subheader("Geographical Distribution")
        geo_fig = plot_geographical_distribution(data)
        st.plotly_chart(geo_fig, use_container_width=True, key="geo_chart")
        
        st.subheader("Predictive Analytics Dashboard")
        pred_fig = plot_predictive_analytics_dashboard(data)
        st.plotly_chart(pred_fig, use_container_width=True, key="pred_chart")

# Tab 3: Predictions
with tab3:
    st.header("Churn Predictions")
    
    # Train models with progress bar
    with st.spinner("Training models..."):
        progress_bar = st.progress(0)
        try:
            # Get trained models
            result = cached_train_models(data)
            
            # Handle both old and new return value formats
            if isinstance(result, tuple) and len(result) >= 3:
                # New format: (pipelines, feature_columns, trained_models)
                pipelines, feature_columns, trained_models = result[:3]
                # Extract individual pipelines
                rf_pipeline, xgb_pipeline, ensemble_pipeline = pipelines
                rf_features, xgb_features, ensemble_features = feature_columns
                rf_model, xgb_model, ensemble_model = trained_models
            else:
                # Old format: just the pipelines
                rf_pipeline, xgb_pipeline, ensemble_pipeline = result
                rf_features = xgb_features = ensemble_features = None
            
            progress_bar.progress(100)
        except Exception as e:
            st.error(f"Error training models: {str(e)}")
            logger.error(f"Error in model training: {e}")
            st.stop()
    
    # Select the appropriate model based on user choice
    if model_choice == "Random Forest":
        selected_pipeline = rf_pipeline
        model_name = "Random Forest"
        feature_columns = rf_features
        selected_model = rf_model if 'rf_model' in locals() else None
    elif model_choice == "XGBoost":
        selected_pipeline = xgb_pipeline
        model_name = "XGBoost"
        feature_columns = xgb_features
        selected_model = xgb_model if 'xgb_model' in locals() else None
    else:
        selected_pipeline = ensemble_pipeline
        model_name = "Ensemble"
        feature_columns = ensemble_features
        selected_model = ensemble_model if 'ensemble_model' in locals() else None
    
    st.success(f"{model_name} model selected!")
    
    # Show model performance metrics
    st.subheader("Model Performance")
    try:
        # Make predictions on the entire dataset for evaluation
        if feature_columns is not None:
            # Align features with the trained model
            X_eval = data[feature_columns].copy()
        else:
            # Use all features except target
            target_col = find_similar_column(['Churn'], data.columns)
            if target_col:
                X_eval = data.drop(columns=[target_col])
            else:
                X_eval = data
        
        # Get predictions
        predictions = predict_churn(selected_pipeline, X_eval, model_name, feature_columns)
        
        # Calculate metrics
        if 'Churn' in data.columns:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            y_true = (data['Churn'] == 'Yes').astype(int)
            y_pred = predictions
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.3f}")
            with col2:
                st.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.3f}")
            with col3:
                st.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.3f}")
            with col4:
                st.metric("F1-Score", f"{f1_score(y_true, y_pred, zero_division=0):.3f}")
                
        # Feature Importance
        if selected_model is not None and feature_columns is not None:
            st.subheader("Feature Importance")
            feature_imp = get_feature_importance(selected_model, feature_columns)
            if feature_imp is not None:
                st.bar_chart(feature_imp.set_index('feature'))
    except Exception as e:
        st.warning(f"Could not calculate performance metrics: {str(e)}")
        logger.warning(f"Performance metrics calculation failed: {e}")
    
    # Customer prediction section
    st.subheader("Predict Churn for Individual Customers")
    
    # Create input fields for prediction
    st.write("Enter customer details:")
    
    # Get numerical and categorical columns for input
    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = data.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target column from features
    target_col = find_similar_column(['Churn'], data.columns)
    if target_col:
        if target_col in num_cols:
            num_cols.remove(target_col)
        if target_col in cat_cols:
            cat_cols.remove(target_col)
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        user_inputs = {}
        
        # Numerical inputs
        with col1:
            for col in num_cols[:len(num_cols)//2]:
                user_inputs[col] = st.number_input(
                    col, 
                    value=float(data[col].median()) if not pd.isna(data[col].median()) else 0.0,
                    key=f"num_{col}"
                )
        
        with col2:
            for col in num_cols[len(num_cols)//2:]:
                user_inputs[col] = st.number_input(
                    col, 
                    value=float(data[col].median()) if not pd.isna(data[col].median()) else 0.0,
                    key=f"num2_{col}"
                )
        
        # Categorical inputs
        st.markdown("---")
        st.write("Categorical Features:")
        cat_col1, cat_col2 = st.columns(2)
        
        with cat_col1:
            for col in cat_cols[:len(cat_cols)//2]:
                unique_vals = data[col].dropna().unique().tolist()
                user_inputs[col] = st.selectbox(
                    col, 
                    unique_vals, 
                    index=0,
                    key=f"cat_{col}"
                )
        
        with cat_col2:
            for col in cat_cols[len(cat_cols)//2:]:
                unique_vals = data[col].dropna().unique().tolist()
                user_inputs[col] = st.selectbox(
                    col, 
                    unique_vals, 
                    index=0,
                    key=f"cat2_{col}"
                )
        
        submitted = st.form_submit_button("🔮 Predict Churn")
    
    if submitted:
        try:
            # Create DataFrame from user inputs
            input_df = pd.DataFrame([user_inputs])
            
            # Make prediction
            prediction = predict_churn(selected_pipeline, input_df, model_name, feature_columns)
            
            # Display result
            st.subheader("Prediction Result")
            if prediction[0] == 1:
                st.error("🚨 High Risk of Churn")
                st.warning("This customer is predicted to churn. Consider implementing retention strategies.")
            else:
                st.success("✅ Low Risk of Churn")
                st.info("This customer is predicted to stay. Continue providing good service.")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            logger.error(f"Prediction error: {e}")
    
    # Real-time monitoring section
    st.markdown("---")
    with st.expander("⏱️ Real-Time Model Monitoring", expanded=False):
        st.info("In a production environment, this would connect to live monitoring systems.")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", "0.852")
        with col2:
            st.metric("Precision", "0.818")
        with col3:
            st.metric("Recall", "0.794")
        with col4:
            st.metric("F1-Score", "0.806")
        
        # System resources
        st.subheader("System Resources")
        resources = get_system_resources()
        if resources:
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            with res_col1:
                st.metric("CPU Usage", f"{resources['cpu_percent']:.1f}%")
            with res_col2:
                st.metric("Memory Used", f"{resources['memory_percent']:.1f}%")
            with res_col3:
                st.metric("Available Memory", f"{resources['memory_available_gb']:.1f} GB")
            with res_col4:
                st.metric("Disk Usage", f"{resources['disk_percent']:.1f}%")

# Tab 4: Recommendations
with tab4:
    st.header("Personalized Recommendations")
    
    # Generate recommendations with a progress spinner
    with st.spinner("Generating recommendations..."):
        try:
            recommendations = generate_recommendations(data)
            if recommendations is None:
                st.warning("Could not generate recommendations at this time.")
            else:
                # Display recommendations with priority levels
                for i, rec in enumerate(recommendations[:10]):  # Show top 10
                    # Determine priority level based on content
                    priority_class = ""
                    if "🚨" in rec or "high-value" in rec.lower():
                        priority_class = "high-priority"
                    elif "⚠️" in rec or "risk" in rec.lower():
                        priority_class = "medium-priority"
                    
                    st.markdown(f"""
                    <div class="recommendation-card {priority_class}">
                        <h4>Recommendation #{i+1}</h4>
                        <p>{rec}</p>
                    </div>
                    """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
            logger.error(f"Recommendation generation error: {e}")

# Resource monitoring
monitor_resources()

# Footer
st.markdown("---")
st.caption("Customer Churn Prediction Dashboard | Powered by Streamlit & Machine Learning")