import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_churn_distribution(df):
    """
    Returns a Plotly Donut chart for Churn distribution.
    """
    try:
        churn_counts = df['Churn'].value_counts()
        fig = px.pie(
            values=churn_counts.values, 
            names=churn_counts.index, 
            hole=0.5, 
            title='Overall Churn Distribution',
            color_discrete_sequence=['#EF553B', '#636EFA'] # Red for Churn (usually), Blue for No
        )
        fig.update_layout(title_x=0.5)
        return fig
    except Exception as e:
        logger.error(f"Error in plot_churn_distribution: {e}")
        fig = px.pie(title='Churn Distribution (Data Not Available)')
        return fig

def plot_numerical_features(df, feature):
    """
    Returns a Box Plot comparing a numerical feature against Churn.
    """
    try:
        # Check if feature exists in the dataset
        if feature not in df.columns:
            # Try to find a similar column name
            similar_cols = [col for col in df.columns if feature.lower() in col.lower()]
            if similar_cols:
                feature = similar_cols[0]
            else:
                # Return a simple bar chart if feature not found
                fig = px.bar(title=f'{feature} Distribution (Feature Not Found)')
                return fig
        
        # Check if the feature is numerical
        if not pd.api.types.is_numeric_dtype(df[feature]):
            # If not numerical, return a bar chart
            value_counts = df[feature].value_counts().reset_index()
            value_counts.columns = [feature, 'count']
            fig = px.bar(value_counts, x=feature, y='count', title=f'{feature} Distribution')
            return fig
        
        # Sample data if too large for performance
        if len(df) > 10000:
            df_sample = df.sample(n=10000, random_state=42)
        else:
            df_sample = df
        
        fig = px.box(
            df_sample, 
            x='Churn', 
            y=feature, 
            color='Churn',
            title=f'{feature} Distribution by Churn Status',
            color_discrete_map={'Yes': '#EF553B', 'No': '#636EFA'}
        )
        return fig
    except Exception as e:
        logger.error(f"Error in plot_numerical_features: {e}")
        fig = px.bar(title=f'{feature} Distribution (Error Occurred)')
        return fig

def plot_categorical_features(df, feature):
    """
    Returns a Stacked Bar chart for a categorical feature against Churn.
    """
    try:
        # Check if feature exists in the dataset
        if feature not in df.columns:
            # Try to find a similar column name
            similar_cols = [col for col in df.columns if feature.lower() in col.lower()]
            if similar_cols:
                feature = similar_cols[0]
            else:
                # Return a simple bar chart if feature not found
                fig = px.bar(title=f'{feature} Distribution (Feature Not Found)')
                return fig
        
        # Check if the feature is categorical
        if pd.api.types.is_numeric_dtype(df[feature]):
            # If numerical, convert to categorical
            df_copy = df.copy()
            df_copy[feature] = df_copy[feature].astype(str)
        else:
            df_copy = df
        
        # Sample data if too large for performance
        if len(df_copy) > 10000:
            df_sample = df_copy.sample(n=10000, random_state=42)
        else:
            df_sample = df_copy
        
        # Group by feature and Churn
        grouped = df_sample.groupby([feature, 'Churn']).size().reset_index(name='Count')
        
        fig = px.bar(
            grouped, 
            x=feature, 
            y='Count', 
            color='Churn', 
            title=f'Churn Rate by {feature}',
            barmode='stack',
            color_discrete_map={'Yes': '#EF553B', 'No': '#636EFA'}
        )
        return fig
    except Exception as e:
        logger.error(f"Error in plot_categorical_features: {e}")
        fig = px.bar(title=f'{feature} Distribution (Error Occurred)')
        return fig

def correlation_heatmap(df):
    """
    Returns a Heatmap of correlations for numerical features.
    """
    try:
        # Select only numeric cols
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        if numeric_df.empty:
            # Return a simple figure if no numeric columns found
            fig = px.scatter(title='No Numeric Columns Found for Correlation Analysis')
            return fig
        
        # Sample data if too large for performance
        if len(numeric_df) > 5000:
            numeric_df = numeric_df.sample(n=5000, random_state=42)
        
        corr = numeric_df.corr()
        
        fig = px.imshow(
            corr,
            text_auto=True, 
            aspect="auto",
            title="Features Correlation Heatmap",
            color_continuous_scale='RdBu_r'
        )
        return fig
    except Exception as e:
        logger.error(f"Error in correlation_heatmap: {e}")
        fig = px.scatter(title='Correlation Analysis (Error Occurred)')
        return fig

def plot_churn_by_tenure(df):
    """
    Returns a histogram showing churn distribution by tenure.
    """
    try:
        # Check if tenure column exists
        if 'tenure' not in df.columns:
            # Try to find a similar column name
            tenure_cols = [col for col in df.columns if 'tenure' in col.lower()]
            if tenure_cols:
                tenure_col = tenure_cols[0]
            else:
                # Return a simple histogram if tenure not found
                fig = px.bar(title='Tenure Distribution (Feature Not Found)')
                return fig
        else:
            tenure_col = 'tenure'
        
        # Check if the feature is numerical
        if not pd.api.types.is_numeric_dtype(df[tenure_col]):
            # If not numerical, return a bar chart
            value_counts = df[tenure_col].value_counts().reset_index()
            value_counts.columns = [tenure_col, 'count']
            fig = px.bar(value_counts, x=tenure_col, y='count', title=f'{tenure_col} Distribution')
            return fig
        
        # Sample data if too large for performance
        if len(df) > 10000:
            df_sample = df.sample(n=10000, random_state=42)
        else:
            df_sample = df
        
        fig = px.histogram(
            df_sample, 
            x=tenure_col, 
            color='Churn',
            title='Churn Distribution by Tenure',
            nbins=30,
            color_discrete_map={'Yes': '#EF553B', 'No': '#636EFA'}
        )
        fig.update_layout(barmode='overlay')
        fig.update_traces(opacity=0.75)
        return fig
    except Exception as e:
        logger.error(f"Error in plot_churn_by_tenure: {e}")
        fig = px.bar(title='Tenure Distribution (Error Occurred)')
        return fig

def plot_monthly_charges_distribution(df):
    """
    Returns a violin plot showing Monthly Charges distribution by Churn.
    """
    try:
        # Check if MonthlyCharges column exists
        if 'MonthlyCharges' not in df.columns:
            # Try to find a similar column name
            mc_cols = [col for col in df.columns if 'monthly' in col.lower() and 'charge' in col.lower()]
            if mc_cols:
                mc_col = mc_cols[0]
            else:
                # Return a simple violin plot if MonthlyCharges not found
                fig = px.violin(title='Monthly Charges Distribution (Feature Not Found)')
                return fig
        else:
            mc_col = 'MonthlyCharges'
        
        # Check if the feature is numerical
        if not pd.api.types.is_numeric_dtype(df[mc_col]):
            # If not numerical, return a bar chart
            value_counts = df[mc_col].value_counts().reset_index()
            value_counts.columns = [mc_col, 'count']
            fig = px.bar(value_counts, x=mc_col, y='count', title=f'{mc_col} Distribution')
            return fig
        
        # Sample data if too large for performance
        if len(df) > 10000:
            df_sample = df.sample(n=10000, random_state=42)
        else:
            df_sample = df
        
        fig = px.violin(
            df_sample, 
            y=mc_col, 
            x='Churn',
            color='Churn',
            title='Monthly Charges Distribution by Churn Status',
            color_discrete_map={'Yes': '#EF553B', 'No': '#636EFA'}
        )
        return fig
    except Exception as e:
        logger.error(f"Error in plot_monthly_charges_distribution: {e}")
        fig = px.violin(title='Monthly Charges Distribution (Error Occurred)')
        return fig

def plot_contract_churn_heatmap(df):
    """
    Returns a heatmap showing churn rates by contract type and internet service.
    """
    try:
        # Check if required columns exist
        if 'Contract' not in df.columns or 'InternetService' not in df.columns:
            # Return a simple heatmap if required columns not found
            fig = px.imshow([[1, 2], [3, 4]], title='Contract/Internet Service Data Not Available')
            return fig
        
        # Sample data if too large for performance
        if len(df) > 10000:
            df_sample = df.sample(n=10000, random_state=42)
        else:
            df_sample = df
        
        # Create pivot table
        pivot = df_sample.pivot_table(index='Contract', columns='InternetService', values='Churn', aggfunc=lambda x: sum(x == 'Yes')/len(x) if len(x) > 0 else 0)
        
        fig = px.imshow(
            pivot,
            text_auto='.2%',
            aspect="auto",
            title='Churn Rate by Contract and Internet Service',
            color_continuous_scale='Reds'
        )
        return fig
    except Exception as e:
        logger.error(f"Error in plot_contract_churn_heatmap: {e}")
        fig = px.imshow([[1, 2], [3, 4]], title='Contract/Internet Service (Error Occurred)')
        return fig

def plot_feature_importance(feature_names, importances, top_n=10):
    """
    Returns a bar chart of feature importances.
    """
    try:
        # Create DataFrame and sort by importance
        feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feat_imp = feat_imp.sort_values('importance', ascending=False).head(top_n)
        
        fig = px.bar(
            feat_imp,
            x='importance',
            y='feature',
            orientation='h',
            title=f'Top {top_n} Feature Importances',
            color='importance',
            color_continuous_scale='Blues'
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        return fig
    except Exception as e:
        logger.error(f"Error in plot_feature_importance: {e}")
        fig = px.bar(title='Feature Importances (Error Occurred)')
        return fig

def plot_churn_trend_over_time(df):
    """
    Returns a line chart showing churn trend over time (based on tenure groups).
    """
    try:
        # Check if tenure column exists
        if 'tenure' not in df.columns:
            # Try to find a similar column name
            tenure_cols = [col for col in df.columns if 'tenure' in col.lower()]
            if tenure_cols:
                tenure_col = tenure_cols[0]
            else:
                # Return a simple line chart if tenure not found
                fig = px.line(title='Churn Trend Over Time (Tenure Data Not Found)')
                return fig
        else:
            tenure_col = 'tenure'
        
        # Check if the feature is numerical
        if not pd.api.types.is_numeric_dtype(df[tenure_col]):
            # If not numerical, return a bar chart
            value_counts = df[tenure_col].value_counts().reset_index()
            value_counts.columns = [tenure_col, 'count']
            fig = px.bar(value_counts, x=tenure_col, y='count', title=f'{tenure_col} Distribution')
            return fig
        
        # Sample data if too large for performance
        if len(df) > 10000:
            df_sample = df.sample(n=10000, random_state=42)
        else:
            df_sample = df
        
        # Create tenure groups
        df_sample['tenure_group'] = pd.cut(df_sample[tenure_col], bins=[0, 12, 24, 36, 48, 60, 72], 
                                    labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61-72'])
        
        # Calculate churn rate by tenure group
        churn_rates = df_sample.groupby('tenure_group')['Churn'].apply(lambda x: (x == 'Yes').sum() / len(x) if len(x) > 0 else 0).reset_index()
        churn_rates.columns = ['Tenure Group', 'Churn Rate']
        
        fig = px.line(
            churn_rates,
            x='Tenure Group',
            y='Churn Rate',
            markers=True,
            title='Churn Rate Trend by Tenure Group'
        )
        fig.update_layout(yaxis_tickformat='.1%')
        return fig
    except Exception as e:
        logger.error(f"Error in plot_churn_trend_over_time: {e}")
        fig = px.line(title='Churn Trend Over Time (Error Occurred)')
        return fig

def plot_payment_method_analysis(df):
    """
    Returns a stacked bar chart showing churn by payment method.
    """
    try:
        if 'PaymentMethod' not in df.columns:
            fig = px.bar(title='Payment Method Analysis (Data Not Available)')
            return fig
        
        # Check if the feature is categorical
        if pd.api.types.is_numeric_dtype(df['PaymentMethod']):
            # If numerical, convert to categorical
            df_copy = df.copy()
            df_copy['PaymentMethod'] = df_copy['PaymentMethod'].astype(str)
        else:
            df_copy = df
        
        # Sample data if too large for performance
        if len(df_copy) > 10000:
            df_sample = df_copy.sample(n=10000, random_state=42)
        else:
            df_sample = df_copy
        
        # Group by payment method and churn
        grouped = df_sample.groupby(['PaymentMethod', 'Churn']).size().reset_index(name='Count')
        
        fig = px.bar(
            grouped,
            x='PaymentMethod',
            y='Count',
            color='Churn',
            title='Churn Distribution by Payment Method',
            barmode='stack',
            color_discrete_map={'Yes': '#EF553B', 'No': '#636EFA'}
        )
        return fig
    except Exception as e:
        logger.error(f"Error in plot_payment_method_analysis: {e}")
        fig = px.bar(title='Payment Method Analysis (Error Occurred)')
        return fig

def plot_tenure_histogram(df):
    """
    Returns a histogram of customer tenure.
    """
    try:
        if 'tenure' not in df.columns:
            tenure_cols = [col for col in df.columns if 'tenure' in col.lower()]
            if tenure_cols:
                tenure_col = tenure_cols[0]
            else:
                fig = px.bar(title='Tenure Distribution (Data Not Available)')
                return fig
        else:
            tenure_col = 'tenure'
        
        # Check if the feature is numerical
        if not pd.api.types.is_numeric_dtype(df[tenure_col]):
            # If not numerical, return a bar chart
            value_counts = df[tenure_col].value_counts().reset_index()
            value_counts.columns = [tenure_col, 'count']
            fig = px.bar(value_counts, x=tenure_col, y='count', title=f'{tenure_col} Distribution')
            return fig
        
        # Sample data if too large for performance
        if len(df) > 10000:
            df_sample = df.sample(n=10000, random_state=42)
        else:
            df_sample = df
        
        fig = px.histogram(
            df_sample,
            x=tenure_col,
            nbins=30,
            title='Customer Tenure Distribution',
            marginal='box'
        )
        return fig
    except Exception as e:
        logger.error(f"Error in plot_tenure_histogram: {e}")
        fig = px.bar(title='Tenure Distribution (Error Occurred)')
        return fig

# New advanced charts for deeper insights

def plot_churn_by_multiple_features(df, feature1, feature2):
    """
    Returns a heatmap showing churn rates by two categorical features.
    """
    try:
        # Check if required columns exist
        if feature1 not in df.columns or feature2 not in df.columns:
            fig = px.imshow([[1, 2], [3, 4]], title=f'{feature1} vs {feature2} Data Not Available')
            return fig
        
        # Sample data if too large for performance
        if len(df) > 10000:
            df_sample = df.sample(n=10000, random_state=42)
        else:
            df_sample = df
        
        # Create pivot table
        pivot = df_sample.pivot_table(index=feature1, columns=feature2, values='Churn', aggfunc=lambda x: sum(x == 'Yes')/len(x) if len(x) > 0 else 0)
        
        fig = px.imshow(
            pivot,
            text_auto='.2%',
            aspect="auto",
            title=f'Churn Rate by {feature1} and {feature2}',
            color_continuous_scale='Reds'
        )
        return fig
    except Exception as e:
        logger.error(f"Error in plot_churn_by_multiple_features: {e}")
        fig = px.imshow([[1, 2], [3, 4]], title='Multi-feature Analysis (Error Occurred)')
        return fig

def plot_numerical_distribution_comparison(df, feature):
    """
    Returns a comparative histogram showing distribution of a numerical feature by churn status.
    """
    try:
        # Check if feature exists
        if feature not in df.columns:
            similar_cols = [col for col in df.columns if feature.lower() in col.lower()]
            if similar_cols:
                feature = similar_cols[0]
            else:
                fig = px.histogram(title=f'{feature} Distribution (Feature Not Found)')
                return fig
        
        # Check if the feature is numerical
        if not pd.api.types.is_numeric_dtype(df[feature]):
            value_counts = df[feature].value_counts().reset_index()
            value_counts.columns = [feature, 'count']
            fig = px.bar(value_counts, x=feature, y='count', title=f'{feature} Distribution')
            return fig
        
        # Sample data if too large for performance
        if len(df) > 10000:
            df_sample = df.sample(n=10000, random_state=42)
        else:
            df_sample = df
        
        fig = px.histogram(
            df_sample,
            x=feature,
            color='Churn',
            title=f'{feature} Distribution by Churn Status',
            marginal='rug',
            color_discrete_map={'Yes': '#EF553B', 'No': '#636EFA'}
        )
        return fig
    except Exception as e:
        logger.error(f"Error in plot_numerical_distribution_comparison: {e}")
        fig = px.histogram(title=f'{feature} Distribution (Error Occurred)')
        return fig

def plot_service_adoption_heatmap(df):
    """
    Returns a heatmap showing adoption rates of various services.
    """
    try:
        # Define service columns
        service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        # Filter to only columns that exist
        available_services = [col for col in service_cols if col in df.columns]
        
        if not available_services:
            fig = px.imshow([[1, 2], [3, 4]], title='Service Adoption Data Not Available')
            return fig
        
        # Sample data if too large for performance
        if len(df) > 10000:
            df_sample = df.sample(n=10000, random_state=42)
        else:
            df_sample = df
        
        # Calculate adoption rates
        adoption_rates = []
        for service in available_services:
            if pd.api.types.is_numeric_dtype(df_sample[service]):
                # For numeric columns, calculate mean
                rate = df_sample[service].mean()
            else:
                # For categorical columns, calculate percentage of "Yes" values
                rate = (df_sample[service] == 'Yes').mean()
            adoption_rates.append(rate)
        
        # Create heatmap data
        heatmap_data = pd.DataFrame({
            'Service': available_services,
            'Adoption_Rate': adoption_rates
        })
        
        fig = px.bar(
            heatmap_data,
            x='Service',
            y='Adoption_Rate',
            title='Service Adoption Rates',
            color='Adoption_Rate',
            color_continuous_scale='Blues'
        )
        fig.update_layout(yaxis_tickformat='.1%')
        return fig
    except Exception as e:
        logger.error(f"Error in plot_service_adoption_heatmap: {e}")
        fig = px.bar(title='Service Adoption Rates (Error Occurred)')
        return fig

def plot_churn_probability_by_tenure(df):
    """
    Returns a scatter plot showing churn probability by tenure and monthly charges.
    """
    try:
        # Check if required columns exist
        tenure_col = 'tenure' if 'tenure' in df.columns else None
        mc_col = 'MonthlyCharges' if 'MonthlyCharges' in df.columns else None
        
        if not tenure_col or not mc_col:
            fig = px.scatter(title='Churn Probability Analysis (Data Not Available)')
            return fig
        
        # Sample data if too large for performance
        if len(df) > 10000:
            df_sample = df.sample(n=10000, random_state=42)
        else:
            df_sample = df
        
        # Add churn probability (1 for Yes, 0 for No)
        df_sample['churn_prob'] = (df_sample['Churn'] == 'Yes').astype(int)
        
        fig = px.scatter(
            df_sample,
            x=tenure_col,
            y=mc_col,
            color='churn_prob',
            title='Churn Probability by Tenure and Monthly Charges',
            color_continuous_scale='Reds',
            hover_data=[tenure_col, mc_col, 'Churn']
        )
        return fig
    except Exception as e:
        logger.error(f"Error in plot_churn_probability_by_tenure: {e}")
        fig = px.scatter(title='Churn Probability Analysis (Error Occurred)')
        return fig

def plot_customer_value_segments(df):
    """
    Returns a scatter plot showing customer value segmentation.
    """
    try:
        # Check if required columns exist
        tenure_col = 'tenure' if 'tenure' in df.columns else None
        mc_col = 'MonthlyCharges' if 'MonthlyCharges' in df.columns else None
        
        if not tenure_col or not mc_col:
            fig = px.scatter(title='Customer Value Segmentation (Data Not Available)')
            return fig
        
        # Sample data if too large for performance
        if len(df) > 10000:
            df_sample = df.sample(n=10000, random_state=42)
        else:
            df_sample = df
        
        # Calculate customer value (tenure * monthly charges)
        df_sample['Customer_Value'] = df_sample[tenure_col] * df_sample[mc_col]
        
        fig = px.scatter(
            df_sample,
            x=tenure_col,
            y=mc_col,
            size='Customer_Value',
            color='Churn',
            title='Customer Value Segmentation',
            color_discrete_map={'Yes': '#EF553B', 'No': '#636EFA'},
            hover_data=[tenure_col, mc_col, 'Customer_Value']
        )
        return fig
    except Exception as e:
        logger.error(f"Error in plot_customer_value_segments: {e}")
        fig = px.scatter(title='Customer Value Segmentation (Error Occurred)')
        return fig

def plot_customer_journey(df):
    """
    Returns a Sankey diagram showing the customer journey.
    """
    try:
        # Check if required columns exist
        required_cols = ['Contract', 'InternetService', 'Churn']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            fig = px.bar(title='Customer Journey Visualization (Data Not Available)')
            return fig
        
        # Sample data if too large for performance
        if len(df) > 5000:
            df_sample = df.sample(n=5000, random_state=42)
        else:
            df_sample = df
        
        # Create journey stages
        df_sample['Journey_Stage'] = df_sample['Contract'] + ' -> ' + df_sample['InternetService']
        
        # Count transitions
        journey_counts = df_sample.groupby(['Journey_Stage', 'Churn']).size().reset_index(name='Count')
        
        # Create Sankey diagram data
        all_stages = list(set(journey_counts['Journey_Stage'].tolist()))
        all_outcomes = list(set(journey_counts['Churn'].tolist()))
        
        # Create node list
        nodes = all_stages + all_outcomes
        node_indices = {node: i for i, node in enumerate(nodes)}
        
        # Create link data
        sources = []
        targets = []
        values = []
        
        for _, row in journey_counts.iterrows():
            sources.append(node_indices[row['Journey_Stage']])
            targets.append(node_indices[row['Churn']])
            values.append(row['Count'])
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodes
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values
            )
        )])
        
        fig.update_layout(title_text="Customer Journey Analysis", font_size=10)
        return fig
    except Exception as e:
        logger.error(f"Error in plot_customer_journey: {e}")
        fig = px.bar(title='Customer Journey Visualization (Error Occurred)')
        return fig

def plot_cohort_analysis(df):
    """
    Returns a heatmap showing cohort analysis based on tenure groups and churn.
    """
    try:
        # Check if tenure column exists
        if 'tenure' not in df.columns:
            tenure_cols = [col for col in df.columns if 'tenure' in col.lower()]
            if tenure_cols:
                tenure_col = tenure_cols[0]
            else:
                fig = px.imshow([[1, 2], [3, 4]], title='Cohort Analysis (Tenure Data Not Available)')
                return fig
        else:
            tenure_col = 'tenure'
        
        # Check if Churn column exists
        if 'Churn' not in df.columns:
            fig = px.imshow([[1, 2], [3, 4]], title='Cohort Analysis (Churn Data Not Available)')
            return fig
        
        # Sample data if too large for performance
        if len(df) > 10000:
            df_sample = df.sample(n=10000, random_state=42)
        else:
            df_sample = df
        
        # Create tenure groups
        df_sample['tenure_group'] = pd.cut(df_sample[tenure_col], bins=[0, 12, 24, 36, 48, 60, 72], 
                                    labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61-72'])
        
        # Create cohort table
        cohort_table = df_sample.groupby(['tenure_group', 'Churn']).size().unstack(fill_value=0)
        
        # Calculate percentages
        cohort_percentage = cohort_table.div(cohort_table.sum(axis=1), axis=0)
        
        fig = px.imshow(
            cohort_percentage,
            text_auto='.1%',
            aspect="auto",
            title='Cohort Analysis: Churn Rate by Tenure Group',
            color_continuous_scale='Reds'
        )
        return fig
    except Exception as e:
        logger.error(f"Error in plot_cohort_analysis: {e}")
        fig = px.imshow([[1, 2], [3, 4]], title='Cohort Analysis (Error Occurred)')
        return fig

def plot_geographical_distribution(df):
    """
    Returns a choropleth map showing geographical distribution of customers.
    """
    try:
        # For demo purposes, we'll create a mock geographical visualization
        # since the dataset doesn't typically include geographical data
        
        # Check if State or similar column exists
        geo_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['state', 'city', 'region', 'area'])]
        
        if not geo_cols:
            # Create mock data for demonstration
            states = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI']
            churn_rates = np.random.uniform(0.1, 0.4, len(states))
            
            geo_data = pd.DataFrame({
                'State': states,
                'Churn_Rate': churn_rates
            })
        else:
            # Use actual data if available
            geo_col = geo_cols[0]
            if len(df) > 10000:
                df_sample = df.sample(n=10000, random_state=42)
            else:
                df_sample = df
                
            # Calculate churn rates by geographical area
            geo_data = df_sample.groupby(geo_col)['Churn'].apply(
                lambda x: (x == 'Yes').mean()
            ).reset_index()
            geo_data.columns = [geo_col, 'Churn_Rate']
        
        fig = px.bar(
            geo_data,
            x=geo_data.columns[0],
            y='Churn_Rate',
            title='Geographical Distribution of Churn Rates',
            color='Churn_Rate',
            color_continuous_scale='Reds'
        )
        fig.update_layout(yaxis_tickformat='.1%')
        return fig
    except Exception as e:
        logger.error(f"Error in plot_geographical_distribution: {e}")
        fig = px.bar(title='Geographical Distribution (Error Occurred)')
        return fig

def plot_predictive_analytics_dashboard(df):
    """
    Returns a dashboard showing predictive analytics metrics.
    """
    try:
        # Sample data if too large for performance
        if len(df) > 5000:
            df_sample = df.sample(n=5000, random_state=42)
        else:
            df_sample = df
        
        # Calculate various predictive metrics
        metrics_data = {
            'Metric': ['Churn Rate', 'Retention Rate', 'Avg. Tenure', 'CLV Estimate', 'Risk Score'],
            'Value': [
                (df_sample['Churn'] == 'Yes').mean(),
                (df_sample['Churn'] == 'No').mean(),
                df_sample['tenure'].mean() if 'tenure' in df_sample.columns else 0,
                df_sample['MonthlyCharges'].mean() * 12 if 'MonthlyCharges' in df_sample.columns else 0,
                (df_sample['Churn'] == 'Yes').mean() * 100  # Simplified risk score
            ]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        
        fig = px.bar(
            metrics_df,
            x='Metric',
            y='Value',
            title='Predictive Analytics Dashboard',
            color='Value',
            color_continuous_scale='Blues'
        )
        
        # Update specific formatting
        fig.update_traces(marker_coloraxis=None)
        for i, row in metrics_df.iterrows():
            if row['Metric'] in ['Churn Rate', 'Risk Score']:
                fig.data[i].marker.color = 'red'
            elif row['Metric'] in ['Retention Rate']:
                fig.data[i].marker.color = 'green'
            else:
                fig.data[i].marker.color = 'blue'
        
        fig.update_layout(yaxis_tickformat='.1%' if any(m in ['Churn Rate', 'Retention Rate'] for m in metrics_df['Metric']) else None)
        return fig
    except Exception as e:
        logger.error(f"Error in plot_predictive_analytics_dashboard: {e}")
        fig = px.bar(title='Predictive Analytics Dashboard (Error Occurred)')
        return fig
