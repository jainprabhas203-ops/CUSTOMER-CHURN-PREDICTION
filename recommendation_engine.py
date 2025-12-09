import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_recommendations(df):
    """
    Generate personalized customer retention recommendations.
    """
    try:
        recommendations = []
        
        # Ensure we're working with a copy to avoid SettingWithCopyWarning
        df_copy = df.copy()
        
        # Convert Churn to binary for calculations
        df_copy['churn_binary'] = (df_copy['Churn'] == 'Yes').astype(int)
        
        # Calculate customer lifetime value proxy (tenure * monthly charges)
        tenure_col = 'tenure' if 'tenure' in df_copy.columns else None
        mc_col = 'MonthlyCharges' if 'MonthlyCharges' in df_copy.columns else None
        
        if tenure_col and mc_col:
            df_copy['customer_value'] = df_copy[tenure_col] * df_copy[mc_col]
        else:
            df_copy['customer_value'] = 1  # Default value if columns not found
        
        # High value customers at risk
        high_value_at_risk = df_copy[
            (df_copy['churn_binary'] == 1) & 
            (df_copy['customer_value'] > df_copy['customer_value'].quantile(0.75))
        ]
        
        if len(high_value_at_risk) > 0:
            recommendations.append(
                f"🚨 Priority Alert: {len(high_value_at_risk)} high-value customers are at risk of churning. "
                f"Consider immediate retention offers such as discounts or loyalty rewards."
            )
        
        # Customers with short tenure but high monthly charges
        if tenure_col and mc_col:
            short_tenure_high_charge = df_copy[
                (df_copy[tenure_col] < df_copy[tenure_col].quantile(0.25)) &
                (df_copy[mc_col] > df_copy[mc_col].quantile(0.75)) &
                (df_copy['churn_binary'] == 1)
            ]
            
            if len(short_tenure_high_charge) > 0:
                recommendations.append(
                    f"⚠️ {len(short_tenure_high_charge)} customers have high monthly charges but short tenure. "
                    f"Consider reviewing pricing strategy or offering introductory promotions."
                )
        
        # Customers without premium services
        service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        available_service_cols = [col for col in service_cols if col in df_copy.columns]
        
        if available_service_cols:
            # Count how many services each customer has
            df_copy['services_count'] = df_copy[available_service_cols].apply(
                lambda row: (row == 'Yes').sum(), axis=1
            )
            
            low_services_high_risk = df_copy[
                (df_copy['services_count'] <= 1) & 
                (df_copy['churn_binary'] == 1)
            ]
            
            if len(low_services_high_risk) > 0:
                recommendations.append(
                    f"📱 {len(low_services_high_risk)} customers have minimal service packages and are at risk. "
                    f"Upselling additional services could improve retention."
                )
        
        # Contract-based recommendations
        if 'Contract' in df_copy.columns:
            month_to_month_churn = df_copy[
                (df_copy['Contract'] == 'Month-to-month') & 
                (df_copy['churn_binary'] == 1)
            ]
            
            if len(month_to_month_churn) > 0:
                recommendations.append(
                    f"📅 {len(month_to_month_churn)} month-to-month customers are at risk. "
                    f"Encourage them to switch to longer-term contracts with incentives."
                )
        
        # Payment method recommendations
        if 'PaymentMethod' in df_copy.columns:
            electronic_check_churn = df_copy[
                (df_copy['PaymentMethod'] == 'Electronic check') & 
                (df_copy['churn_binary'] == 1)
            ]
            
            if len(electronic_check_churn) > 0:
                recommendations.append(
                    f"💳 {len(electronic_check_churn)} customers paying by electronic check are at risk. "
                    f"Promote automatic payment methods for convenience and retention."
                )
        
        # Internet service recommendations
        if 'InternetService' in df_copy.columns:
            fiber_optic_churn = df_copy[
                (df_copy['InternetService'] == 'Fiber optic') & 
                (df_copy['churn_binary'] == 1)
            ]
            
            if len(fiber_optic_churn) > 0:
                recommendations.append(
                    f"🌐 {len(fiber_optic_churn)} Fiber optic customers are at risk. "
                    f"Ensure network quality and consider bundling services for better value."
                )
        
        # General recommendations if none of the above apply
        if not recommendations:
            recommendations.extend([
                "🔄 Implement regular customer satisfaction surveys to identify at-risk customers early.",
                "🎁 Develop a loyalty program to reward long-term customers and reduce churn.",
                "📞 Increase proactive customer outreach to build stronger relationships.",
                "📊 Monitor customer usage patterns to identify behavioral changes that may indicate churn risk.",
                "💬 Improve customer service response times and quality to enhance customer experience."
            ])
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error in generate_recommendations: {e}")
        # Return general recommendations if there's an error
        return [
            "🔄 Implement regular customer satisfaction surveys to identify at-risk customers early.",
            "🎁 Develop a loyalty program to reward long-term customers and reduce churn.",
            "📞 Increase proactive customer outreach to build stronger relationships.",
            "📊 Monitor customer usage patterns to identify behavioral changes that may indicate churn risk.",
            "💬 Improve customer service response times and quality to enhance customer experience."
        ]

def calculate_customer_lifetime_value(df):
    """
    Calculate estimated customer lifetime value.
    """
    try:
        # Ensure we're working with a copy to avoid SettingWithCopyWarning
        df_copy = df.copy()
        
        # Find required columns
        tenure_col = 'tenure' if 'tenure' in df_copy.columns else None
        mc_col = 'MonthlyCharges' if 'MonthlyCharges' in df_copy.columns else None
        churn_col = 'Churn' if 'Churn' in df_copy.columns else None
        
        if not all([tenure_col, mc_col, churn_col]):
            return None
        
        # Calculate CLV as tenure * monthly charges
        df_copy['clv'] = df_copy[tenure_col] * df_copy[mc_col]
        
        # Adjust for churn probability (simplified)
        df_copy['churn_prob'] = (df_copy[churn_col] == 'Yes').astype(int)
        df_copy['adjusted_clv'] = df_copy['clv'] * (1 - df_copy['churn_prob'])
        
        return df_copy[['clv', 'adjusted_clv']]
    except Exception as e:
        logger.error(f"Error in calculate_customer_lifetime_value: {e}")
        return None

def segment_customers(df):
    """
    Segment customers based on value and churn risk.
    """
    try:
        # Ensure we're working with a copy to avoid SettingWithCopyWarning
        df_copy = df.copy()
        
        # Find required columns
        tenure_col = 'tenure' if 'tenure' in df_copy.columns else None
        mc_col = 'MonthlyCharges' if 'MonthlyCharges' in df_copy.columns else None
        churn_col = 'Churn' if 'Churn' in df_copy.columns else None
        
        if not all([tenure_col, mc_col, churn_col]):
            return None
        
        # Calculate customer value
        df_copy['customer_value'] = df_copy[tenure_col] * df_copy[mc_col]
        
        # Create segments
        # Value segments
        value_q75 = df_copy['customer_value'].quantile(0.75)
        value_q25 = df_copy['customer_value'].quantile(0.25)
        
        df_copy['value_segment'] = pd.cut(
            df_copy['customer_value'],
            bins=[-np.inf, value_q25, value_q75, np.inf],
            labels=['Low Value', 'Medium Value', 'High Value']
        )
        
        # Risk segments
        df_copy['churn_risk'] = (df_copy[churn_col] == 'Yes').astype(int)
        
        # Combine segments
        df_copy['segment'] = df_copy['value_segment'].astype(str) + ' & ' + \
                            np.where(df_copy['churn_risk'] == 1, 'High Risk', 'Low Risk')
        
        return df_copy[['customer_value', 'value_segment', 'churn_risk', 'segment']]
    except Exception as e:
        logger.error(f"Error in segment_customers: {e}")
        return None