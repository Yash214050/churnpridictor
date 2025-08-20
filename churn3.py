import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Dynamic Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #1f77b4;
}
.stMetric > label {
    font-size: 14px !important;
    color: #666 !important;
}
.nav-pills {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    border: 1px solid #e9ecef;
}
.nav-pills .nav-link {
    color: #495057;
    background-color: white;
    border: 1px solid #dee2e6;
    margin-right: 0.5rem;
    border-radius: 20px;
    padding: 0.5rem 1rem;
    transition: all 0.3s;
}
.nav-pills .nav-link.active {
    background-color: #007bff;
    border-color: #007bff;
    color: white;
}
.nav-pills .nav-link:hover {
    background-color: #e9ecef;
    border-color: #adb5bd;
}
</style>
""", unsafe_allow_html=True)

def create_navigation():
    """Create horizontal navigation bar using tabs."""
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 KPI Overview", 
        "🔍 Customer Explorer", 
        "🎯 Model Performance", 
        "🔮 Churn Predictor"
    ])
    
    return tab1, tab2, tab3, tab4

def load_and_validate_data(uploaded_file):
    """Load and validate the uploaded CSV file."""
    try:
        df = pd.read_csv(uploaded_file)
        return df, None
    except Exception as e:
        return None, str(e)

@st.cache_data
def prepare_data(df, target_col, churn_value, id_col=None, _file_name=None):
    """Prepare data for modeling."""
    # Create a copy of the dataframe
    data = df.copy()
    
    # Validate target column exists
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset. Available columns: {list(data.columns)}")
    
    # Remove ID column if specified
    if id_col and id_col in data.columns:
        data = data.drop(columns=[id_col])
    
    # Create binary target variable
    data['churn_binary'] = (data[target_col] == churn_value).astype(int)
    
    # Remove original target column
    if target_col in data.columns:
        data = data.drop(columns=[target_col])
    
    return data

@st.cache_resource
def train_model(data):
    """Train the logistic regression model with preprocessing pipeline."""
    X = data.drop('churn_binary', axis=1)
    y = data['churn_binary']
    
    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create preprocessing steps
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Create full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    # Check if stratification is possible
    min_class_count = min(np.bincount(y))
    use_stratify = min_class_count >= 2
    
    if not use_stratify:
        st.warning("⚠️ Dataset is highly imbalanced. Stratified split disabled.")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if use_stratify else None
    )
    
    # Train the model
    with st.spinner("Training model..."):
        pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }
    
    # Safe confusion matrix calculation
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        # Ensure 2x2 matrix even if one class is missing
        cm = np.zeros((2, 2), dtype=int)
        unique_labels = np.unique(np.concatenate([y_test, y_pred]))
        for i, true_label in enumerate([0, 1]):
            for j, pred_label in enumerate([0, 1]):
                if true_label in unique_labels and pred_label in unique_labels:
                    cm[i, j] = np.sum((y_test == true_label) & (y_pred == pred_label))
    
    return pipeline, metrics, cm, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, numerical_cols, categorical_cols

def get_feature_importance(pipeline, feature_names):
    """Extract and format feature importance from the trained model."""
    try:
        # Get coefficients
        coefficients = pipeline.named_steps['classifier'].coef_[0]
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': coefficients
        }).sort_values('importance', key=abs, ascending=False)
        
        return feature_importance
    except:
        return pd.DataFrame()

def sidebar_config():
    """Configure the sidebar for data upload and settings."""
    st.sidebar.title("🚀 Configuration")
    
    # Clear cache button
    if st.sidebar.button("🔄 Clear Cache", help="Clear cached data and models"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.sidebar.success("Cache cleared!")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload your CSV file",
        type=['csv'],
        help="Upload a CSV file containing customer data for churn prediction"
    )
    
    if uploaded_file is not None:
        # Load data
        df, error = load_and_validate_data(uploaded_file)
        
        if error:
            st.sidebar.error(f"Error loading file: {error}")
            return None, None, None, None, None
        
        st.sidebar.success(f"✅ File loaded successfully!")
        st.sidebar.info(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Target column selection
        st.sidebar.subheader("Target Configuration")
        target_col = st.sidebar.selectbox(
            "Select target column (churn indicator)",
            options=df.columns.tolist(),
            help="Select the column that indicates customer churn"
        )
        
        if target_col:
            unique_values = df[target_col].unique()
            churn_value = st.sidebar.selectbox(
                f"Value representing 'Churn' in {target_col}",
                options=unique_values,
                help="Select the value that represents a churned customer"
            )
            
            # Optional ID column
            st.sidebar.subheader("Optional Settings")
            id_cols = [None] + df.columns.tolist()
            id_col = st.sidebar.selectbox(
                "Select ID column to exclude (optional)",
                options=id_cols,
                help="Select a unique identifier column to exclude from modeling"
            )
            
            return df, target_col, churn_value, id_col, uploaded_file
    
    return None, None, None, None, None

def page_kpi_overview(data, target_col, churn_value):
    """Page 1: KPI Overview."""
    st.title("📈 KPI Overview")
    
    # Calculate main metrics
    total_customers = len(data)
    churn_rate = (data[target_col] == churn_value).mean()
    
    # Calculate tenure if available
    tenure_cols = [col for col in data.columns if 'tenure' in col.lower()]
    avg_tenure = data[tenure_cols[0]].mean() if tenure_cols else None
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="📊 Overall Churn Rate",
            value=f"{churn_rate:.2%}",
            help="Percentage of customers who churned"
        )
    
    with col2:
        st.metric(
            label="👥 Total Customers",
            value=f"{total_customers:,}",
            help="Total number of customers in the dataset"
        )
    
    with col3:
        if avg_tenure is not None:
            st.metric(
                label="⏰ Average Tenure",
                value=f"{avg_tenure:.1f}",
                help=f"Average value in {tenure_cols[0]} column"
            )
        else:
            st.metric(
                label="📋 Features",
                value=f"{len(data.columns)-1}",
                help="Total number of features available"
            )
    
    st.markdown("---")
    
    # Interactive charts
    st.subheader("📊 Feature Analysis")
    col1, col2 = st.columns(2)
    
    # Prepare data for analysis
    analysis_data = data.copy()
    analysis_data['Churn'] = (analysis_data[target_col] == churn_value).map({True: 'Churned', False: 'Retained'})
    
    numerical_cols = analysis_data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = analysis_data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove target column from options
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    with col1:
        st.subheader("🔢 Numerical Feature Analysis")
        if numerical_cols:
            selected_num_col = st.selectbox(
                "Select numerical feature:",
                options=numerical_cols,
                key="num_feature_kpi"
            )
            
            if selected_num_col:
                fig = px.box(
                    analysis_data,
                    x='Churn',
                    y=selected_num_col,
                    color='Churn',
                    title=f"{selected_num_col} Distribution by Churn Status"
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numerical features available")
    
    with col2:
        st.subheader("📊 Categorical Feature Analysis")
        if categorical_cols:
            selected_cat_col = st.selectbox(
                "Select categorical feature:",
                options=categorical_cols,
                key="cat_feature_kpi"
            )
            
            if selected_cat_col:
                # Create crosstab
                ct = pd.crosstab(analysis_data[selected_cat_col], analysis_data['Churn'], normalize='index') * 100
                
                fig = px.bar(
                    ct,
                    title=f"Churn Rate by {selected_cat_col}",
                    labels={'value': 'Percentage', 'index': selected_cat_col}
                )
                fig.update_layout(showlegend=True, legend_title="Status")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical features available")

def page_customer_explorer(data, target_col, churn_value):
    """Page 2: Customer Explorer."""
    st.title("🔍 Customer Explorer")
    
    # Prepare data
    analysis_data = data.copy()
    analysis_data['Churn'] = (analysis_data[target_col] == churn_value).map({True: 'Churned', False: 'Retained'})
    
    numerical_cols = analysis_data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = analysis_data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove target column from options
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔢 Numerical Features")
        if numerical_cols:
            num_feature = st.selectbox(
                "Select numerical feature:",
                options=numerical_cols,
                key="explorer_num"
            )
            
            if num_feature:
                # Histogram with overlayed distributions
                fig = px.histogram(
                    analysis_data,
                    x=num_feature,
                    color='Churn',
                    barmode='overlay',
                    nbins=30,
                    title=f"{num_feature} Distribution by Churn Status",
                    opacity=0.7
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                st.write("**Summary Statistics:**")
                summary = analysis_data.groupby('Churn')[num_feature].describe().round(2)
                st.dataframe(summary)
        else:
            st.info("No numerical features available")
    
    with col2:
        st.subheader("📊 Categorical Features")
        if categorical_cols:
            cat_feature = st.selectbox(
                "Select categorical feature:",
                options=categorical_cols,
                key="explorer_cat"
            )
            
            if cat_feature:
                # Count plot
                fig = px.histogram(
                    analysis_data,
                    x=cat_feature,
                    color='Churn',
                    barmode='group',
                    title=f"{cat_feature} Distribution by Churn Status"
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Churn rate by category
                churn_rate = analysis_data.groupby(cat_feature)['Churn'].apply(lambda x: (x == 'Churned').mean() * 100).round(2)
                st.write("**Churn Rate by Category:**")
                churn_df = pd.DataFrame({'Category': churn_rate.index, 'Churn Rate (%)': churn_rate.values})
                st.dataframe(churn_df)
        else:
            st.info("No categorical features available")

def page_model_performance(pipeline, metrics, cm, feature_importance):
    """Page 3: Model Performance."""
    st.title("🎯 Model Performance")
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Accuracy", f"{metrics['accuracy']:.3f}")
    with col2:
        st.metric("🎯 Precision", f"{metrics['precision']:.3f}")
    with col3:
        st.metric("🎯 Recall", f"{metrics['recall']:.3f}")
    with col4:
        st.metric("🎯 F1-Score", f"{metrics['f1']:.3f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🔥 Confusion Matrix")
        
        # Create confusion matrix heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Retained', 'Churned'],
                    yticklabels=['Retained', 'Churned'], ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
    
    with col2:
        st.subheader("📊 Feature Importance")
        
        if not feature_importance.empty:
            # Show top 15 features
            top_features = feature_importance.head(15)
            
            fig = px.bar(
                top_features,
                x='importance',
                y='feature',
                orientation='h',
                title='Top 15 Most Important Features',
                color='importance',
                color_continuous_scale='RdYlBu_r'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show interpretation
            st.info("""
            **Interpretation:**
            - Positive coefficients (red/yellow) increase churn probability
            - Negative coefficients (blue) decrease churn probability
            - Larger absolute values indicate stronger influence
            """)
        else:
            st.error("Could not extract feature importance from the model.")

def page_churn_predictor(pipeline, data, feature_importance, numerical_cols, categorical_cols):
    """Page 4: Churn Predictor with simplified UI."""
    st.title("🔮 Churn Predictor")
    
    st.info("Enter values for the most important features. Default values will be used for other features.")
    
    if feature_importance.empty:
        st.error("Feature importance not available. Cannot create prediction interface.")
        return
    
    # Get top 8 most important features
    top_features = feature_importance.head(8)['feature'].tolist()
    
    st.subheader("🎛️ Feature Input Panel")
    
    # Create input form
    with st.form("prediction_form"):
        user_inputs = {}
        
        # Create inputs in a 2x4 grid
        cols = st.columns(2)
        
        for i, feature in enumerate(top_features):
            col_idx = i % 2
            
            with cols[col_idx]:
                if feature in numerical_cols:
                    # Numerical input
                    mean_val = data[feature].mean()
                    min_val = data[feature].min()
                    max_val = data[feature].max()
                    
                    user_inputs[feature] = st.number_input(
                        f"{feature}",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(mean_val),
                        help=f"Range: {min_val:.2f} to {max_val:.2f}"
                    )
                    
                elif feature in categorical_cols:
                    # Categorical input
                    unique_vals = data[feature].unique()
                    mode_val = data[feature].mode().iloc[0] if len(data[feature].mode()) > 0 else unique_vals[0]
                    
                    user_inputs[feature] = st.selectbox(
                        f"{feature}",
                        options=unique_vals,
                        index=list(unique_vals).index(mode_val) if mode_val in unique_vals else 0
                    )
        
        # Predict button
        predict_button = st.form_submit_button("🔮 Predict Churn", type="primary")
    
    if predict_button:
        # Create prediction input using user inputs + dataset averages/modes
        prediction_input = {}
        
        # Fill with dataset defaults first
        for col in data.columns:
            if col != 'churn_binary':  # Skip target
                if col in numerical_cols:
                    prediction_input[col] = data[col].mean()
                elif col in categorical_cols:
                    prediction_input[col] = data[col].mode().iloc[0] if len(data[col].mode()) > 0 else data[col].unique()[0]
        
        # Override with user inputs
        for feature, value in user_inputs.items():
            prediction_input[feature] = value
        
        # Create DataFrame for prediction
        pred_df = pd.DataFrame([prediction_input])
        
        # Make prediction
        try:
            prediction = pipeline.predict(pred_df)[0]
            probability = pipeline.predict_proba(pred_df)[0, 1]
            
            # Display results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ **HIGH RISK** - Likely to Churn")
                else:
                    st.success("✅ **LOW RISK** - Likely to Stay")
            
            with col2:
                st.metric(
                    "🎲 Churn Probability",
                    f"{probability:.1%}",
                    help="Probability that this customer will churn"
                )
            
            with col3:
                risk_level = "High" if probability > 0.7 else "Medium" if probability > 0.3 else "Low"
                st.metric("📊 Risk Level", risk_level)
            
            # Show input summary
            st.subheader("📋 Input Summary")
            input_df = pd.DataFrame([{
                'Feature': k,
                'Value': v,
                'Type': 'User Input' if k in user_inputs else 'Dataset Default'
            } for k, v in prediction_input.items()])
            
            # Highlight user inputs
            st.dataframe(
                input_df.style.applymap(
                    lambda x: 'background-color: #e1f5fe' if x == 'User Input' else '',
                    subset=['Type']
                )
            )
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

def main():
    """Main application function."""
    st.title("🚀 Dynamic Churn Prediction Dashboard")
    st.markdown("Upload your customer data and get insights with automatic model training!")
    
    # Sidebar configuration
    df, target_col, churn_value, id_col, uploaded_file = sidebar_config()
    
    if df is not None and target_col is not None and churn_value is not None:
        # Prepare data
        try:
            processed_data = prepare_data(df, target_col, churn_value, id_col, uploaded_file.name)
        except ValueError as e:
            st.error(f"Data preparation error: {str(e)}")
            st.stop()
        
        # Train model
        try:
            pipeline, metrics, cm, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, numerical_cols, categorical_cols = train_model(processed_data)
            
            # Get feature names after preprocessing
            try:
                # Get feature names from the preprocessor
                num_features = numerical_cols
                cat_features = []
                
                if categorical_cols:
                    # Get categorical feature names after one-hot encoding
                    cat_transformer = pipeline.named_steps['preprocessor'].named_transformers_['cat']
                    if hasattr(cat_transformer.named_steps['onehot'], 'get_feature_names_out'):
                        cat_features = cat_transformer.named_steps['onehot'].get_feature_names_out(categorical_cols).tolist()
                    else:
                        cat_features = [f"{col}_{val}" for col in categorical_cols for val in df[col].unique()[1:]]
                
                all_feature_names = num_features + cat_features
                feature_importance = get_feature_importance(pipeline, all_feature_names)
                
            except Exception as e:
                st.warning(f"Could not extract feature names: {str(e)}")
                feature_importance = pd.DataFrame()
            
            # Create horizontal navigation using tabs
            tab1, tab2, tab3, tab4 = create_navigation()
            
            # Page content in tabs
            with tab1:
                page_kpi_overview(df, target_col, churn_value)
                
            with tab2:
                page_customer_explorer(df, target_col, churn_value)
                
            with tab3:
                page_model_performance(pipeline, metrics, cm, feature_importance)
                
            with tab4:
                page_churn_predictor(pipeline, processed_data, feature_importance, numerical_cols, categorical_cols)
            
        except Exception as e:
            st.error(f"Error in model training: {str(e)}")
            st.exception(e)
    
    else:
        # Welcome screen
        st.markdown("""
        ## 🎯 Welcome to the Dynamic Churn Prediction Dashboard!
        
        This application helps you:
        - 📊 **Analyze** customer churn patterns in your data
        - 🤖 **Train** machine learning models automatically
        - 📈 **Visualize** key performance indicators
        - 🔮 **Predict** individual customer churn risk
        
        ### 🚀 Getting Started:
        1. Upload your CSV file using the sidebar
        2. Configure your target column and churn value
        3. Explore the four main sections of the dashboard
        
        ### 📋 Requirements:
        - CSV file with customer data
        - A column indicating churn status
        - At least 10 rows of data for reliable predictions
        
        **Ready to start? Upload your file in the sidebar! 👈**
        """)

if __name__ == "__main__":
    main()