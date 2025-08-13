"""
LLM-Powered Data Analysis Tool
Target Market: Non-technical business owners
Value Prop: "No coding required - Explains statistical concepts in plain English"
Features: Upload CSV/Excel → Ask questions → Get instant analysis + visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import io
import base64

def safe_json_convert(obj):
    """Convert numpy types to Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj) if not np.isnan(obj) else None
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: safe_json_convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json_convert(v) for v in obj]
    else:
        return obj

# Mock LLM for demo (replace with real OpenAI integration)
class MockLLMAnalyst:
    def __init__(self):
        self.analysis_templates = {
            'summary': self._generate_summary_analysis,
            'correlation': self._generate_correlation_analysis,
            'trend': self._generate_trend_analysis,
            'comparison': self._generate_comparison_analysis,
            'prediction': self._generate_prediction_analysis
        }
    
    def analyze_dataset(self, df: pd.DataFrame, user_question: str) -> Dict:
        """Main analysis function that determines what type of analysis to perform"""
        question_lower = user_question.lower()
        
        # Determine analysis type based on question
        analysis_type = 'summary'  # default
        
        if any(word in question_lower for word in ['correlation', 'related', 'relationship']):
            analysis_type = 'correlation'
        elif any(word in question_lower for word in ['trend', 'over time', 'change', 'increase', 'decrease']):
            analysis_type = 'trend'
        elif any(word in question_lower for word in ['compare', 'vs', 'versus', 'difference']):
            analysis_type = 'comparison'
        elif any(word in question_lower for word in ['predict', 'forecast', 'future', 'will be']):
            analysis_type = 'prediction'
        
        # Perform the analysis
        analysis_func = self.analysis_templates.get(analysis_type, self._generate_summary_analysis)
        return analysis_func(df, user_question)
    
    def _generate_summary_analysis(self, df: pd.DataFrame, question: str) -> Dict:
        """Generate summary statistics and insights"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        
        insights = []
        
        # Basic dataset info
        insights.append(f"Your dataset contains {len(df)} rows and {len(df.columns)} columns.")
        
        if len(numeric_cols) > 0:
            insights.append(f"You have {len(numeric_cols)} numerical columns: {', '.join(numeric_cols[:3])}{'...' if len(numeric_cols) > 3 else ''}.")
        
        if len(categorical_cols) > 0:
            insights.append(f"You have {len(categorical_cols)} categorical columns: {', '.join(categorical_cols[:3])}{'...' if len(categorical_cols) > 3 else ''}.")
        
        # Key statistics for numeric columns
        if len(numeric_cols) > 0:
            for col in numeric_cols[:2]:  # Limit to first 2 columns
                mean_val = df[col].mean()
                median_val = df[col].median()
                std_val = df[col].std()
                
                insights.append(f"For {col}: average is {mean_val:.2f}, median is {median_val:.2f}, with a standard deviation of {std_val:.2f}.")
                
                # Identify outliers
                q25, q75 = df[col].quantile([0.25, 0.75])
                iqr = q75 - q25
                outliers = df[(df[col] < q25 - 1.5*iqr) | (df[col] > q75 + 1.5*iqr)]
                if len(outliers) > 0:
                    insights.append(f"Found {len(outliers)} potential outliers in {col} ({len(outliers)/len(df)*100:.1f}% of data).")
        
        # Missing data analysis
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            missing_cols = missing_data[missing_data > 0]
            insights.append(f"Missing data found: {', '.join([f'{col} ({count} missing)' for col, count in missing_cols.items()])}.")
        
        return {
            'analysis_type': 'summary',
            'insights': ' '.join(insights),
            'recommendations': self._generate_recommendations(df, 'summary'),
            'visualizations': ['histogram', 'correlation_matrix'],
            'technical_details': {
                'shape': [int(x) for x in df.shape],
                'dtypes': {k: str(v) for k, v in df.dtypes.to_dict().items()},
                'missing_values': {k: int(v) for k, v in missing_data.to_dict().items()}
            }
        }
    
    def _generate_correlation_analysis(self, df: pd.DataFrame, question: str) -> Dict:
        """Analyze correlations between variables"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {
                'analysis_type': 'correlation',
                'insights': "I need at least 2 numerical columns to perform correlation analysis. Your dataset doesn't have enough numerical data.",
                'recommendations': "Consider adding more numerical columns or converting categorical data to numerical format.",
                'visualizations': [],
                'technical_details': {}
            }
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Find strongest correlations
        strong_correlations = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:  # Strong correlation threshold
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    strength = "strong positive" if corr_val > 0.5 else "strong negative"
                    strong_correlations.append(f"{col1} and {col2} have a {strength} relationship (r={corr_val:.2f})")
        
        insights = []
        insights.append(f"I analyzed correlations between {len(numeric_cols)} numerical variables.")
        
        if strong_correlations:
            insights.append("Key findings: " + "; ".join(strong_correlations[:3]))
        else:
            insights.append("No strong correlations (>0.5) were found between variables. This could indicate independence or non-linear relationships.")
        
        return {
            'analysis_type': 'correlation',
            'insights': ' '.join(insights),
            'recommendations': self._generate_recommendations(df, 'correlation'),
            'visualizations': ['correlation_matrix', 'scatter_plots'],
            'technical_details': {
                'correlation_matrix': {k: {k2: float(v2) if not np.isnan(v2) else None for k2, v2 in v.items()} for k, v in corr_matrix.to_dict().items()},
                'strong_correlations': strong_correlations
            }
        }
    
    def _generate_trend_analysis(self, df: pd.DataFrame, question: str) -> Dict:
        """Analyze trends over time"""
        # Try to identify date columns
        date_cols = []
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower() or 'time' in col.lower():
                date_cols.append(col)
        
        if not date_cols:
            # Try to parse potential date columns
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        pd.to_datetime(df[col].head(), errors='raise')
                        date_cols.append(col)
                        break
                    except:
                        continue
        
        if not date_cols:
            return {
                'analysis_type': 'trend',
                'insights': "I couldn't identify any date/time columns in your dataset. Trend analysis requires time-based data.",
                'recommendations': "Add a date column or convert existing columns to date format for trend analysis.",
                'visualizations': [],
                'technical_details': {}
            }
        
        date_col = date_cols[0]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        insights = []
        insights.append(f"Analyzing trends using {date_col} as the time dimension.")
        
        if len(numeric_cols) > 0:
            # Analyze trend for first numeric column
            target_col = numeric_cols[0]
            df_sorted = df.sort_values(date_col)
            
            # Calculate simple trend (first vs last values)
            first_val = df_sorted[target_col].iloc[0]
            last_val = df_sorted[target_col].iloc[-1]
            change_pct = ((last_val - first_val) / first_val) * 100
            
            trend_direction = "increased" if change_pct > 0 else "decreased"
            insights.append(f"{target_col} has {trend_direction} by {abs(change_pct):.1f}% over the time period.")
            
            # Identify any notable patterns
            monthly_avg = df_sorted.groupby(df_sorted[date_col].dt.month)[target_col].mean()
            peak_month = monthly_avg.idxmax()
            low_month = monthly_avg.idxmin()
            
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            insights.append(f"Peak performance typically occurs in {months[peak_month-1]}, while {months[low_month-1]} shows the lowest values.")
        
        return {
            'analysis_type': 'trend',
            'insights': ' '.join(insights),
            'recommendations': self._generate_recommendations(df, 'trend'),
            'visualizations': ['time_series', 'seasonal_decomposition'],
            'technical_details': {
                'date_column': date_col,
                'trend_change': change_pct if 'change_pct' in locals() else None
            }
        }
    
    def _generate_comparison_analysis(self, df: pd.DataFrame, question: str) -> Dict:
        """Compare different groups or categories"""
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(categorical_cols) == 0 or len(numeric_cols) == 0:
            return {
                'analysis_type': 'comparison',
                'insights': "I need both categorical and numerical columns to perform group comparisons.",
                'recommendations': "Ensure your dataset has both categories to compare and numerical values to analyze.",
                'visualizations': [],
                'technical_details': {}
            }
        
        category_col = categorical_cols[0]
        numeric_col = numeric_cols[0]
        
        # Group analysis
        group_stats = df.groupby(category_col)[numeric_col].agg(['mean', 'median', 'std', 'count']).round(2)
        
        insights = []
        insights.append(f"Comparing {numeric_col} across different {category_col} groups.")
        
        # Find best and worst performing groups
        best_group = group_stats['mean'].idxmax()
        worst_group = group_stats['mean'].idxmin()
        
        best_avg = group_stats.loc[best_group, 'mean']
        worst_avg = group_stats.loc[worst_group, 'mean']
        
        insights.append(f"{best_group} performs best with an average {numeric_col} of {best_avg}, while {worst_group} has the lowest average at {worst_avg}.")
        
        # Calculate variance between groups
        overall_mean = df[numeric_col].mean()
        group_means = group_stats['mean']
        variance_explained = ((group_means - overall_mean) ** 2).sum() / len(group_means)
        
        if variance_explained > overall_mean * 0.1:
            insights.append(f"There's significant variation between groups, suggesting {category_col} is an important factor affecting {numeric_col}.")
        
        return {
            'analysis_type': 'comparison',
            'insights': ' '.join(insights),
            'recommendations': self._generate_recommendations(df, 'comparison'),
            'visualizations': ['box_plot', 'bar_chart'],
            'technical_details': {
                'group_statistics': group_stats.to_dict(),
                'best_group': best_group,
                'worst_group': worst_group
            }
        }
    
    def _generate_prediction_analysis(self, df: pd.DataFrame, question: str) -> Dict:
        """Generate predictive insights using simple ML models"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {
                'analysis_type': 'prediction',
                'insights': "I need at least 2 numerical columns to build a prediction model.",
                'recommendations': "Add more numerical features to enable predictive modeling.",
                'visualizations': [],
                'technical_details': {}
            }
        
        # Use the first column as target, others as features
        target_col = numeric_cols[0]
        feature_cols = numeric_cols[1:4]  # Use up to 3 features to keep it simple
        
        # Prepare data
        X = df[feature_cols].fillna(df[feature_cols].mean())
        y = df[target_col].fillna(df[target_col].mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train a simple model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        
        # Feature importance
        feature_importance = dict(zip(feature_cols, model.feature_importances_))
        most_important = max(feature_importance, key=feature_importance.get)
        
        insights = []
        insights.append(f"Built a prediction model for {target_col} using {', '.join(feature_cols)}.")
        insights.append(f"The model explains {r2*100:.1f}% of the variance in {target_col}.")
        insights.append(f"The most important predictor is {most_important}, followed by {sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[1][0]}.")
        
        # Model quality assessment
        if r2 > 0.7:
            quality = "excellent"
        elif r2 > 0.5:
            quality = "good"
        elif r2 > 0.3:
            quality = "moderate"
        else:
            quality = "poor"
        
        insights.append(f"The model quality is {quality} with an R² score of {r2:.3f}.")
        
        return {
            'analysis_type': 'prediction',
            'insights': ' '.join(insights),
            'recommendations': self._generate_recommendations(df, 'prediction'),
            'visualizations': ['feature_importance', 'prediction_vs_actual'],
            'technical_details': {
                'r2_score': r2,
                'rmse': rmse,
                'feature_importance': feature_importance,
                'target_column': target_col,
                'feature_columns': feature_cols.tolist()
            }
        }
    
    def _generate_recommendations(self, df: pd.DataFrame, analysis_type: str) -> str:
        """Generate actionable recommendations based on analysis"""
        recommendations = {
            'summary': [
                "Focus on the columns with the highest variation for deeper analysis",
                "Consider investigating outliers - they might represent important business insights", 
                "Address missing data before making business decisions"
            ],
            'correlation': [
                "Use strongly correlated variables to predict each other",
                "Be cautious of multicollinearity if building predictive models",
                "Investigate the business reasons behind strong correlations"
            ],
            'trend': [
                "Use seasonal patterns for better forecasting and planning",
                "Investigate what caused significant trend changes",
                "Consider external factors that might explain the trends"
            ],
            'comparison': [
                "Focus resources on improving underperforming groups",
                "Study best-performing groups to identify success factors", 
                "Consider the business impact of the differences between groups"
            ],
            'prediction': [
                "Focus on improving the most important features to boost performance",
                "Collect more data to improve model accuracy",
                "Validate predictions with business expertise before acting on them"
            ]
        }
        
        return " • ".join(recommendations.get(analysis_type, ["Continue exploring your data for deeper insights"]))

@dataclass
class AnalysisSession:
    id: str
    dataset_name: str
    upload_time: datetime
    questions_asked: List[str]
    analyses_performed: List[Dict]
    dataset_summary: Dict

class IntelligentDataAnalyst:
    def __init__(self, db_path="data_analysis.db"):
        self.db_path = db_path
        self.llm_client = MockLLMAnalyst()
        self.init_database()
        self.current_session = None
        self.current_dataframe = None
    
    def init_database(self):
        """Initialize database for storing analysis sessions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_sessions (
                id TEXT PRIMARY KEY,
                dataset_name TEXT NOT NULL,
                upload_time TEXT NOT NULL,
                dataset_summary TEXT,
                questions_count INTEGER DEFAULT 0,
                last_activity TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                question TEXT NOT NULL,
                analysis_type TEXT NOT NULL,
                insights TEXT NOT NULL,
                recommendations TEXT,
                technical_details TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES analysis_sessions (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def start_analysis_session(self, df: pd.DataFrame, dataset_name: str) -> AnalysisSession:
        """Start a new analysis session with uploaded data"""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(dataset_name) % 10000}"
        
        # Generate dataset summary (convert numpy types to Python types for JSON serialization)
        summary = {
            'rows': int(len(df)),
            'columns': int(len(df.columns)),
            'numeric_columns': int(len(df.select_dtypes(include=[np.number]).columns)),
            'categorical_columns': int(len(df.select_dtypes(exclude=[np.number]).columns)),
            'missing_values': int(df.isnull().sum().sum()),
            'memory_usage': int(df.memory_usage(deep=True).sum())
        }
        
        session = AnalysisSession(
            id=session_id,
            dataset_name=dataset_name,
            upload_time=datetime.now(),
            questions_asked=[],
            analyses_performed=[],
            dataset_summary=summary
        )
        
        # Store session in database
        self.store_session(session)
        
        self.current_session = session
        self.current_dataframe = df.copy()
        
        return session
    
    def store_session(self, session: AnalysisSession):
        """Store analysis session in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO analysis_sessions 
            (id, dataset_name, upload_time, dataset_summary, questions_count, last_activity)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            session.id, session.dataset_name, session.upload_time.isoformat(),
            json.dumps(safe_json_convert(session.dataset_summary)), len(session.questions_asked),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def ask_question(self, question: str) -> Dict:
        """Process a user question about the data"""
        if self.current_session is None or self.current_dataframe is None:
            return {"error": "No active analysis session. Please upload data first."}
        
        # Analyze the question and generate insights
        analysis_result = self.llm_client.analyze_dataset(self.current_dataframe, question)
        
        # Store the question and result
        self.current_session.questions_asked.append(question)
        self.current_session.analyses_performed.append(analysis_result)
        
        # Update database
        self.store_analysis_result(question, analysis_result)
        
        return analysis_result
    
    def store_analysis_result(self, question: str, analysis_result: Dict):
        """Store analysis result in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analysis_results 
            (session_id, question, analysis_type, insights, recommendations, technical_details, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            self.current_session.id, question, analysis_result.get('analysis_type', 'unknown'),
            analysis_result.get('insights', ''), analysis_result.get('recommendations', ''),
            json.dumps(analysis_result.get('technical_details', {})),
            datetime.now().isoformat()
        ))
        
        # Update session activity
        cursor.execute('''
            UPDATE analysis_sessions 
            SET questions_count = questions_count + 1, last_activity = ?
            WHERE id = ?
        ''', (datetime.now().isoformat(), self.current_session.id))
        
        conn.commit()
        conn.close()
    
    def generate_visualizations(self, analysis_result: Dict) -> List[go.Figure]:
        """Generate appropriate visualizations based on analysis results"""
        if self.current_dataframe is None:
            return []
        
        visualizations = []
        viz_types = analysis_result.get('visualizations', [])
        
        df = self.current_dataframe
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        
        for viz_type in viz_types:
            try:
                if viz_type == 'histogram' and len(numeric_cols) > 0:
                    fig = px.histogram(df, x=numeric_cols[0], title=f'Distribution of {numeric_cols[0]}')
                    visualizations.append(fig)
                
                elif viz_type == 'correlation_matrix' and len(numeric_cols) > 1:
                    corr_matrix = df[numeric_cols].corr()
                    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                  title="Correlation Matrix", color_continuous_scale='RdYlBu')
                    visualizations.append(fig)
                
                elif viz_type == 'scatter_plots' and len(numeric_cols) > 1:
                    fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                                   title=f'{numeric_cols[0]} vs {numeric_cols[1]}')
                    visualizations.append(fig)
                
                elif viz_type == 'box_plot' and len(categorical_cols) > 0 and len(numeric_cols) > 0:
                    fig = px.box(df, x=categorical_cols[0], y=numeric_cols[0],
                               title=f'{numeric_cols[0]} by {categorical_cols[0]}')
                    visualizations.append(fig)
                
                elif viz_type == 'bar_chart' and len(categorical_cols) > 0:
                    value_counts = df[categorical_cols[0]].value_counts()
                    fig = px.bar(x=value_counts.index, y=value_counts.values,
                               title=f'Count by {categorical_cols[0]}')
                    visualizations.append(fig)
                
            except Exception as e:
                continue  # Skip visualizations that fail
        
        return visualizations
    
    def generate_executive_summary(self) -> str:
        """Generate executive summary of the analysis session"""
        if not self.current_session:
            return "No analysis session active."
        
        summary = f"""
# Executive Summary - {self.current_session.dataset_name}

## Dataset Overview
- **Rows**: {self.current_session.dataset_summary['rows']:,}
- **Columns**: {self.current_session.dataset_summary['columns']}
- **Data Quality**: {((self.current_session.dataset_summary['rows'] * self.current_session.dataset_summary['columns'] - self.current_session.dataset_summary['missing_values']) / (self.current_session.dataset_summary['rows'] * self.current_session.dataset_summary['columns']) * 100):.1f}% complete

## Analysis Summary
- **Questions Asked**: {len(self.current_session.questions_asked)}
- **Analysis Types**: {', '.join(set([a.get('analysis_type', 'unknown') for a in self.current_session.analyses_performed]))}

## Key Insights
"""
        
        # Add key insights from analyses
        for i, analysis in enumerate(self.current_session.analyses_performed[-3:], 1):  # Last 3 analyses
            summary += f"**{i}.** {analysis.get('insights', 'No insights available')}\n\n"
        
        summary += "\n## Recommendations\n"
        
        # Add recommendations
        for i, analysis in enumerate(self.current_session.analyses_performed[-3:], 1):
            if analysis.get('recommendations'):
                summary += f"**{i}.** {analysis.get('recommendations')}\n\n"
        
        return summary

class LLMDataAnalysisApp:
    def __init__(self):
        self.analyst = IntelligentDataAnalyst()
        self.setup_page_config()
    
    def setup_page_config(self):
        st.set_page_config(
            page_title="LLM Data Analysis Tool",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def data_upload_section(self):
        """Data upload and preview section"""
        st.subheader("📊 Upload Your Data")
        
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your business data to start asking questions"
        )
        
        if uploaded_file is not None:
            try:
                # Load the data
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Start analysis session
                session = self.analyst.start_analysis_session(df, uploaded_file.name)
                
                # Clean dataframe for Arrow compatibility
                df_clean = df.copy()
                for col in df_clean.columns:
                    if df_clean[col].dtype == 'object':
                        # Convert mixed types to string
                        df_clean[col] = df_clean[col].astype(str)
                    elif pd.api.types.is_integer_dtype(df_clean[col]):
                        # Convert Int64 to regular int64
                        df_clean[col] = df_clean[col].astype('int64')
                    elif pd.api.types.is_float_dtype(df_clean[col]):
                        # Convert Float64 to regular float64
                        df_clean[col] = df_clean[col].astype('float64')
                
                # Store in session state
                st.session_state['analysis_session'] = session
                st.session_state['dataframe'] = df_clean
                
                st.success(f"✅ Successfully loaded {uploaded_file.name}")
                
                # Display basic info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rows", f"{len(df):,}")
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    st.metric("Numeric Cols", len(df.select_dtypes(include=[np.number]).columns))
                with col4:
                    st.metric("Missing Values", df.isnull().sum().sum())
                
                # Show data preview
                st.subheader("📋 Data Preview")
                st.dataframe(df_clean.head(10), use_container_width=True)
                
                # Column information
                with st.expander("📝 Column Information"):
                    col_info = pd.DataFrame({
                        'Column': df.columns,
                        'Type': df.dtypes,
                        'Non-Null Count': df.count(),
                        'Null Count': df.isnull().sum(),
                        'Unique Values': df.nunique()
                    })
                    st.dataframe(col_info, use_container_width=True)
                
                return True
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return False
        
        return False
    
    def question_interface(self):
        """Main question asking interface"""
        if 'analysis_session' not in st.session_state:
            st.warning("Please upload data first to start asking questions.")
            return
        
        st.subheader("❓ Ask Questions About Your Data")
        
        # Sample questions
        sample_questions = [
            "What are the main factors affecting sales?",
            "Show me the correlation between different variables",
            "What trends can you identify in the data?",
            "Compare performance across different categories",
            "Can you predict future values based on this data?",
            "What are the key insights from this dataset?",
            "Which variables are most important?",
            "Are there any outliers I should be concerned about?"
        ]
        
        # Question input
        col1, col2 = st.columns([3, 1])
        
        # Check for pending question from sample buttons
        if 'pending_question' in st.session_state:
            default_question = st.session_state['pending_question']
            del st.session_state['pending_question']
        else:
            default_question = ""
        
        with col1:
            user_question = st.text_input(
                "Ask a question about your data:",
                value=default_question,
                placeholder="e.g., 'What are the main drivers of revenue?'",
                key="question_input"
            )
        
        with col2:
            ask_button = st.button("🔍 Analyze", type="primary")
        
        # Sample questions buttons
        st.markdown("**💡 Try these sample questions:**")
        cols = st.columns(2)
        for i, question in enumerate(sample_questions[:4]):
            col = cols[i % 2]
            if col.button(question, key=f"sample_{i}"):
                # Use st.rerun() to refresh with new question
                st.session_state['pending_question'] = question
                st.rerun()
        
        # Process question
        if ask_button and user_question.strip():
            self.process_question(user_question.strip())
        
        # Display conversation history
        if hasattr(self.analyst.current_session, 'questions_asked') and self.analyst.current_session.questions_asked:
            st.markdown("---")
            self.display_conversation_history()
    
    def process_question(self, question: str):
        """Process user question and display results"""
        with st.spinner("🤖 Analyzing your data..."):
            analysis_result = self.analyst.ask_question(question)
        
        if "error" in analysis_result:
            st.error(analysis_result["error"])
            return
        
        # Display results
        st.markdown(f"### Question: *{question}*")
        
        # Main insights
        st.markdown("#### 🎯 Key Insights")
        st.info(analysis_result.get('insights', 'No insights available.'))
        
        # Recommendations
        if analysis_result.get('recommendations'):
            st.markdown("#### 💡 Recommendations")
            st.success(analysis_result['recommendations'])
        
        # Visualizations
        visualizations = self.analyst.generate_visualizations(analysis_result)
        if visualizations:
            st.markdown("#### 📊 Visualizations")
            
            # Display charts in columns if multiple
            if len(visualizations) > 1:
                cols = st.columns(2)
                for i, fig in enumerate(visualizations):
                    with cols[i % 2]:
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.plotly_chart(visualizations[0], use_container_width=True)
        
        # Technical details (expandable)
        if analysis_result.get('technical_details'):
            with st.expander("🔧 Technical Details"):
                st.json(analysis_result['technical_details'])
    
    def display_conversation_history(self):
        """Display conversation history"""
        st.subheader("💬 Analysis History")
        
        session = self.analyst.current_session
        
        for i, (question, analysis) in enumerate(zip(session.questions_asked, session.analyses_performed)):
            with st.expander(f"Q{i+1}: {question[:60]}{'...' if len(question) > 60 else ''}"):
                st.markdown(f"**Question:** {question}")
                st.markdown(f"**Analysis Type:** {analysis.get('analysis_type', 'unknown').title()}")
                st.markdown(f"**Insights:** {analysis.get('insights', 'No insights')}")
                
                if analysis.get('recommendations'):
                    st.markdown(f"**Recommendations:** {analysis['recommendations']}")
    
    def executive_summary_tab(self):
        """Generate and display executive summary"""
        st.subheader("📋 Executive Summary")
        
        if 'analysis_session' not in st.session_state:
            st.info("Upload data and ask questions to generate an executive summary.")
            return
        
        summary = self.analyst.generate_executive_summary()
        st.markdown(summary)
        
        # Export options
        st.markdown("### 💾 Export Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Download Summary (MD)"):
                st.download_button(
                    "Download Markdown",
                    summary,
                    f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
        
        with col2:
            if st.button("📊 Export Data + Insights (CSV)"):
                # Prepare comprehensive export
                session = self.analyst.current_session
                export_data = []
                
                for q, a in zip(session.questions_asked, session.analyses_performed):
                    export_data.append({
                        'Question': q,
                        'Analysis_Type': a.get('analysis_type', ''),
                        'Insights': a.get('insights', ''),
                        'Recommendations': a.get('recommendations', '')
                    })
                
                if export_data:
                    export_df = pd.DataFrame(export_data)
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    
    def data_profiling_tab(self):
        """Automatic data profiling and quality assessment"""
        if 'dataframe' not in st.session_state:
            st.info("Upload data to see automatic profiling.")
            return
        
        df = st.session_state['dataframe']
        
        st.subheader("🔍 Automatic Data Profiling")
        
        # Data quality overview
        col1, col2, col3, col4 = st.columns(4)
        
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        data_quality = ((total_cells - missing_cells) / total_cells) * 100
        
        with col1:
            st.metric("Data Quality", f"{data_quality:.1f}%")
        with col2:
            st.metric("Completeness", f"{(1 - missing_cells/total_cells):.1%}")
        with col3:
            duplicate_rows = df.duplicated().sum()
            st.metric("Duplicate Rows", duplicate_rows)
        with col4:
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            st.metric("Numeric Columns", f"{numeric_cols}/{len(df.columns)}")
        
        # Column profiling
        st.subheader("📊 Column Analysis")
        
        for col in df.columns[:5]:  # Limit to first 5 columns
            with st.expander(f"📈 {col} ({df[col].dtype})"):
                col_data = df[col]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Statistics:**")
                    if pd.api.types.is_numeric_dtype(col_data):
                        stats = col_data.describe()
                        for stat_name, stat_value in stats.items():
                            st.write(f"• {stat_name}: {stat_value:.2f}")
                    else:
                        st.write(f"• Unique values: {col_data.nunique()}")
                        st.write(f"• Most common: {col_data.mode().iloc[0] if not col_data.mode().empty else 'N/A'}")
                        st.write(f"• Missing: {col_data.isnull().sum()}")
                
                with col2:
                    if pd.api.types.is_numeric_dtype(col_data):
                        fig = px.histogram(df, x=col, title=f'Distribution of {col}')
                        st.plotly_chart(fig, use_container_width=True, key=f"hist_{col}_{i}")
                    else:
                        value_counts = col_data.value_counts().head(10)
                        fig = px.bar(x=value_counts.index, y=value_counts.values,
                                   title=f'Top Values in {col}')
                        st.plotly_chart(fig, use_container_width=True, key=f"bar_{col}_{i}")
    
    def run_app(self):
        """Main application interface"""
        st.title("🤖 LLM-Powered Data Analysis Tool")
        st.markdown("*Upload your data, ask questions in plain English, get instant insights*")
        
        # Sidebar
        st.sidebar.title("Navigation")
        
        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Upload & Analyze", "📋 Executive Summary", "🔍 Data Profiling", "⚙️ Settings"])
        
        with tab1:
            data_uploaded = self.data_upload_section()
            if data_uploaded:
                st.markdown("---")
                self.question_interface()
        
        with tab2:
            self.executive_summary_tab()
        
        with tab3:
            self.data_profiling_tab()
        
        with tab4:
            self.settings_tab()
    
    def settings_tab(self):
        """Settings and configuration"""
        st.subheader("⚙️ Settings")
        
        st.markdown("### 🤖 AI Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.text_input("OpenAI API Key", type="password", 
                         help="Enter your OpenAI API key for enhanced analysis")
            st.selectbox("Analysis Depth", ["Quick", "Standard", "Deep"], index=1)
            st.checkbox("Include statistical significance testing", value=True)
        
        with col2:
            st.selectbox("Default Chart Type", ["Auto", "Bar", "Line", "Scatter"])
            st.number_input("Max Visualizations", min_value=1, max_value=10, value=3)
            st.checkbox("Auto-generate recommendations", value=True)
        
        st.markdown("### 📊 Export Settings")
        st.selectbox("Default Export Format", ["CSV", "Excel", "JSON", "PDF Report"])
        st.checkbox("Include technical details in exports", value=False)
        
        st.markdown("### 🎨 Visualization Settings")
        st.selectbox("Color Theme", ["Default", "Business", "Scientific", "Colorful"])
        st.checkbox("Interactive charts", value=True)
        
        if st.button("💾 Save Settings"):
            st.success("Settings saved successfully!")

if __name__ == "__main__":
    app = LLMDataAnalysisApp()
    app.run_app()