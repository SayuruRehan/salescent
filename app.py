import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pytrends.request import TrendReq
import google.generativeai as genai
from datetime import datetime, timedelta
import json
import numpy as np
from typing import Dict, List, Any
import time

# MCP (Model Context Protocol) Implementation
class MCPContextManager:
    def __init__(self):
        self.context_history = []
        self.max_context_length = 10
        
    def add_context(self, user_input: str, trends_data: Dict, ai_response: str):
        """Add interaction to context history"""
        context_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "trends_data": trends_data,
            "ai_response": ai_response
        }
        self.context_history.append(context_entry)
        
        # Maintain context window
        if len(self.context_history) > self.max_context_length:
            self.context_history.pop(0)
    
    def get_context_summary(self) -> str:
        """Generate context summary for AI model"""
        if not self.context_history:
            return "No previous conversation history."
        
        summary = "Previous conversation context:\n"
        for entry in self.context_history[-3:]:  # Last 3 interactions
            summary += f"- User asked about: {entry['user_input']}\n"
            summary += f"- Trend insight: {entry['trends_data'].get('summary', 'N/A')}\n"
        
        return summary

class TrendsAnalyzer:
    def __init__(self):
        self.pytrends = TrendReq(hl='en-US', tz=360)
    
    def get_trends_data(self, keyword: str, timeframe: str = 'today 12-m') -> Dict[str, Any]:
        """Fetch and analyze trends data"""
        try:
            # Build payload with error handling
            self.pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo='', gprop='')
            
            # Get interest over time
            interest_data = self.pytrends.interest_over_time()
            
            if interest_data.empty:
                return {"error": f"No trend data found for '{keyword}'. Try a different keyword or timeframe."}
            
            # Remove 'isPartial' column if it exists
            if 'isPartial' in interest_data.columns:
                interest_data = interest_data.drop('isPartial', axis=1)
            
            # Check if keyword column exists
            if keyword not in interest_data.columns:
                available_cols = [col for col in interest_data.columns if col != 'isPartial']
                if not available_cols:
                    return {"error": f"No valid data columns found for '{keyword}'"}
                keyword = available_cols[0]  # Use first available column
            
            # Get related queries with error handling
            try:
                related_queries = self.pytrends.related_queries()
            except:
                related_queries = {}
            
            # Calculate trend metrics with better error handling
            values = interest_data[keyword].values
            
            if len(values) == 0:
                return {"error": "No data points available for analysis"}
            
            # Safe calculation of averages
            if len(values) >= 4:
                current_avg = np.mean(values[-4:])
                if len(values) >= 8:
                    previous_avg = np.mean(values[-8:-4])
                else:
                    previous_avg = np.mean(values[:-4]) if len(values) > 4 else current_avg
            else:
                current_avg = np.mean(values)
                previous_avg = current_avg
            
            # Calculate trend direction and strength
            if previous_avg > 0 and current_avg != previous_avg:
                trend_strength = abs(current_avg - previous_avg) / previous_avg * 100
                trend_direction = "increasing" if current_avg > previous_avg else "decreasing"
            else:
                trend_strength = 0
                trend_direction = "stable"
            
            # Generate forecast
            forecast = self._generate_simple_forecast(values)
            
            return {
                "keyword": keyword,
                "data": interest_data,
                "related_queries": related_queries,
                "current_avg": float(current_avg),
                "previous_avg": float(previous_avg),
                "trend_direction": trend_direction,
                "trend_strength": float(trend_strength),
                "forecast": forecast,
                "summary": f"{keyword} is {trend_direction} with {trend_strength:.1f}% change",
                "data_points": len(values)
            }
            
        except Exception as e:
            return {"error": f"Error fetching data: {str(e)}. Try a more common keyword."}
    
    def _generate_simple_forecast(self, values: np.ndarray, periods: int = 4) -> Dict:
        """Generate simple forecast based on trend data"""
        try:
            if len(values) < 2:
                return {"error": "Insufficient data for forecasting (need at least 2 data points)"}
            
            # Simple linear trend forecast
            x = np.arange(len(values))
            
            # Handle case where all values are the same
            if np.std(values) == 0:
                return {
                    "values": [float(values[0])] * periods,
                    "trend_slope": 0.0,
                    "confidence": "low",
                    "note": "Flat trend - no variation in historical data"
                }
            
            # Fit linear trend
            z = np.polyfit(x, values, 1)
            p = np.poly1d(z)
            
            # Forecast future periods
            future_x = np.arange(len(values), len(values) + periods)
            forecast_values = p(future_x)
            
            # Ensure non-negative values and reasonable bounds
            forecast_values = np.maximum(forecast_values, 0)
            forecast_values = np.minimum(forecast_values, 100)  # Cap at 100 for trends
            
            # Determine confidence based on trend consistency
            confidence = "high" if abs(z[0]) > 1 else "medium" if abs(z[0]) > 0.5 else "low"
            
            return {
                "values": forecast_values.tolist(),
                "trend_slope": float(z[0]),
                "confidence": confidence,
                "note": f"Forecast based on {len(values)} historical data points"
            }
            
        except Exception as e:
            return {"error": f"Forecasting error: {str(e)}"}

class GeminiChatbot:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def generate_response(self, user_input: str, trends_data: Dict, context: str) -> str:
        """Generate AI response based on trends data and context"""
        prompt = f"""
        You are a product demand forecasting expert. Analyze the following trends data and provide insights.
        
        User Query: {user_input}
        
        Context: {context}
        
        Trends Data:
        - Keyword: {trends_data.get('keyword', 'N/A')}
        - Current Average Interest: {trends_data.get('current_avg', 'N/A')}
        - Previous Average Interest: {trends_data.get('previous_avg', 'N/A')}
        - Trend Direction: {trends_data.get('trend_direction', 'N/A')}
        - Trend Strength: {trends_data.get('trend_strength', 'N/A')}%
        - Forecast Trend Slope: {trends_data.get('forecast', {}).get('trend_slope', 'N/A')}
        
        Please provide:
        1. A clear analysis of the current demand trend
        2. Actionable business insights
        3. Risk factors to consider
        4. Recommendations for the product category
        
        Keep the response conversational and insightful, around 200-300 words.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Sorry, I encountered an error generating the response: {str(e)}"

# Initialize session state
if 'mcp_context' not in st.session_state:
    st.session_state.mcp_context = MCPContextManager()
if 'trends_analyzer' not in st.session_state:
    st.session_state.trends_analyzer = TrendsAnalyzer()
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None

# Streamlit UI
st.set_page_config(
    page_title="Product Demand Forecasting Chatbot",
    page_icon="📈",
    layout="wide"
)

st.title("🤖 Product Demand Forecasting Chatbot")
st.markdown("Get AI-powered insights on product category demand using Google Trends data")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Gemini API Key input
    api_key = st.text_input("Enter Gemini API Key", type="password")
    if api_key:
        st.session_state.chatbot = GeminiChatbot(api_key)
        st.success("✅ Gemini API configured!")
    
    st.header("Settings")
    timeframe = st.selectbox(
        "Analysis Timeframe",
        ["today 3-m", "today 12-m", "today 5-y"],
        index=1
    )
    
    st.header("Quick Categories")
    quick_categories = [
        "smartphone", "laptop", "gaming chair", "fitness tracker",
        "electric car", "solar panels", "cryptocurrency", "air purifier"
    ]
    
    for category in quick_categories:
        if st.button(f"📊 {category.title()}", key=f"quick_{category}"):
            st.session_state.quick_search = category

# Main chat interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Chat Interface")
    
    # Chat input
    user_input = st.text_input(
        "Ask about product demand (e.g., 'What's the demand trend for smartphones?')",
        value=getattr(st.session_state, 'quick_search', ''),
        key="chat_input"
    )
    
    if st.button("🔍 Analyze Demand", type="primary") or getattr(st.session_state, 'quick_search', None) or getattr(st.session_state, 'suggested_search', None):
        # Handle different input sources
        if hasattr(st.session_state, 'quick_search'):
            user_input = st.session_state.quick_search
            delattr(st.session_state, 'quick_search')
        elif hasattr(st.session_state, 'suggested_search'):
            user_input = st.session_state.suggested_search
            delattr(st.session_state, 'suggested_search')
        
        if user_input and st.session_state.chatbot:
            with st.spinner("Analyzing trends and generating insights..."):
                # Better keyword extraction from user input
                keyword = user_input.lower()
                
                # Remove common question words and phrases
                remove_phrases = [
                    "what's the demand for", "what is the demand for",
                    "demand trend for", "trend for", "trends for",
                    "show me", "analyze", "analysis of",
                    "demand of", "popularity of", "interest in",
                    "how popular is", "market for"
                ]
                
                for phrase in remove_phrases:
                    keyword = keyword.replace(phrase, "")
                
                # Clean up the keyword
                keyword = keyword.replace("?", "").replace("!", "").strip()
                
                # Remove articles and common words
                stop_words = ["the", "a", "an", "in", "on", "at", "for", "with", "about"]
                keyword_words = [word for word in keyword.split() if word not in stop_words]
                keyword = " ".join(keyword_words) if keyword_words else keyword
                
                # If keyword is still empty or too generic, use original input
                if not keyword or len(keyword) < 2:
                    keyword = user_input.split()[-1] if user_input.split() else "smartphone"
                
                st.info(f"🔍 Analyzing keyword: **{keyword}**")
                
                # Get trends data
                trends_data = st.session_state.trends_analyzer.get_trends_data(keyword, timeframe)
                
                if "error" not in trends_data:
                    # Get context from MCP
                    context = st.session_state.mcp_context.get_context_summary()
                    
                    # Generate AI response
                    ai_response = st.session_state.chatbot.generate_response(user_input, trends_data, context)
                    
                    # Add to context
                    st.session_state.mcp_context.add_context(user_input, trends_data, ai_response)
                    
                    # Display response
                    st.subheader("🤖 AI Analysis")
                    st.write(ai_response)
                    
                    # Display trends chart
                    if not trends_data["data"].empty:
                        st.subheader("📈 Trends Visualization")
                        
                        fig = px.line(
                            trends_data["data"], 
                            x=trends_data["data"].index, 
                            y=keyword,
                            title=f"Search Interest for '{keyword}' Over Time"
                        )
                        fig.update_layout(
                            xaxis_title="Date",
                            yaxis_title="Search Interest",
                            hovermode='x'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Forecast visualization
                        if "forecast" in trends_data and "values" in trends_data["forecast"] and "error" not in trends_data["forecast"]:
                            st.subheader("🔮 Demand Forecast")
                            
                            try:
                                # Create forecast chart
                                historical_data = trends_data["data"][keyword].values
                                forecast_values = trends_data["forecast"]["values"]
                                
                                if len(historical_data) > 0 and len(forecast_values) > 0:
                                    # Create date range for visualization
                                    last_date = trends_data["data"].index[-1]
                                    
                                    # Generate future dates
                                    if timeframe == 'today 3-m':
                                        freq = 'D'
                                        periods = len(forecast_values)
                                    elif timeframe == 'today 5-y':
                                        freq = 'M'
                                        periods = len(forecast_values)
                                    else:  # 12-m default
                                        freq = 'W'
                                        periods = len(forecast_values)
                                    
                                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq=freq)
                                    
                                    fig_forecast = go.Figure()
                                    
                                    # Historical data
                                    fig_forecast.add_trace(go.Scatter(
                                        x=trends_data["data"].index,
                                        y=historical_data,
                                        mode='lines',
                                        name='Historical Data',
                                        line=dict(color='blue', width=2)
                                    ))
                                    
                                    # Forecast data
                                    fig_forecast.add_trace(go.Scatter(
                                        x=future_dates,
                                        y=forecast_values,
                                        mode='lines',
                                        name=f'Forecast ({trends_data["forecast"].get("confidence", "medium")} confidence)',
                                        line=dict(color='red', dash='dash', width=2)
                                    ))
                                    
                                    fig_forecast.update_layout(
                                        title=f"Demand Forecast for '{keyword}' (Next {len(forecast_values)} periods)",
                                        xaxis_title="Date",
                                        yaxis_title="Search Interest",
                                        hovermode='x',
                                        showlegend=True
                                    )
                                    
                                    st.plotly_chart(fig_forecast, use_container_width=True)
                                    
                                    # Forecast insights
                                    trend_slope = trends_data["forecast"].get("trend_slope", 0)
                                    if trend_slope > 0.1:
                                        st.success(f"📈 **Positive trend**: Demand is expected to increase (slope: +{trend_slope:.2f})")
                                    elif trend_slope < -0.1:
                                        st.warning(f"📉 **Negative trend**: Demand is expected to decrease (slope: {trend_slope:.2f})")
                                    else:
                                        st.info(f"➡️ **Stable trend**: Demand is expected to remain relatively stable (slope: {trend_slope:.2f})")
                                    
                                    if "note" in trends_data["forecast"]:
                                        st.caption(trends_data["forecast"]["note"])
                                        
                            except Exception as e:
                                st.error(f"Error creating forecast visualization: {str(e)}")
                                st.info("Forecast data is available but couldn't be visualized properly.")
                        
                        elif "forecast" in trends_data and "error" in trends_data["forecast"]:
                            st.warning(f"⚠️ Forecast unavailable: {trends_data['forecast']['error']}")
                        
                        # Display key metrics
                        col_metric1, col_metric2, col_metric3 = st.columns(3)
                        
                        with col_metric1:
                            st.metric(
                                "Data Points",
                                trends_data.get("data_points", "N/A"),
                                help="Number of historical data points used"
                            )
                        
                        with col_metric2:
                            st.metric(
                                "Trend Direction", 
                                trends_data.get("trend_direction", "N/A").title(),
                                delta=f"{trends_data.get('trend_strength', 0):.1f}%",
                                help="Direction and strength of recent trend"
                            )
                        
                        with col_metric3:
                            forecast_conf = trends_data.get("forecast", {}).get("confidence", "N/A")
                            st.metric(
                                "Forecast Confidence",
                                forecast_conf.title() if forecast_conf != "N/A" else "N/A",
                                help="Reliability of the forecast prediction"
                            )
                else:
                    st.error(f"❌ {trends_data['error']}")
                    st.info("💡 **Tips for better results:**")
                    st.write("• Try more popular/common keywords (e.g., 'iPhone', 'laptop', 'coffee')")
                    st.write("• Use English terms")
                    st.write("• Avoid very specific brand names or technical terms")
                    st.write("• Try different timeframes (3 months, 12 months, 5 years)")
                    
                    # Suggest alternatives
                    st.subheader("🔄 Try these popular categories:")
                    suggested_keywords = ["smartphone", "laptop", "electric car", "bitcoin", "netflix", "zoom", "tesla"]
                    cols = st.columns(len(suggested_keywords))
                    for i, suggestion in enumerate(suggested_keywords):
                        with cols[i]:
                            if st.button(f"📱 {suggestion}", key=f"suggest_{suggestion}"):
                                st.session_state.suggested_search = suggestion
                                st.rerun()
        
        elif not st.session_state.chatbot:
            st.warning("⚠️ Please enter your Gemini API key in the sidebar first.")
        else:
            st.warning("⚠️ Please enter a product category to analyze.")

with col2:
    st.header("📊 Quick Stats")
    
    # Display recent trends summary
    if st.session_state.mcp_context.context_history:
        latest_context = st.session_state.mcp_context.context_history[-1]
        trends_info = latest_context.get("trends_data", {})
        
        if trends_info and "error" not in trends_info:
            st.metric(
                "Current Trend",
                trends_info.get("trend_direction", "N/A").title(),
                delta=f"{trends_info.get('trend_strength', 0):.1f}%"
            )
            
            st.metric(
                "Average Interest",
                f"{trends_info.get('current_avg', 0):.1f}",
                delta=f"{trends_info.get('current_avg', 0) - trends_info.get('previous_avg', 0):.1f}"
            )
            
            # Related queries
            if trends_info.get("related_queries") and trends_info["related_queries"].get(trends_info["keyword"]):
                st.subheader("🔗 Related Searches")
                related = trends_info["related_queries"][trends_info["keyword"]]["top"]
                if related is not None and not related.empty:
                    for _, row in related.head(5).iterrows():
                        st.write(f"• {row['query']}")
    
    # Context history
    st.header("💭 Context History")
    if st.session_state.mcp_context.context_history:
        with st.expander("View Conversation History"):
            for i, entry in enumerate(reversed(st.session_state.mcp_context.context_history[-5:])):
                st.write(f"**Query {len(st.session_state.mcp_context.context_history)-i}:** {entry['user_input']}")
                st.write(f"*Analysis:* {entry['trends_data'].get('summary', 'N/A')}")
                st.write("---")
    else:
        st.info("No conversation history yet. Start by asking about a product category!")

# Footer
st.markdown("---")
st.markdown("""
**How to use:**
1. Enter your Gemini API key in the sidebar
2. Type a product category or question about demand
3. Get AI-powered insights with trends visualization and forecasting
4. The system maintains context across conversations using MCP

**Example queries:**
- "What's the demand trend for electric vehicles?"
- "Show me smartphone market demand"
- "Is there increasing interest in solar panels?"
""")