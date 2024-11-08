import pandas as pd
import streamlit as st
import os
import matplotlib.pyplot as plt
import numpy as np
from textblob import TextBlob
import plotly.express as px

# Load data from CSV file
def load_data(file_path):
    return pd.read_csv(file_path)

# Preprocess feedback and calculate average rating
def preprocess_data(df):
    df['feedback'] = df['feedback'].astype(str)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Convert to datetime, ignore errors
    else:
        df['date'] = pd.NaT  # Set to Not a Time if the column is missing
    return df


# Sentiment analysis
def sentiment_analysis(feedback):
    sentiment = TextBlob(feedback).sentiment.polarity
    if sentiment > 0:
        return 'Positive'
    elif sentiment == 0:
        return 'Neutral'
    else:
        return 'Negative'

# Analyze feedback for a given restaurant
def analyze_feedback(df, restaurant_name):
    if restaurant_name not in df['restaurant_name'].values:
        return "Restaurant not found in the dataset.", None, None
    
    restaurant_data = df[df['restaurant_name'] == restaurant_name]
    average_rating = restaurant_data['rating'].mean()
    feedback_texts = ' '.join(restaurant_data['feedback'])
    restaurant_data['sentiment'] = restaurant_data['feedback'].apply(sentiment_analysis)
    return average_rating, feedback_texts, restaurant_data

# Add feedback for a given restaurant
def add_feedback(df, restaurant_name, new_feedback, new_rating, new_date):
    new_entry = pd.DataFrame({
        'restaurant_name': [restaurant_name],
        'feedback': [new_feedback],
        'rating': [new_rating],
        'date': [new_date]
    })
    return pd.concat([df, new_entry], ignore_index=True)

# Predict future rating using a simple average-based model
def predict_future_rating(df, restaurant_name):
    restaurant_data = df[df['restaurant_name'] == restaurant_name]
    return restaurant_data['rating'].rolling(window=3).mean().iloc[-1] if len(restaurant_data) >= 3 else restaurant_data['rating'].mean()

# Plot pie chart of ratings
def plot_pie_chart(ratings):
    rating_counts = ratings.value_counts()
    fig, ax = plt.subplots()
    ax.pie(rating_counts, labels=rating_counts.index, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    plt.title('Rating Distribution')
    return fig

# Plot trend analysis
def plot_trend(restaurant_data):
    fig = px.line(restaurant_data, x='date', y='rating', title='Rating Trend Over Time')
    return fig

# Main function
def main():
    st.title('Restaurant Feedback Analytics and Insights')

    # File path
    file_path = os.path.join('data', 'restaurant_feedback01.csv')
    
    # Load and preprocess data
    data = load_data(file_path)
    data = preprocess_data(data)
    
    # Dropdown for restaurant selection
    restaurant_names = data['restaurant_name'].unique()
    selected_restaurant = st.selectbox("Select a restaurant:", restaurant_names)
    
    if selected_restaurant:
        # Analyze feedback
        average_rating, feedback_texts, restaurant_data = analyze_feedback(data, selected_restaurant)
        
        if feedback_texts is None:
            st.error("Restaurant not found in the dataset.")
        else:
            st.subheader(f"Average Rating for '{selected_restaurant}': {average_rating:.2f}")
            st.subheader("Feedback Analysis Summary")
            
            # Show sentiment analysis results
            sentiment_counts = restaurant_data['sentiment'].value_counts()
            st.bar_chart(sentiment_counts)
            
            # Show trend analysis
            st.subheader("Rating Trend Over Time")
            trend_chart = plot_trend(restaurant_data)
            st.plotly_chart(trend_chart)
            
            # Predict future rating
            future_rating = predict_future_rating(data, selected_restaurant)
            st.subheader(f"Predicted Future Rating: {future_rating:.2f}")
            
            # Show pie chart of ratings
            st.subheader("Rating Distribution (Pie Chart)")
            pie_chart_fig = plot_pie_chart(restaurant_data['rating'])
            st.pyplot(pie_chart_fig)
            
            # Feedback input
            st.subheader("Add Your Feedback")
            new_feedback = st.text_area("Your Feedback:")
            new_rating = st.slider("Your Rating (1-5):", min_value=1, max_value=5, value=3)
            new_date = st.date_input("Date of Feedback")
            
            if st.button("Submit Feedback"):
                if new_feedback:
                    # Add new feedback to the dataframe
                    data = add_feedback(data, selected_restaurant, new_feedback, new_rating, new_date)
                    # Save updated data to CSV
                    data.to_csv(file_path, index=False)
                    st.success("Thank you for your feedback!")
                else:
                    st.error("Please enter feedback before submitting.")
            
            # Map visualization
            st.subheader("Restaurant Locations on Map")
            # Example latitude and longitude values for demonstration
            data['latitude'] = np.random.uniform(low=17.3, high=17.6, size=len(data))
            data['longitude'] = np.random.uniform(low=78.4, high=78.6, size=len(data))
            st.map(data[['latitude', 'longitude']])
    
if __name__ == "__main__":
    main()
