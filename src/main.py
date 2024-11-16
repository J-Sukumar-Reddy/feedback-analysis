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

# Analyze feedback for a given restaurant and dish
def analyze_feedback(df, restaurant_name, dish):
    filtered_data = df[(df['restaurant_name'] == restaurant_name) & (df['dish'] == dish)]
    average_rating = filtered_data['rating'].mean()
    feedback_texts = ' '.join(filtered_data['feedback'])
    filtered_data['sentiment'] = filtered_data['feedback'].apply(sentiment_analysis)
    return average_rating, feedback_texts, filtered_data

# Add feedback for a given restaurant
def add_feedback(df, restaurant_name, new_feedback, new_rating, new_date, dish):
    new_entry = pd.DataFrame({
        'restaurant_name': [restaurant_name],
        'feedback': [new_feedback],
        'rating': [new_rating],
        'date': [new_date],
        'dish': [dish]
    })
    return pd.concat([df, new_entry], ignore_index=True)

# Predict future rating using a simple average-based model
def predict_future_rating(df, restaurant_name, dish):
    restaurant_data = df[(df['restaurant_name'] == restaurant_name) & (df['dish'] == dish)]
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
    
    # Create two columns for layout: one for restaurant selection and the other for dish selection
    choice = st.radio("Choose an option:", ["Select a Restaurant", "Select a Dish"])

    if choice == "Select a Restaurant":
        # Dropdown for restaurant selection
        restaurant_names = data['restaurant_name'].unique()
        selected_restaurant = st.selectbox("Select a restaurant:", restaurant_names)
        
        if selected_restaurant:
            # Display the list of dishes served at the selected restaurant
            restaurant_dishes = data[data['restaurant_name'] == selected_restaurant]['dish'].unique()
            selected_dish = st.selectbox("Select a dish:", restaurant_dishes)

            if selected_dish:
                # Analyze feedback for the selected restaurant and dish
                average_rating, feedback_texts, restaurant_data = analyze_feedback(data, selected_restaurant, selected_dish)
                
                if feedback_texts is None:
                    st.error(f"No feedback found for the selected dish at {selected_restaurant}.")
                else:
                    st.subheader(f"Average Rating for '{selected_dish}' at '{selected_restaurant}': {average_rating:.2f}")
                    st.subheader("Feedback Analysis Summary")
                    
                    # Show sentiment analysis results
                    sentiment_counts = restaurant_data['sentiment'].value_counts()
                    st.bar_chart(sentiment_counts)
                    
                    # Show trend analysis
                    st.subheader("Rating Trend Over Time")
                    trend_chart = plot_trend(restaurant_data)
                    st.plotly_chart(trend_chart)
                    
                    # Predict future rating
                    future_rating = predict_future_rating(data, selected_restaurant, selected_dish)
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
                            data = add_feedback(data, selected_restaurant, new_feedback, new_rating, new_date, selected_dish)
                            # Save updated data to CSV
                            data.to_csv(file_path, index=False)
                            st.success("Thank you for your feedback!")
                        else:
                            st.error("Please enter feedback before submitting.")
                    
                    # Map visualization
                    st.subheader("Restaurant Location on Map")
                    # Filter the data based on the selected restaurant
                    selected_restaurant_data = data[data['restaurant_name'] == selected_restaurant]

                    # Ensure the latitude and longitude columns exist, otherwise, you can generate or set default values
                    if 'latitude' in selected_restaurant_data.columns and 'longitude' in selected_restaurant_data.columns:
                        restaurant_location = selected_restaurant_data[['latitude', 'longitude']]
                    else:
                        # If latitude and longitude are missing, use random values for the demonstration
                        restaurant_location = pd.DataFrame({
                            'latitude': [np.random.uniform(low=17.3, high=17.6)],
                            'longitude': [np.random.uniform(low=78.4, high=78.6)]
                        })

                    # Show the map with the selected restaurant location
                    st.map(restaurant_location)
    
    elif choice == "Select a Dish":
        # Dropdown for dish selection
        dishes = data['dish'].unique()
        selected_dish = st.selectbox("Select a dish:", dishes)

        if selected_dish:
            # Find the restaurant with the highest average rating for the selected dish
            dish_data = data[data['dish'] == selected_dish]
            best_restaurant = dish_data.groupby('restaurant_name').agg(
                avg_rating=('rating', 'mean')
            ).idxmax()['avg_rating']
            
            # Display the best restaurant for the selected dish
            st.subheader(f"Best restaurant for '{selected_dish}' is: {best_restaurant}")
            
            # Analyze feedback for the best restaurant and selected dish
            average_rating, feedback_texts, restaurant_data = analyze_feedback(data, best_restaurant, selected_dish)
            
            if feedback_texts is None:
                st.error(f"No feedback found for the selected dish at {best_restaurant}.")
            else:
                st.subheader(f"Average Rating for '{selected_dish}' at '{best_restaurant}': {average_rating:.2f}")
                st.subheader("Feedback Analysis Summary")
                
                # Show sentiment analysis results
                sentiment_counts = restaurant_data['sentiment'].value_counts()
                st.bar_chart(sentiment_counts)
                
                # Show trend analysis
                st.subheader("Rating Trend Over Time")
                trend_chart = plot_trend(restaurant_data)
                st.plotly_chart(trend_chart)
                
                # Predict future rating
                future_rating = predict_future_rating(data, best_restaurant, selected_dish)
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
                        data = add_feedback(data, best_restaurant, new_feedback, new_rating, new_date, selected_dish)
                        # Save updated data to CSV
                        data.to_csv(file_path, index=False)
                        st.success("Thank you for your feedback!")
                    else:
                        st.error("Please enter feedback before submitting.")
                    
                    # Map visualization
                    st.subheader("Restaurant Location on Map")
                    # Filter the data based on the selected restaurant
                    selected_restaurant_data = data[data['restaurant_name'] == best_restaurant]

                    # Ensure the latitude and longitude columns exist, otherwise, you can generate or set default values
                    if 'latitude' in selected_restaurant_data.columns and 'longitude' in selected_restaurant_data.columns:
                        restaurant_location = selected_restaurant_data[['latitude', 'longitude']]
                    else:
                        # If latitude and longitude are missing, use random values for the demonstration
                        restaurant_location = pd.DataFrame({
                            'latitude': [np.random.uniform(low=17.3, high=17.6)],
                            'longitude': [np.random.uniform(low=78.4, high=78.6)]
                        })

                    # Show the map with the selected restaurant location
                    st.map(restaurant_location)

if __name__ == "__main__":
    main()
