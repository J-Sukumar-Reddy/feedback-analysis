import pandas as pd
import streamlit as st
import os
import matplotlib.pyplot as plt

# Load data from CSV file
def load_data(file_path):
    return pd.read_csv(file_path)

# Preprocess feedback and calculate average rating
def preprocess_data(df):
    df['feedback'] = df['feedback'].astype(str)
    return df

# Analyze feedback for a given restaurant
def analyze_feedback(df, restaurant_name):
    if restaurant_name not in df['restaurant_name'].values:
        return "Restaurant not found in the dataset.", None, None
    
    restaurant_data = df[df['restaurant_name'] == restaurant_name]
    average_rating = restaurant_data['rating'].mean()
    feedback_texts = ' '.join(restaurant_data['feedback'])
    return average_rating, feedback_texts, restaurant_data['rating']

# Add feedback for a given restaurant
def add_feedback(df, restaurant_name, new_feedback, new_rating):
    new_entry = pd.DataFrame({
        'restaurant_name': [restaurant_name],
        'feedback': [new_feedback],
        'rating': [new_rating]
    })
    return pd.concat([df, new_entry], ignore_index=True)

# Plot pie chart of ratings
def plot_pie_chart(ratings):
    rating_counts = ratings.value_counts()
    fig, ax = plt.subplots()
    ax.pie(rating_counts, labels=rating_counts.index, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Rating Distribution')
    return fig

def main():
    st.title('Restaurant Feedback Analysis')

    # File path
    file_path = os.path.join('data', 'restaurant_feedback.csv')
    
    # Load and preprocess data
    data = load_data(file_path)
    data = preprocess_data(data)
    
    # Dropdown for restaurant selection
    restaurant_names = data['restaurant_name'].unique()
    selected_restaurant = st.selectbox("Select a restaurant:", restaurant_names)
    
    if selected_restaurant:
        # Analyze feedback
        average_rating, feedback_texts, ratings = analyze_feedback(data, selected_restaurant)
        
        if feedback_texts is None:
            st.error("Restaurant not found in the dataset.")
        else:
            st.subheader(f"Average Rating for '{selected_restaurant}': {average_rating:.2f}")
            st.subheader("Feedback Texts:")
            st.write(feedback_texts)

            # Show a bar chart of ratings
            st.subheader("Rating Distribution (Bar Chart)")
            st.bar_chart(ratings)
            
            # Show a pie chart of ratings
            st.subheader("Rating Distribution (Pie Chart)")
            pie_chart_fig = plot_pie_chart(ratings)
            st.pyplot(pie_chart_fig)
            
            # Feedback input
            st.subheader("Add Your Feedback")
            new_feedback = st.text_area("Your Feedback:")
            new_rating = st.slider("Your Rating (1-5):", min_value=1, max_value=5, value=3)
            
            if st.button("Submit Feedback"):
                if new_feedback:
                    # Add new feedback to the dataframe
                    data = add_feedback(data, selected_restaurant, new_feedback, new_rating)
                    # Save updated data to CSV
                    data.to_csv(file_path, index=False)
                    st.success("Thank you for your feedback!")
                else:
                    st.error("Please enter feedback before submitting.")

if __name__ == "__main__":
    main()
