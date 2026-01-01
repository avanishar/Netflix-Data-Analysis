# Netflix Data Analysis

## Project Overview
This project performs **Exploratory Data Analysis (EDA)** on a Netflix dataset containing movies and TV shows. The goal is to uncover insights about content types, genres, ratings, countries of production, and trends over time using Python and Pandas.

---

## Dataset
- Source: [Kaggle Netflix Dataset](https://www.kaggle.com/shivamb/netflix-shows)
- Size: 8,807 records
- Columns include:
  - `show_id`, `type`, `title`, `director`, `cast`, `country`, `date_added`, `release_year`, `rating`, `duration`, `listed_in`, `description`

---

## Tools & Libraries
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib/Seaborn (optional for visuals)  
- **Environment:** Jupyter Notebook  

---

## Data Cleaning & Feature Engineering
- Handled missing values and duplicates  
- Converted `date_added` to datetime and extracted `year_added` and `month_added`  
- Cleaned text-based columns like `duration` and `listed_in`  
- Extracted numeric features from `duration` and `seasons` for analysis  

---

## Key Insights
- **Content Growth:** Netflix content increased significantly after 2015, showing global expansion  
- **Content Type:** Movies dominate the platform, but TV shows are growing  
- **Genres:** Drama is the most popular genre, followed by International Movies  
- **Countries:** The United States and India are top content-producing countries  
- **TV Shows:** Majority of TV shows have only 1 season, showing preference for short series  
- **Ratings:** Mature audiences dominate the library with ratings like TV-MA and TV-14  

---

## Conclusion
The analysis highlights Netflix’s strategy focusing on short-format, recent content and international expansion. These insights can help understand Netflix’s content acquisition strategy and audience targeting.

---

## Project Structure
