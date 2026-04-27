# Food Recommendation and Recipe System

A Django-based food recommendation and recipe web application with a custom-built recommendation engine.

## Features

- Goal-based recommendations (`weight_loss`, `weight_gain`, `muscle_gain`, `maintain`)
- Name-based similar food recommendations
- Ingredient-based recommendations
- Custom filter recommendations (calories, diet, taste, cost, prep time)
- Chatbot-style recommendation input
- Meal planner (breakfast, lunch, dinner)
- Favorites system
- Health label and health score on recommendation cards

## Tech Stack

- Python
- Django
- Pandas
- HTML + CSS (no JavaScript)

## Project Structure

- `recommender.py`: Core recommendation engine and helper functions
- `recommender_app/`: Django app (views, forms, models, routes)
- `templates/`: Django HTML templates
- `static/css/styles.css`: Styling
- `build_final_dataset.py`: Dataset build script
- `final_food_dataset_final.csv`: Final prepared dataset used by the recommender

## Run Locally

1. Create and activate a virtual environment.
2. Install dependencies.
3. Run migrations:

```bash
python manage.py migrate
```

4. Start server:

```bash
python manage.py runserver
```

5. Open `http://127.0.0.1:8000/`.

## Dataset Notes

- The repository tracks `final_food_dataset_final.csv` for app usage.
- Large source CSVs are intentionally excluded from Git tracking using `.gitignore`.

## Repository

GitHub: https://github.com/Parag-098/food-recommendation-and-recipe-system
