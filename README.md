# Food Recommendation and Recipe System

This project is a full-stack Django application for food recommendation and recipe exploration.

It includes:
- a custom recommendation engine in Python
- a complete Django backend with forms, views, models, and routing
- responsive HTML + CSS UI (no JavaScript required)
- a dataset preparation pipeline that merges source data into a usable final dataset

GitHub Repository:
https://github.com/Parag-098/food-recommendation-and-recipe-system

## 1. What This Project Does

Users can get recommendations in multiple ways:
- by fitness/health goal
- by entering a food name
- by entering ingredients
- by custom filters (calories, diet, taste, cost, prep time)
- by chatbot-like natural query

The app also supports:
- meal planning (breakfast, lunch, dinner)
- saving favorite recipes
- health label and health score display for each recommendation card

## 2. Core Features and How They Work

### 2.1 Goal-Based Recommendation
Route: /goal/

Supported goals:
- weight_loss
- weight_gain
- muscle_gain
- maintain

Flow:
1. User selects goal.
2. Backend applies goal-specific calorie filtering first.
3. Additional goal-specific rules are applied (for example: avoid desserts for weight_loss and muscle_gain).
4. If enough candidates exist, similarity ranking is applied.
5. Results are diversified so very similar recipe names do not dominate the top list.
6. Final cards include calories, diet, health label, health score, explanation, ingredients, and steps.

### 2.2 Name-Based Recommendation
Route: /name/

Flow:
1. User enters a recipe name.
2. System normalizes input and tries exact/contains matching.
3. Similarity is computed against all candidates.
4. Optional filters can still be applied in backend calls.
5. If name is not found, the system returns close-name suggestions.

### 2.3 Ingredient-Based Recommendation
Route: /ingredients/

Flow:
1. User provides ingredient list.
2. Query text is converted into a manual TF-IDF vector.
3. Similarity is computed against recipe vectors.
4. Ingredient overlap boost is applied.
5. Top diversified recommendations are returned.

### 2.4 Custom Recommendation
Route: /custom/

Flow:
1. User sets optional filters: calories, diet, taste, cost, prep_time.
2. Candidates are filtered by chosen fields.
3. Similarity ranking is computed inside the filtered subset.
4. Results are sorted/diversified and returned.

### 2.5 Chatbot Recommendation
Route: /chatbot/

Flow:
1. User enters natural-language query.
2. Rule-based keyword parser infers intent and constraints.
3. If query matches a known goal intent, goal recommender is used.
4. Otherwise, ingredient-based or custom mode is selected.

### 2.6 Meal Planner
Route: /meal-plan/

Flow:
1. User selects goal.
2. Engine gets top recommendations for the goal.
3. First three picks are mapped to Breakfast, Lunch, Dinner.

### 2.7 Favorites
Routes:
- /favorites/
- /favorites/save/

Flow:
1. User clicks Save from result card.
2. Basic recipe fields are posted to backend.
3. FavoriteRecipe model stores name, calories, diet.
4. Favorites page lists saved items.

### 2.8 Health Label and Health Score
For each result:
- Health Label:
	- Low Calorie if calories < 200
	- High Energy if calories > 500
	- Balanced Meal otherwise
- Health Score:
	- Computed as a 1-10 score using calories, protein, and fat rules

## 3. Recommendation Algorithm (Manual Implementation)

The current recommender does not use sklearn/scipy model utilities for vectorization or similarity.

### 3.1 Manual TF-IDF
Implemented in recommender.py:
- tokenization using regex
- term frequency with Counter
- document frequency map over corpus
- idf formula:
	idf(term) = log((1 + N) / (1 + df(term))) + 1
- per-document sparse vectors stored as dictionaries

### 3.2 Manual Cosine Similarity
For two sparse vectors:
- dot product is computed only on overlapping terms
- norm is precomputed and cached
- cosine score = dot / (norm_a * norm_b)

### 3.3 Composite Scoring
Final score uses weighted combination:
- text similarity weight: 0.75
- calorie similarity weight: 0.20
- categorical similarity weight: 0.05

Where:
- calorie similarity is normalized by calorie range
- categorical similarity compares diet/taste/cost/prep_time

### 3.4 Performance Optimizations
- document vectors and norms are precomputed once at startup
- name-based similarity scores are cached
- filtering is applied before ranking in goal/custom flows
- ranking is limited to shortlist before final diversification

### 3.5 Diversity Control
Before final output:
- recipe name keywords are extracted
- over-repetitive name patterns are reduced
- backfill keeps output count stable

## 4. Backend Architecture

### 4.1 Django Project Layer
- food_project/settings.py configures installed apps, templates, static files, and sqlite database.
- food_project/urls.py routes root URLs to recommender_app.

### 4.2 Django App Layer
- recommender_app/forms.py defines all user input forms.
- recommender_app/views.py handles request parsing, recommender calls, and response rendering.
- recommender_app/models.py stores favorites.
- recommender_app/urls.py defines app endpoints.

### 4.3 Request Lifecycle
1. Request enters URL route.
2. View validates form input.
3. View calls recommender.py function.
4. Recommender returns DataFrame output.
5. View converts DataFrame to records.
6. Template renders cards with metadata and details.

## 5. UI and Frontend Structure

- Templates are in templates/.
- Shared layout and navbar: templates/base.html
- Light theme styling: static/css/styles.css
- No JavaScript is required for core interactions.
- All interactions are HTML form POST + Django rendering.

## 6. File-by-File Project Reference

This section explains what each main file/folder does.

### 6.1 Root Code and Config Files
- manage.py
	- Django command entry point.
- build_final_dataset.py
	- dataset pipeline script that inspects CSVs, auto-selects relevant sources, merges Food.com + USDA nutrition fields, derives helper columns, and writes final output.
- recommender.py
	- core recommendation engine (manual TF-IDF, manual cosine similarity, scoring, explanation logic, wrappers used by views).
- README.md
	- project documentation.
- .gitignore
	- controls what files are tracked.

### 6.2 Django Project Files
- food_project/__init__.py
	- marks folder as Python package.
- food_project/settings.py
	- global Django settings.
- food_project/urls.py
	- root URL routing.
- food_project/asgi.py
	- ASGI app entry.
- food_project/wsgi.py
	- WSGI app entry.

### 6.3 Django App Files
- recommender_app/__init__.py
	- package marker.
- recommender_app/apps.py
	- app config.
- recommender_app/admin.py
	- Django admin hooks.
- recommender_app/forms.py
	- GoalForm, CustomForm, NameForm, IngredientForm, ChatbotForm.
- recommender_app/models.py
	- FavoriteRecipe model.
- recommender_app/views.py
	- all backend view logic and recommender integration.
- recommender_app/urls.py
	- app routes.
- recommender_app/tests.py
	- test placeholder/area.
- recommender_app/migrations/0001_initial.py
	- initial schema migration (FavoriteRecipe).

### 6.4 Templates
- templates/base.html
	- shared shell: header, navbar, footer.
- templates/home.html
	- entry page with feature cards.
- templates/goal.html
	- goal input form.
- templates/custom.html
	- custom filter form.
- templates/name.html
	- name-based search form and suggestions.
- templates/ingredient.html
	- ingredient input form.
- templates/chatbot.html
	- chatbot query form.
- templates/results.html
	- unified result cards for recommendation outputs.
- templates/demo.html
	- prebuilt demo scenarios.
- templates/meal_plan.html
	- breakfast/lunch/dinner planner output.
- templates/favorites.html
	- saved favorites list.

### 6.5 Static Files
- static/css/styles.css
	- full UI styling for layout, cards, forms, navbar, footer, badges.

### 6.6 Data Files (Root CSVs)

#### Final Dataset Files
- final_food_dataset_final.csv
	- main runtime dataset used by recommender.
- final_food_dataset.csv
	- generated dataset variant from pipeline.

#### Food.com Related Files
- RAW_recipes.csv
- RAW_interactions.csv
- PP_recipes.csv
- PP_users.csv
- interactions_train.csv
- interactions_test.csv
- interactions_validation.csv

Purpose:
- recipe text, ingredients, tags, and interaction references used for dataset preparation and enrichment.

#### USDA and Nutrition Source Files
- acquisition_samples.csv
- agricultural_acquisition.csv
- branded_food.csv
- fndds_derivation.csv
- fndds_ingredient_nutrient_value.csv
- food.csv
- food_attribute.csv
- food_attribute_type.csv
- food_calorie_conversion_factor.csv
- food_category.csv
- food_component.csv
- food_nutrient.csv
- food_nutrient_conversion_factor.csv
- food_nutrient_derivation.csv
- food_nutrient_source.csv
- food_portion.csv
- food_protein_conversion_factor.csv
- food_update_log_entry.csv
- foundation_food.csv
- input_food.csv
- lab_method.csv
- lab_method_code.csv
- lab_method_nutrient.csv
- market_acquisition.csv
- measure_unit.csv
- nutrient.csv
- nutrient_incoming_name.csv
- retention_factor.csv
- sample_food.csv
- sr_legacy_food.csv
- sub_sample_food.csv
- sub_sample_result.csv
- survey_fndds_food.csv
- wweia_food_category.csv

Purpose:
- USDA nutritional schema and nutrient values used by build_final_dataset.py to derive calories/protein/fat and related metadata.

### 6.7 Local Runtime/Artifacts
- db.sqlite3
	- local Django database (favorites and auth/system tables).
- ingr_map.pkl
	- local artifact file.

## 7. Backend Functions and Responsibilities

### 7.1 Main Recommender Functions
From recommender.py:
- recommend_by_name
- recommend_by_goal
- recommend_custom
- recommend_by_ingredients
- chatbot_recommend
- generate_meal_plan
- compute_health_score
- suggest_food_names

### 7.2 View Functions
From recommender_app/views.py:
- home
- goal_view
- custom_view
- name_view
- ingredient_view
- chatbot_view
- demo_view
- meal_plan_view
- favorites_view
- save_favorite_view

## 8. Setup and Run Instructions

### 8.1 Environment
1. Create virtual environment.
2. Activate virtual environment.
3. Install dependencies:

```bash
pip install django pandas
```

### 8.2 Database Setup
```bash
python manage.py migrate
```

### 8.3 Start App
```bash
python manage.py runserver
```

Open:
http://127.0.0.1:8000/

## 9. Notes and Constraints

- Core recommendation behavior is implemented in recommender.py and consumed by Django views.
- UI uses server-rendered HTML and CSS, no client-side JavaScript logic required.
- Large raw CSV files are usually excluded from GitHub due size limits; final dataset file may be tracked explicitly.

## 10. Future Improvements

- Add duplicate prevention and delete action in favorites.
- Add tests for recommender scoring and Django views.
- Add model persistence for meal plans.
- Add deployment configuration for production use.
