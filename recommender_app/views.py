from django.shortcuts import redirect, render

from recommender import (
	chatbot_recommend,
	generate_meal_plan,
	recommend_by_goal,
	recommend_by_ingredients,
	recommend_by_name,
	recommend_custom,
	suggest_food_names,
)

from .forms import ChatbotForm, CustomForm, GoalForm, IngredientForm, NameForm
from .models import FavoriteRecipe


def _df_to_records(results):
	if results is None or results.empty:
		return []
	return results.to_dict(orient='records')


def _results_context(title, results, empty_message):
	records = _df_to_records(results)
	return {
		'title': title,
		'results': records,
		'empty_message': empty_message,
	}


def home(request):
	return render(request, 'home.html')


def goal_view(request):
	form = GoalForm(request.POST or None)
	if request.method == 'POST' and form.is_valid():
		goal = form.cleaned_data['goal']
		results = recommend_by_goal(goal, top_n=10)
		return render(
			request,
			'results.html',
			{
				**_results_context(
				f'Goal-based results: {goal}',
				results,
				'No recommendations found for this goal.',
				),
				'back_url': '/goal/',
			},
		)
	return render(request, 'goal.html', {'form': form})


def custom_view(request):
	form = CustomForm(request.POST or None)
	if request.method == 'POST' and form.is_valid():
		cleaned = form.cleaned_data
		results = recommend_custom(
			calories=cleaned['calories'],
			diet=cleaned['diet'] or None,
			taste=cleaned['taste'] or None,
			cost=cleaned['cost'] or None,
			prep_time=cleaned['prep_time'] or None,
			top_n=10,
		)
		return render(
			request,
			'results.html',
			{**_results_context('Custom recommendations', results, 'No recommendations matched your filters.'), 'back_url': '/custom/'},
		)
	return render(request, 'custom.html', {'form': form})


def name_view(request):
	form = NameForm(request.POST or None)
	suggestions = []
	error = ''
	if request.method == 'POST' and form.is_valid():
		food_name = form.cleaned_data['food_name']
		try:
			results = recommend_by_name(food_name, top_n=10)
			return render(
				request,
				'results.html',
				{**_results_context('Name-based recommendations', results, 'No similar foods found.'), 'back_url': '/name/'},
			)
		except ValueError:
			error = f'Food name not found: {food_name}'
			suggestions = suggest_food_names(food_name)
	return render(request, 'name.html', {'form': form, 'error': error, 'suggestions': suggestions})


def ingredient_view(request):
	form = IngredientForm(request.POST or None)
	if request.method == 'POST' and form.is_valid():
		raw = form.cleaned_data['ingredients']
		ingredient_list = [x.strip() for x in raw.split(',') if x.strip()]
		results = recommend_by_ingredients(ingredient_list, top_n=10)
		return render(
			request,
			'results.html',
			{**_results_context('Ingredient-based recommendations', results, 'No recommendations matched the ingredients.'), 'back_url': '/ingredients/'},
		)
	return render(request, 'ingredient.html', {'form': form})


def chatbot_view(request):
	form = ChatbotForm(request.POST or None)
	if request.method == 'POST' and form.is_valid():
		query = form.cleaned_data['query']
		results = chatbot_recommend(query, top_n=10)
		return render(
			request,
			'results.html',
			{**_results_context('Chatbot recommendations', results, 'No recommendations matched your query.'), 'back_url': '/chatbot/'},
		)
	return render(request, 'chatbot.html', {'form': form})


def demo_view(request):
	gym_user = recommend_by_goal('muscle_gain', top_n=5)
	sick_user = recommend_custom(calories=180, prep_time='quick', top_n=5)
	normal_user = recommend_by_goal('maintain', top_n=5)

	context = {
		'gym_results': _df_to_records(gym_user),
		'sick_results': _df_to_records(sick_user),
		'normal_results': _df_to_records(normal_user),
	}
	return render(request, 'demo.html', context)


def meal_plan_view(request):
	form = GoalForm(request.POST or None)
	selected_goal = 'maintain'
	if request.method == 'POST' and form.is_valid():
		selected_goal = form.cleaned_data['goal']
	elif request.method == 'GET':
		selected_goal = request.GET.get('goal', 'maintain')

	if selected_goal not in {'weight_loss', 'weight_gain', 'muscle_gain', 'maintain'}:
		selected_goal = 'maintain'

	if request.method != 'POST':
		form = GoalForm(initial={'goal': selected_goal})

	meal_plan = generate_meal_plan(selected_goal)
	context = {
		'title': 'Meal Plan',
		'form': form,
		'selected_goal': selected_goal,
		'meal_plan': meal_plan,
	}
	return render(request, 'meal_plan.html', context)


def favorites_view(request):
	favorites = FavoriteRecipe.objects.all()
	context = {
		'title': 'Favorites',
		'favorites': favorites,
	}
	return render(request, 'favorites.html', context)


def save_favorite_view(request):
	if request.method != 'POST':
		return redirect('favorites')

	name = (request.POST.get('name') or '').strip()
	diet = (request.POST.get('diet') or '').strip()
	calories_raw = (request.POST.get('calories') or '0').strip()

	if not name:
		return redirect('favorites')

	try:
		calories = float(calories_raw)
	except ValueError:
		calories = 0.0

	FavoriteRecipe.objects.create(name=name, calories=calories, diet=diet)

	next_url = request.POST.get('next') or request.META.get('HTTP_REFERER') or '/favorites/'
	return redirect(next_url)
