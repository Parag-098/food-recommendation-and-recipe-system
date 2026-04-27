from django import forms


GOAL_CHOICES = [
    ("weight_loss", "weight_loss"),
    ("weight_gain", "weight_gain"),
    ("muscle_gain", "muscle_gain"),
    ("maintain", "maintain"),
]

DIET_CHOICES = [
    ("", "Any"),
    ("veg", "veg"),
    ("non-veg", "non-veg"),
]

TASTE_CHOICES = [
    ("", "Any"),
    ("sweet", "sweet"),
    ("spicy", "spicy"),
]

COST_CHOICES = [
    ("", "Any"),
    ("low", "low"),
    ("medium", "medium"),
    ("high", "high"),
]

PREP_TIME_CHOICES = [
    ("", "Any"),
    ("quick", "quick"),
    ("medium", "medium"),
    ("long", "long"),
]


class GoalForm(forms.Form):
    goal = forms.ChoiceField(choices=GOAL_CHOICES, widget=forms.Select(attrs={"class": "form-control"}))


class CustomForm(forms.Form):
    calories = forms.DecimalField(
        required=False,
        min_value=0,
        widget=forms.NumberInput(attrs={"class": "form-control", "placeholder": "Optional"}),
    )
    diet = forms.ChoiceField(choices=DIET_CHOICES, required=False, widget=forms.Select(attrs={"class": "form-control"}))
    taste = forms.ChoiceField(choices=TASTE_CHOICES, required=False, widget=forms.Select(attrs={"class": "form-control"}))
    cost = forms.ChoiceField(choices=COST_CHOICES, required=False, widget=forms.Select(attrs={"class": "form-control"}))
    prep_time = forms.ChoiceField(
        choices=PREP_TIME_CHOICES,
        required=False,
        widget=forms.Select(attrs={"class": "form-control"}),
    )


class NameForm(forms.Form):
    food_name = forms.CharField(
        max_length=200,
        widget=forms.TextInput(attrs={"class": "form-control", "placeholder": "Enter food name"}),
    )


class IngredientForm(forms.Form):
    ingredients = forms.CharField(
        widget=forms.Textarea(
            attrs={"class": "form-control", "rows": 4, "placeholder": "Example: chicken, rice, garlic"}
        )
    )


class ChatbotForm(forms.Form):
    query = forms.CharField(
        widget=forms.Textarea(attrs={"class": "form-control", "rows": 4, "placeholder": "Ask for a recommendation"})
    )
