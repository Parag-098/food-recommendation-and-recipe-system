from django.urls import path

from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('goal/', views.goal_view, name='goal'),
    path('custom/', views.custom_view, name='custom'),
    path('meal-plan/', views.meal_plan_view, name='meal_plan'),
    path('favorites/', views.favorites_view, name='favorites'),
    path('favorites/save/', views.save_favorite_view, name='save_favorite'),
    path('name/', views.name_view, name='name'),
    path('ingredients/', views.ingredient_view, name='ingredients'),
    path('chatbot/', views.chatbot_view, name='chatbot'),
    path('demo/', views.demo_view, name='demo'),
]
