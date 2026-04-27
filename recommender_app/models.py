from django.db import models


class FavoriteRecipe(models.Model):
	name = models.CharField(max_length=255)
	calories = models.FloatField()
	diet = models.CharField(max_length=50)
	created_at = models.DateTimeField(auto_now_add=True)

	class Meta:
		ordering = ["-created_at"]

	def __str__(self) -> str:
		return f"{self.name} ({self.calories:.1f} kcal)"
