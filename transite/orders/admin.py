from django.contrib import admin
from .models import Material,Order,Border
# Register your models here.
@admin.register(Border)
class MaterialAdmin(admin.ModelAdmin):
    list_display = ('id', 'border_name','border_rate')


@admin.register(Material)
class MaterialAdmin(admin.ModelAdmin):
    list_display = ('id', 'material_name','material_rate')

@admin.register(Order)
class derAdmin(admin.ModelAdmin):
    list_display = ('photo_url','border','pantwidth','pantheight', 'material','cost_all','state','author','created_time','address')


