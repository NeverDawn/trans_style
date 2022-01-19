
from django.urls import include, path

from . import views




urlpatterns = [
    path('', views.Make_Order, name='Make_Order'),
    path('make_order/', views.make_order, name='make_order'),
    path('my_order/', views.get_user_orders, name='my_order'),
    path('all_order/', views.get_all_orders, name='get_all_order'),
    path('change_state/', views.change_state, name='change_state'),
    path('change_state_admin/', views.change_state_admin, name='change_state_admin'),


]