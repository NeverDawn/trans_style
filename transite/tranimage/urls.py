from django.urls import include, path

from . import views
from . import trans
from django.conf.urls.static import static
from django.conf import settings



urlpatterns = [
    path('', views.Base, name='home'),
    path('CreateTrans/', views.CreateTrans, name='CreateTrans'),
    path('CreateTrans2/', views.CreateTrans2, name='CreateTrans2'),
    path('SortByWays/<slug:foo>/', views.SortByWays, name='SortByWays'),
    path('User_All/', views.User_All, name='User_All'),
    path('My_All/', views.My_All, name='My_All'),
    path("comments/", include("django_comments.urls")),
    
    path('progressurl/', trans.show_progress),
    path('search/', views.Search, name='search'),
    path('delete/', views.Delete, name='delete'),
    path('deleteComment/', views.deleteComment, name='deleteComment'),


]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
