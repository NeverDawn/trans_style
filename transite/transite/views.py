import datetime
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.contenttypes.models import ContentType
from django.utils import timezone
from django.db.models import Sum
from django.core.cache import cache
from django.contrib import auth

def login(request):
    username = request.POST.get('username', '')
    password = request.POST.get('password', '')
    user = auth.authenticate(request, username=username, password=password)
    if user is not None:
        auth.login(request, user)
        return redirect('/')
    else:
        return render(request, 'error.html', {'message':'用户名或密码不正确'})

def register(request):
    username = request.POST.get('username', '')
    password = request.POST.get('password', '')
    u = User.objects.filter(username=username).first()
    if u:
        return render(request,'error.html',{'message':'该用户名已被注册'})
    elif len(password)<6:
        return render(request,'error.html',{'message':'密码太短，必须大于6个字符'})
    else:
        User.objects.create_user(username=username,password=password)
        user = auth.authenticate(request, username=username, password=password)
        auth.login(request, user)
        return redirect('/')

def lagout(request):
    auth.logout(request)
    return redirect('/')

def change_password(request):
    username = request.user
    password = request.POST.get('password', '')
    new_password = request.POST.get('new_password', '')
    user = auth.authenticate(request, username=username, password=password)
    if user is not None:
        InitialUsername = User.objects.get(username=username)
        InitialUsername.set_password(new_password)
        InitialUsername.save()
        user = auth.authenticate(request, username=username, password=new_password)
        auth.login(request, user)
        return redirect('/')
    else:
        return render(request, 'error.html', {'message':'原密码不正确'})
    

    
