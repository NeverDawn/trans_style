from io import BytesIO
from django.shortcuts import get_object_or_404, render,redirect
from django.conf import settings
from django.db.models import Count
from django.contrib.contenttypes.models import ContentType
from django.contrib.auth.models import User
from .models import TransferImageTag, TransferImage
import time
from django.http import JsonResponse
from PIL import Image
from .trans import MakePic
from django.core.files import File
from io import BytesIO
from urllib.request import urlopen
from likes.utils import GetLikesSort_Z,GetLikesSort_N
from django.db.models import Q
from django_comments.models import Comment
from PIL import Image 
import base64

from django.core.files.uploadedfile import InMemoryUploadedFile

def User_All(request):
    username = request.POST.get('username', '')
    u = User.objects.filter(username=username).first()
    context = {}
    context['transferImages'] = TransferImage.objects.filter(author=u).order_by('created_time') 
    context['transferImageTags'] = TransferImageTag.objects.all()
    return render(request,'list.html', context)

def My_All(request):
    username = request.user
    u = User.objects.filter(username=username).first()
    context = {}
    context['transferImages'] = TransferImage.objects.filter(author=u).order_by('created_time') 
    context['transferImageTags'] = TransferImageTag.objects.all()
    return render(request,'list.html', context)    

def Search(request):
    target = request.POST.get('target', '')

    context = {}
    context['transferImages'] = TransferImage.objects.filter(Q(title__icontains=target)|Q(author__username__icontains=target)|Q(tag__tag_name__icontains=target))
    context['transferImageTags'] = TransferImageTag.objects.all()
    return render(request,'list.html', context)

def Delete(request):
    target = request.POST.get('target', '')

    obj=TransferImage.objects.get(id=target)
    obj.delete()
    return redirect('/')

def deleteComment(request):
    target = request.POST.get('object_id', '')
    data={}
    obj=Comment.objects.get(id=target)
    obj.delete()
    return JsonResponse(data)

def SortByWays(request,foo):
    context = {}
    context['transferImages'] = TransferImage.objects.all().order_by(foo) 
    context['transferImageTags'] = TransferImageTag.objects.all()
    return render(request,'list.html', context)

def Base(request):
    context = {}
    context['transferImages'] = TransferImage.objects.all().order_by('-created_time') 
    context['transferImageTags'] = TransferImageTag.objects.all()
    return render(request,'base.html', context)

def SortByWay(request):
    context = {}
    context['transferImages'] = TransferImage.objects.all().order_by('-created_time') 
    context['transferImageTags'] = TransferImageTag.objects.all()
    return render(request,'list.html', context)

def CreateTrans(request):
    style_photo_name = request.FILES.get('style_photo')
    content_photo_name = request.FILES.get('content_photo')
    style_weight = int(request.POST.get('style_weight',""))
    num_steps = int(request.POST.get('num_steps',""))
    photo_tag = request.POST.get('tag',"")
    modle_urlname = './media/model/'+'model_'+request.user.username+'_'+str(int(time.time()))+'.pth'
    style_urlname = './media/img/style/'+'style_'+request.user.username+'_'+str(int(time.time()))+'.png'
    content_urlname = './media/img/content/'+'content_'+request.user.username+'_'+str(int(time.time()))+'.png'
    output_urlname = './media/img/output/'+'output_'+request.user.username+'_'+str(int(time.time()))+'.png'
    
    transferImage=TransferImage()
    transferImage.title = request.POST.get('title')
    transferImage.style_photo = style_photo_name
    transferImage.content_photo = content_photo_name
    transferImage.output_photo = output_urlname
    transferImage.modle = modle_urlname
    transferImage.used_num = 0
    transferImage.collect_num = 0
    transferImage.author = request.user
    transferImage.tag = TransferImageTag.objects.get(id=str(photo_tag))
    transferImage.style_weight = style_weight
    transferImage.num_steps = num_steps

    MakePic(transferImage.style_photo ,transferImage.content_photo,transferImage.modle,style_weight,num_steps).save(output_urlname)
    transferImage.save()

    return redirect('/')

def CreateTrans2(request):
    style_photo_name = TransferImage.objects.filter(id=str(request.POST.get('style_photo'))).first().style_photo

    content_photo_name = request.FILES.get('content_photo')
    style_weight = int(request.POST.get('style_weight',""))
    num_steps = int(request.POST.get('num_steps',""))
    photo_tag = request.POST.get('tag',"")
    modle_urlname = './media/model/'+'model_'+request.user.username+'_'+str(int(time.time()))+'.pth'
    style_urlname = './media/img/style/'+'style_'+request.user.username+'_'+str(int(time.time()))+'.png'
    content_urlname = './media/img/content/'+'content_'+request.user.username+'_'+str(int(time.time()))+'.png'
    output_urlname = './media/img/output/'+'output_'+request.user.username+'_'+str(int(time.time()))+'.png'

    transferImage=TransferImage()
    transferImage.title = request.POST.get('title')
    transferImage.style_photo = style_photo_name
    transferImage.content_photo = content_photo_name
    transferImage.output_photo = output_urlname
    transferImage.modle = modle_urlname
    transferImage.used_num = 0
    transferImage.collect_num = 0
    transferImage.author = request.user
    transferImage.tag = TransferImageTag.objects.get(id=str(photo_tag))
    transferImage.style_weight = style_weight
    transferImage.num_steps = num_steps

    MakePic(style_photo_name ,transferImage.content_photo,transferImage.modle,style_weight,num_steps).save(output_urlname)
    transferImage.save()

    return redirect('/')
