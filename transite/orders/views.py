from django.shortcuts import render,redirect
from django.contrib.contenttypes.models import ContentType
from django.contrib.auth.models import User
from .models import Material,Order,Border
from tranimage.models import TransferImage
import time
from django.core.files import File
from io import BytesIO
from urllib.request import urlopen
from PIL import Image

def get_user_orders(request):
    context = {}
    context['Orders'] = Order.objects.filter(author=request.user)

    return render(request,'get_user_orders.html', context)

def get_all_orders(request):
    context = {}
    context['Orders'] = Order.objects.all()

    return render(request,'get_user_orders.html', context)

def change_state(request):
    target = request.POST.get('tid', '')
    new_state = request.POST.get('state', '')

    obj = Order.objects.get(id=target)
    obj.state = new_state
    obj.save()
    context = {}
    context['Orders'] = Order.objects.filter(author=request.user)


    return render(request,'get_user_orders.html', context)

def change_state_admin(request):
    target = request.POST.get('tid', '')
    new_state = request.POST.get('state', '')

    obj = Order.objects.get(id=target)
    obj.state = new_state
    obj.save()
    context = {}
    context['Orders'] = Order.objects.all()


    return render(request,'get_user_orders.html', context)


def Make_Order(request):
    style = request.POST.get('style', '')

    context = {}
    context['transferImages'] = TransferImage.objects.filter(id=style)
    context['Material'] = Material.objects.all()
    context['Border'] = Border.objects.all()

    return render(request,'Make_Order.html', context)


def make_order(request):
    image_id = request.POST.get('tid',"")
    image_url = TransferImage.objects.get(id=str(image_id)).output_photo
    material = request.POST.get('material',"")
    border = request.POST.get('border',"")
    x = int(float(request.POST.get('x',"")))
    y = int(float(request.POST.get('y',"")))
    width = int(request.POST.get('width',""))
    height = int(request.POST.get('height',""))
    

    img = Image.open(image_url)
    img=img.resize((480,480),Image.ANTIALIAS) 

    photo_urlname = './media/order_img/'+'output_'+request.user.username+'_'+str(int(time.time()))+'.png'
    cropped = img.crop((x,y , x+width, y+height)) # (left, upper, right, lower)
    cropped.save(photo_urlname)
    order=Order()
    order.address = request.POST.get('address')
    order.pantheight = request.POST.get('pantheight')
    order.pantwidth = request.POST.get('pantwidth')
    order.pantarea = request.POST.get('pantarea')
    order.cost_all = request.POST.get('cost_all')

    order.object_id = image_id
    order.photo_url = photo_urlname
    order.state = 0
    order.author = request.user
    order.material = Material.objects.get(material_rate=str(material))
    order.border = Border.objects.get(border_rate=str(border))

    order.save()

    return redirect('/')