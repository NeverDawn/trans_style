from django.db import models
from django.contrib.auth.models import User
from django.contrib.contenttypes.fields import GenericRelation
from django.urls import reverse
from tranimage.models import TransferImage

#from ckeditor_uploader.fields import RichTextUploadingField
#from read_statistics.models import ReadNumExpandMethod, ReadDetail

class Border(models.Model):
    border_name = models.CharField(max_length=15)
    border_rate = models.IntegerField(default=0)

    def __str__(self):
        return self.border_name

class Material(models.Model):
    material_name = models.CharField(max_length=15)
    material_rate = models.IntegerField(default=0)

    def __str__(self):
        return self.material_name

class Order(models.Model):
    object_id = models.IntegerField(default=0)
    photo_url = models.CharField(max_length=100)
    material = models.ForeignKey(Material, on_delete=models.DO_NOTHING)
    border = models.ForeignKey(Border, on_delete=models.DO_NOTHING)
    pantwidth = models.IntegerField(default=0)
    pantheight = models.IntegerField(default=0)
    pantarea = models.IntegerField(default=0)
    cost_all = models.IntegerField(default=0)
    state = models.IntegerField(default=0)    #0：尚未接单  1：制作中  2：已发出  3：完成  9:被删除
    author = models.ForeignKey(User, on_delete=models.DO_NOTHING)
    created_time = models.DateTimeField(auto_now_add=True)
    address = models.CharField(max_length=100)

    
