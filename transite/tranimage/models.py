from django.db import models
from django.contrib.auth.models import User
from django.contrib.contenttypes.fields import GenericRelation
from django.urls import reverse
from likes.models import LikeCount

#from ckeditor_uploader.fields import RichTextUploadingField
#from read_statistics.models import ReadNumExpandMethod, ReadDetail

class TransferImageTag(models.Model):
    tag_name = models.CharField(max_length=15)

    def __str__(self):
        return self.tag_name

    
class TransferImage(models.Model):
    title = models.CharField(max_length=50,default='NameLess') #名称
    style_photo = models.ImageField(upload_to="img/style/")  #风格图路径
    content_photo = models.ImageField(upload_to="img/content/")    #内容图路径
    output_photo = models.CharField(max_length=100)    #合成图路径
    #modle = models.CharField(max_length=100)
    used_num = models.IntegerField(default=0)  #被使用次数
    collect_num = models.IntegerField(default=0)  #收藏数
    author = models.ForeignKey(User, on_delete=models.DO_NOTHING)      #上传者
    tag = models.ForeignKey(TransferImageTag, on_delete=models.DO_NOTHING)
    created_time = models.DateTimeField(auto_now_add=True)
    last_updated_time = models.DateTimeField(auto_now=True)
    like_num = GenericRelation(LikeCount)
    style_weight = models.IntegerField(default=0)
    num_steps = models.IntegerField(default=0)
