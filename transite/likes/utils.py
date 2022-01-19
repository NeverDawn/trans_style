
from django.contrib.contenttypes.models import ContentType

from django.db.models import Sum
from .models import LikeCount,LikeRecord

def GetLikesSort_Z(content_type):
    lists=[]
    sorts = LikeCount.objects.filter(content_type=content_type).order_by('liked_num')
    
    return lists

def GetLikesSort_N(content_type):
    lists=[]
    sorts = LikeCount.objects.filter(content_type=content_type).order_by('-liked_num')
    for i in sorts:
        lists.append(i.content_object)
    return lists