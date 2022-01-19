from django.contrib import admin
from .models import TransferImageTag, TransferImage

@admin.register(TransferImageTag)
class TransferImageTagAdmin(admin.ModelAdmin):
    list_display = ('id', 'tag_name')

@admin.register(TransferImage)
class TransferImageAdmin(admin.ModelAdmin):
    list_display = ('title', 'style_photo', 'content_photo', 'output_photo','used_num', 'collect_num', 'author', 'tag', 'created_time', 'last_updated_time','style_weight','num_steps')
