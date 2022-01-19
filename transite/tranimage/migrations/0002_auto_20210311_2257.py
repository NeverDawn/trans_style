# Generated by Django 2.0 on 2021-03-11 14:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('tranimage', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='transferimage',
            name='content_photo_path',
        ),
        migrations.RemoveField(
            model_name='transferimage',
            name='output_photo_path',
        ),
        migrations.RemoveField(
            model_name='transferimage',
            name='style_photo_path',
        ),
        migrations.AddField(
            model_name='transferimage',
            name='avatar',
            field=models.FileField(default='', upload_to='img/'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='transferimage',
            name='content_photo',
            field=models.FileField(default='', upload_to='img/'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='transferimage',
            name='output_photo',
            field=models.FileField(default='', upload_to='img/'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='transferimage',
            name='style_photo',
            field=models.FileField(default='', upload_to='img/'),
            preserve_default=False,
        ),
    ]