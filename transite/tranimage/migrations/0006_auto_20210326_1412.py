# Generated by Django 3.1.5 on 2021-03-26 14:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('tranimage', '0005_auto_20210320_1351'),
    ]

    operations = [
        migrations.AlterField(
            model_name='transferimage',
            name='title',
            field=models.CharField(default='NameLess', max_length=50),
        ),
    ]