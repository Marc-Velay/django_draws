from django.shortcuts import render

from django.views import generic

# Create your views here.


def index(request):
    context = {}
    return render(request, "drawpage/index.html", context)