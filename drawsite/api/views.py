from django.shortcuts import render

from django.views import generic
from django.http import JsonResponse

import json

from .classifier import classify_img

def index(request):
    context = {}
    return render(request, "api/index.html", context)

def classify(request):
    if request.method == 'POST':
        req_data = json.loads(request.body.decode("utf-8"))
        b64_img = req_data['img']

        num_class = classify_img(b64_img)
        print(num_class)
    
        return JsonResponse({'number_class': num_class})
    
    context = {}
    return render(request, "api/index.html", context)