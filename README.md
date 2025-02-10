# django_draws

A web app that hosts a drawing space, sending digits to a django server for classification.


# Build the django docker

docker compose up --build


# Share image with kubernetes local cluster

Make sure to enable kubernetes local registry:

'''microk8s enable registry'''


'''docker push localhost:32000/django-docker:registry

microk8s ctr image pull --plain-http localhost:32000/django-docker:registry'''


Launch the k8s pod:

'''microk8s kubectl apply -f djangodraws.yaml'''

Kill the k8s pod:

'''microk8s kubectl delete -f djangodraws.yaml'''


# Check the health of the k8s cluster

'''microk8s dashboard-proxy'''