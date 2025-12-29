import os
import jwt
import datetime as dt
from django.conf import settings
from django.http import JsonResponse, HttpResponse
from django.urls import path
from django.core.management import execute_from_command_line

# -------------------------
# Django minimal settings
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

settings.configure(
    DEBUG=True,
    SECRET_KEY="super-secret-key",
    ROOT_URLCONF=__name__,
    ALLOWED_HOSTS=["*"],
    MIDDLEWARE=[],
)

# -------------------------
# Generate JWT URL
# -------------------------
def generate_url(request):
    payload = {
        "user_id": 101,
        "exp": dt.datetime.utcnow() + dt.timedelta(minutes=1),
        "iat": dt.datetime.utcnow(),
    }

    token = jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")

    url = f"http://127.0.0.1:8000/secure/?token={token}"

    return JsonResponse({
        "secure_url": url,
        "message": "URL valid for 1 minute"
    })

# -------------------------
# Secure endpoint
# -------------------------
def secure_endpoint(request):
    token = request.GET.get("token")

    if not token:
        return JsonResponse({"error": "Token missing"}, status=400)

    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        return HttpResponse(
            f"Access allowed. User ID: {payload['user_id']}"
        )

    except jwt.ExpiredSignatureError:
        return JsonResponse(
            {"error": "Token expired. URL not valid"},
            status=401
        )

    except jwt.InvalidTokenError:
        return JsonResponse(
            {"error": "Invalid token"},
            status=401
        )

# -------------------------
# URL patterns
# -------------------------
urlpatterns = [
    path("generate/", generate_url),
    path("secure/", secure_endpoint),
]

# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "__main__")
    execute_from_command_line(["manage.py", "runserver"])
