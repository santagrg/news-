import math
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.contrib import messages
from django.contrib.auth.models import User
from .models import Post
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import json
from newsApp import models, forms


def context_data():
    context = {
        "site_name": "NewsUp",
        "page": "home",
        "page_title": "News Portal",
        "categories": models.Category.objects.filter(status=1).all(),
    }
    return context


# views.py


def home(request):
    context = context_data()
    posts = models.Post.objects.filter(status=1).order_by("-date_created").all()
    context["page"] = "home"
    context["page_title"] = "Home"
    context["latest_top"] = posts[:2]
    context["latest_bottom"] = posts[2:12]

    # Get the search term from the request's GET parameters
    search_term = request.GET.get("search_term", "")

    if search_term:
        # Filter the queryset based on the search term
        queryset = models.Post.objects.filter(title__icontains=search_term)
        context["posts"] = queryset
    else:
        context["posts"] = posts

    return render(request, "home.html", context)


# login
def login_user(request):
    logout(request)
    resp = {"status": "failed", "msg": ""}
    username = ""
    password = ""
    if request.POST:
        username = request.POST["username"]
        password = request.POST["password"]

        user = authenticate(username=username, password=password)
        if user is not None:
            if user.is_active:
                login(request, user)
                resp["status"] = "success"
            else:
                resp["msg"] = "Incorrect username or password"
        else:
            resp["msg"] = "Incorrect username or password"
    return HttpResponse(json.dumps(resp), content_type="application/json")


# Logout
def logoutuser(request):
    logout(request)
    return redirect("/")


@login_required
def update_profile(request):
    context = context_data()
    context["page_title"] = "Update Profile"
    user = User.objects.get(id=request.user.id)
    if not request.method == "POST":
        form = forms.UpdateProfile(instance=user)
        context["form"] = form
        print(form)
    else:
        form = forms.UpdateProfile(request.POST, instance=user)
        if form.is_valid():
            form.save()
            messages.success(request, "Profile has been updated")
            return redirect("profile-page")
        else:
            context["form"] = form

    return render(request, "update_profile.html", context)


@login_required
def update_password(request):
    context = context_data()
    context["page_title"] = "Update Password"
    if request.method == "POST":
        form = forms.UpdatePasswords(user=request.user, data=request.POST)
        if form.is_valid():
            form.save()
            messages.success(
                request, "Your Account Password has been updated successfully"
            )
            update_session_auth_hash(request, form.user)
            return redirect("profile-page")
        else:
            context["form"] = form
    else:
        form = forms.UpdatePasswords(request.POST)
        context["form"] = form
    return render(request, "update_password.html", context)


@login_required
def profile(request):
    context = context_data()
    context["page"] = "profile"
    context["page_title"] = "Profile"
    return render(request, "profile.html", context)


@login_required
def manage_post(request, pk=None):
    context = context_data()
    if not pk is None:
        context["page"] = "edit_post"
        context["page_title"] = "Edit Post"
        context["post"] = models.Post.objects.get(id=pk)
    else:
        context["page"] = "new_post"
        context["page_title"] = "New Post"
        context["post"] = {}

    return render(request, "manage_post.html", context)


@login_required
def save_post(request):
    resp = {"status": "failed", "msg": "", "id": None}
    if request.method == "POST":
        if request.POST["id"] == "":
            form = forms.savePost(request.POST, request.FILES)
        else:
            post = models.Post.objects.get(id=request.POST["id"])
            form = forms.savePost(request.POST, request.FILES, instance=post)

        if form.is_valid():
            form.save()
            if request.POST["id"] == "":
                postID = models.Post.objects.all().last().id
            else:
                postID = request.POST["id"]
            resp["id"] = postID
            resp["status"] = "success"
            messages.success(request, "Post has been saved successfully.")
        else:
            for field in form:
                for error in field.errors:
                    if not resp["msg"] == "":
                        resp["msg"] += str("<br />")
                    resp["msg"] += str(f"[{field.label}] {error}")

    else:
        resp["msg"] = "Request has no data sent."
    return HttpResponse(json.dumps(resp), content_type="application/json")


# BACKUP
# def view_post(request, pk=None):
#     context = context_data()
#     post = models.Post.objects.get(id = pk)
#     context['page'] = 'post'
#     context['page_title'] = post.title
#     context['post'] = post
#     context['latest'] = models.Post.objects.exclude(id=pk).filter(status = 1).order_by('-date_created').all()[:10]
#     context['comments'] = models.Comment.objects.filter(post=post).all()
#     context['actions'] = False
#     if request.user.is_superuser or request.user.id == post.user.id:
#         context['actions'] = True
#     return render(request, 'single_post.html', context)


def view_post(request, pk=None):
    context = context_data()
    post = models.Post.objects.get(id=pk)
    context["page"] = "post"
    context["page_title"] = post.title
    context["post"] = post
    context["latest"] = (
        models.Post.objects.exclude(id=pk)
        .filter(status=1)
        .order_by("-date_created")[:4]
    )
    context["comments"] = models.Comment.objects.filter(post=post).all()
    context["actions"] = False

    if request.user.is_superuser or request.user.id == post.user.id:
        context["actions"] = True

    similar_posts = ml_recommendation_system(post, top_n=4)
    # Retrieve similar posts using ML recommendation system
    if not similar_posts:
        context["no_related_news"] = True
    else:
        context["similar_posts"] = similar_posts

    return render(request, "single_post.html", context)


def ml_recommendation_system(post, top_n=4):
    try:
        # Preprocess and vectorize the text data
        category = post.category

        # posts = models.Post.objects.exclude(pk=post.pk, category=category)
        posts = models.Post.objects.filter(category=category).exclude(pk=post.pk)
        texts = [
            post.title + " " + post.short_description + " " + post.content
            for post in posts
        ]
        vectorizer = TfidfVectorizer()
        feature_vectors = vectorizer.fit_transform(texts)

        # Calculate similarity scores
        post_vector = vectorizer.transform(
            [post.title + " " + post.short_description + " " + post.content]
        )
        similarity_scores = cosine_similarity(post_vector, feature_vectors).flatten()

        # Get the indices of the top similar posts
        top_indices = similarity_scores.argsort()[::-1][:top_n]

        # Retrieve the top similar posts
        similar_posts = [posts[idx] for idx in top_indices.tolist()]

        return similar_posts
    except ValueError as e:
        print(str(e))
        return []


def save_comment(request):
    resp = {"status": "failed", "msg": "", "id": None}
    if request.method == "POST":
        if request.POST["id"] == "":
            form = forms.saveComment(request.POST)
        else:
            comment = models.Comment.objects.get(id=request.POST["id"])
            form = forms.saveComment(request.POST, instance=comment)

        if form.is_valid():
            form.save()
            if request.POST["id"] == "":
                commentID = models.Post.objects.all().last().id
            else:
                commentID = request.POST["id"]
            resp["id"] = commentID
            resp["status"] = "success"
            messages.success(request, "Comment has been saved successfully.")
        else:
            for field in form:
                for error in field.errors:
                    if not resp["msg"] == "":
                        resp["msg"] += str("<br />")
                    resp["msg"] += str(f"[{field.label}] {error}")

    else:
        resp["msg"] = "Request has no data sent."
    return HttpResponse(json.dumps(resp), content_type="application/json")


@login_required
def list_posts(request):
    context = context_data()
    context["page"] = "all_post"
    context["page_title"] = "All Posts"
    if request.user.is_superuser:
        context["posts"] = models.Post.objects.order_by("-date_created").all()
    else:
        context["posts"] = models.Post.objects.filter(user=request.user).all()

    context["latest"] = (
        models.Post.objects.filter(status=1).order_by("-date_created").all()[:10]
    )

    return render(request, "posts.html", context)


def category_posts(request, pk=None):
    context = context_data()
    if pk is None:
        messages.error(request, "File not Found")
        return redirect("home-page")
    try:
        category = models.Category.objects.get(id=pk)
    except:
        messages.error(request, "File not Found")
        return redirect("home-page")

    context["category"] = category
    context["page"] = "category_post"
    context["page_title"] = f"{category.name} Posts"
    context["posts"] = models.Post.objects.filter(status=1, category=category).all()

    context["latest"] = (
        models.Post.objects.filter(status=1).order_by("-date_created").all()[:10]
    )

    return render(request, "category.html", context)


@login_required
def delete_post(request, pk=None):
    resp = {"status": "failed", "msg": ""}
    if pk is None:
        resp["msg"] = "Post ID is Invalid"
        return HttpResponse(json.dumps(resp), content_type="application/json")
    try:
        post = models.Post.objects.get(id=pk)
        post.delete()
        messages.success(request, "Post has been deleted successfully.")
        resp["status"] = "success"
    except:
        resp["msg"] = "Post ID is Invalid"

    return HttpResponse(json.dumps(resp), content_type="application/json")


@login_required
def delete_comment(request, pk=None):
    resp = {"status": "failed", "msg": ""}
    if pk is None:
        resp["msg"] = "Comment ID is Invalid"
        return HttpResponse(json.dumps(resp), content_type="application/json")
    try:
        comment = models.Comment.objects.get(id=pk)
        comment.delete()
        messages.success(request, "Comment has been deleted successfully.")
        resp["status"] = "success"
    except:
        resp["msg"] = "Comment ID is Invalid"

    return HttpResponse(json.dumps(resp), content_type="application/json")