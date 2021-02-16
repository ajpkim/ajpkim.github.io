---
layout: default
title: Alex Kim
---


<h1> projects </h1>
{% for project in site.projects %}
  <li><a href="{{ project.url }}">{{ project.title }}</a></li>
  {% endfor %}

<h1> thoughts </h1>
{% for thought in site.thoughts %}
  <li><a href="{{ thought.url }}">{{ thought.title }}</a></li>
  {% endfor %}
