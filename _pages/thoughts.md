---
layout: default
title: Thoughts
---

# thoughts

<ul>
  {% for thought in site.categories['thoughts'] %}
  <li>
    <a href="{{ thought.url }}">{{ thought.title }}</a>
  </li>

  {% endfor %}
</ul>
