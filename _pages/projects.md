---
layout: default
title: Projects
---

# projects

<ul>
  {% for project in site.categories['projects'] %}
      <li>
            <a href="{{ project.url }}">{{ project.title }}</a>
	        </li>

  {% endfor %}
  </ul>
 