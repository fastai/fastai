{%- extends 'hide.tpl' -%}{% block body %}---
title: {{resources.title}}
keywords: {{resources.keywords}}
sidebar: home_sidebar
tags: {{resources.tags}}
summary: {% if resources.summary != "" and resources.summary != nil %}"{{resources.summary}}"{% endif %}
---

<div class="container" id="notebook-container">
    {{ super()  }}
</div>
{%- endblock body %}
